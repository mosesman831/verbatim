"""V2 ingest drain: every JobKind executes, fenced commits, op receipts.

Covers the SPEC_V2 §39 drain contract: all ten job kinds run through
``Ingester.run_pending`` — harvest/admit/compare/embed on the ordinary lane,
purge/reindex/review_apply on the reserved control lane (drained strictly
first), and deterministic record-only handlers for episode_index /
procedure_validate / replay. Each handler commits domain effects, the
operation receipt, and the queue completion inside ONE generation-fenced
transaction: a worker holding a superseded lease is fenced out
(``assert_lease``), and a redelivered job replays its recorded receipt
instead of reapplying effects (V2-39.10).

Schema note: the shipped ``jobs.kind`` CHECK lists only the seven original
kinds, so all-kinds coverage runs on ``V2Store`` (imported from
``tests.jobs.test_v2_jobs``) — the same DDL with the CHECK widened. The
``open_store`` test proves the real store DDL already carries the
``operation_key``/``lane`` columns the drain relies on.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Optional

import pytest

from tests.jobs.test_v2_jobs import V2Store
from verbatim.config import VerbatimConfig
from verbatim.core.lifecycle import LifecycleMachine
from verbatim.core.time import now_us
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
    TransitionCommand,
    VerbatimError,
    safe_json_loads,
)
from verbatim.embeddings.codec import Float32Codec
from verbatim.ingest import Ingester
from verbatim.purge import plan_purge, suppress
from verbatim.storage.repos import FtsRepo, ReviewsRepo

PAYLOAD = "My editor is neovim."


# --------------------------------------------------------------------------
# fixtures + fakes
# --------------------------------------------------------------------------


def _cfg() -> VerbatimConfig:
    cfg = VerbatimConfig()
    return replace(cfg, capture=replace(cfg.capture, enabled=True))


@pytest.fixture()
def v2store() -> V2Store:
    return V2Store()


class FakeEncoder:
    """Deterministic encoder: 4-dim float32 blobs derived from the text."""

    encoder_dims = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def encoder_id(self) -> str:
        return "fake:test:v1"

    @property
    def dimensions(self) -> int:
        return self.encoder_dims

    @property
    def normalization(self) -> str:
        return "l2"

    def available(self) -> bool:
        return True

    def encode(self, texts: list[str]) -> list[bytes]:
        self.calls.append(list(texts))
        return [
            Float32Codec.pack([0.5, 0.25, 0.125, float(len(t))])
            for t in texts
        ]

    def manifest(self) -> dict[str, Any]:
        return {
            "dimensions": self.encoder_dims,
            "normalization": self.normalization,
            "artifact_revision": "r1",
            "preprocessing_version": "p1",
        }


class BadVectorEncoder(FakeEncoder):
    """Declares 4 dims but returns 1-float blobs — the codec gate must
    reject the whole batch before any row persists."""

    def encode(self, texts: list[str]) -> list[bytes]:
        self.calls.append(list(texts))
        return [Float32Codec.pack([0.5]) for _ in texts]


class TxGuard:
    """Store delegate exposing whether a write tx is open — the encoder
    asserts inference never runs inside one (V2-39.07)."""

    def __init__(self, inner: V2Store) -> None:
        self._inner = inner
        self.in_write_tx = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @contextmanager
    def tx(self):
        with self._inner.tx() as conn:
            self.in_write_tx = True
            try:
                yield conn
            finally:
                self.in_write_tx = False


class TxGuardEncoder(FakeEncoder):
    def __init__(self, guard: TxGuard) -> None:
        super().__init__()
        self._guard = guard
        self.encode_inside_tx = False

    def encode(self, texts: list[str]) -> list[bytes]:
        if self._guard.in_write_tx:
            self.encode_inside_tx = True
        return super().encode(texts)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _env(scope: Scope, text: str = PAYLOAD) -> SourceEnvelope:
    return SourceEnvelope(
        origin="test:v2",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id=scope.principal_id,
        payload=text.encode("utf-8"),
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
    )


def _enqueue(ing: Ingester, scope_id: str, kind: JobKind,
             refs: dict[str, Any], **kw: Any) -> str:
    with ing.store.tx() as conn:
        return ing.jobs.enqueue(conn, scope_id, kind, refs, **kw)


def _job(store: Any, job_id: str) -> dict[str, Any]:
    with store.read() as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
    return dict(zip(cols, row))


def _jobs(store: Any) -> list[dict[str, Any]]:
    with store.read() as conn:
        cur = conn.execute("SELECT * FROM jobs ORDER BY rowid")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _event_kinds(store: Any) -> list[str]:
    with store.read() as conn:
        return [
            r[0]
            for r in conn.execute("SELECT kind FROM events ORDER BY event_seq")
        ]


def _event_payloads(store: Any, kind: str) -> list[dict[str, Any]]:
    with store.read() as conn:
        return [
            safe_json_loads(r[0])
            for r in conn.execute(
                "SELECT payload_json FROM events WHERE kind = ?"
                " ORDER BY event_seq",
                (kind,),
            )
        ]


def _head_state(store: Any, claim_id: str) -> Optional[str]:
    with store.read() as conn:
        row = conn.execute(
            "SELECT state FROM claim_revisions"
            " WHERE claim_id = ? AND recorded_until IS NULL",
            (claim_id,),
        ).fetchone()
    return row[0] if row else None


def _first_ids(store: Any) -> tuple[str, str, str, str]:
    """(source_id, span_id, claim_id, open review_id) for a seeded scope."""
    with store.read() as conn:
        src = conn.execute("SELECT source_id FROM sources LIMIT 1").fetchone()[0]
        span = conn.execute("SELECT span_id FROM spans LIMIT 1").fetchone()[0]
        claim = conn.execute("SELECT claim_id FROM claims LIMIT 1").fetchone()[0]
        review = conn.execute(
            "SELECT review_id FROM reviews WHERE state = 'open' LIMIT 1"
        ).fetchone()[0]
    return src, span, claim, review


def _seed_pending(
    ing: Ingester, scope: Scope, text: str = PAYLOAD
) -> tuple[str, str, str, str]:
    """Ingest + drain harvest/admit → (source_id, span_id, claim_id, review_id)."""
    receipt = ing.ingest(_env(scope, text))
    source_id = receipt.accepted[0]
    ing.run_pending(scope=scope)
    with ing.store.read() as conn:
        span_id = conn.execute(
            "SELECT span_id FROM spans ORDER BY rowid LIMIT 1"
        ).fetchone()[0]
        claim_id = conn.execute(
            "SELECT claim_id FROM claims ORDER BY rowid LIMIT 1"
        ).fetchone()[0]
        review_id = conn.execute(
            "SELECT review_id FROM reviews WHERE state = 'open'"
            " ORDER BY rowid LIMIT 1"
        ).fetchone()[0]
    return source_id, span_id, claim_id, review_id


def _activate(store: Any, claim_id: str) -> None:
    """Drive a pending claim ACTIVE through the lifecycle machine."""
    machine = LifecycleMachine(store)
    with store.tx() as conn:
        machine.apply(
            TransitionCommand(
                claim_id=claim_id,
                expected_revision=1,
                effect="admit",
                actor_id="op",
                reason="approved",
            ),
            conn,
        )


# --------------------------------------------------------------------------
# harvest/admit chain + receipts
# --------------------------------------------------------------------------


def test_harvest_admit_chain_drains(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    receipt = ing.ingest(_env(scope))
    assert receipt.accepted and receipt.job_ids

    assert ing.run_pending(scope=scope) == 2

    kinds = [(j["kind"], j["state"]) for j in _jobs(v2store)]
    assert kinds == [("harvest", "succeeded"), ("admit", "succeeded")]
    src, span, claim, review = _first_ids(v2store)
    assert _head_state(v2store, claim) == "pending"
    # Operation receipts committed for both jobs (V2-39.10).
    with v2store.read() as conn:
        keys = {
            r[0]
            for r in conn.execute("SELECT operation_key FROM operations")
        }
    assert keys == {f"harvest:{src}:1", f"admit:{span}"}
    for kind in ("source_accepted", "harvested", "claim_proposed", "admitted"):
        assert kind in _event_kinds(v2store)


def test_kinds_filter_limits_drain(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    ing.ingest(_env(scope))
    assert ing.run_pending(scope=scope, kinds=[JobKind.HARVEST]) == 1
    states = {j["kind"]: j["state"] for j in _jobs(v2store)}
    assert states == {"harvest": "succeeded", "admit": "queued"}
    assert ing.run_pending(scope=scope, kinds=[JobKind.ADMIT]) == 1
    assert all(j["state"] == "succeeded" for j in _jobs(v2store))


def test_operation_key_replay_skips_effects(v2store, scope, scope_id):
    """A redelivered job whose receipt committed replays it — no effects."""
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    with v2store.read() as conn:
        n_spans = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]

    jid = _enqueue(
        ing, scope_id, JobKind.HARVEST,
        {"source_id": src, "revision": 1},
        operation_key=f"harvest:{src}:1",
    )
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid)["state"] == "succeeded"
    with v2store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == n_spans
    # exactly one harvest event — the receipt replay added none
    assert _event_kinds(v2store).count("harvested") == 1


def test_operation_key_conflict_fails_terminally(v2store, scope, scope_id):
    """Same operation_key + different input digest = a real conflict."""
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    with v2store.read() as conn:
        n_spans = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]

    jid = _enqueue(
        ing, scope_id, JobKind.HARVEST,
        {"source_id": src, "revision": 1, "note": "different-input"},
        operation_key=f"harvest:{src}:1",
    )
    assert ing.run_pending(scope=scope) == 1
    job = _job(v2store, jid)
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.VALIDATION.value
    with v2store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == n_spans


def test_deadline_exceeded_fails_without_effects(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    ing.ingest(_env(scope))
    with v2store.tx() as conn:
        conn.execute("UPDATE jobs SET deadline_us = 1 WHERE kind = 'harvest'")
    assert ing.run_pending(scope=scope) == 1
    job = _jobs(v2store)[0]
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.DEADLINE_EXCEEDED.value
    with v2store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 0
        ev = conn.execute(
            "SELECT state, error_code FROM job_events WHERE job_id = ?"
            " ORDER BY event_seq",
            (job["job_id"],),
        ).fetchall()
    assert ("failed", ErrorCode.DEADLINE_EXCEEDED.value) in [
        (r[0], r[1]) for r in ev
    ]


def test_malformed_input_refs_fail_validation(v2store, scope, scope_id):
    """A job whose refs lack required keys fails terminally — never a
    silent no-op (V2-39)."""
    ing = Ingester(v2store, _cfg())
    jid = _enqueue(ing, scope_id, JobKind.PURGE, {"not_a_key": 1})
    assert ing.run_pending(scope=scope) == 1
    job = _job(v2store, jid)
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.VALIDATION.value


# --------------------------------------------------------------------------
# embed
# --------------------------------------------------------------------------


def test_embed_writes_rows_event_and_inputs(v2store, scope, scope_id):
    enc = FakeEncoder()
    ing = Ingester(v2store, _cfg(), encoder=enc)
    src, span, claim, review = _seed_pending(ing, scope)
    jid = _enqueue(
        ing, scope_id, JobKind.EMBED,
        {"span_ids": [span], "encoder_id": enc.encoder_id},
        operation_key="op:embed:1",
    )
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid)["state"] == "succeeded"
    # The encoder received exactly the persisted span bytes.
    with v2store.read() as conn:
        sb, eb, payload = conn.execute(
            "SELECT sp.start_byte, sp.end_byte, sr.payload FROM spans sp"
            " JOIN source_revisions sr ON sr.source_id = sp.source_id"
            "   AND sr.revision = sp.revision WHERE sp.span_id = ?",
            (span,),
        ).fetchone()
    expected = bytes(payload)[sb:eb].decode("utf-8")
    assert enc.calls == [[expected]]
    with v2store.read() as conn:
        emb = conn.execute(
            "SELECT span_id, encoder_id, dimensions, dtype FROM embeddings"
        ).fetchone()
        assert emb == (span, enc.encoder_id, 4, "float32le")
        assert conn.execute(
            "SELECT 1 FROM encoder_manifests WHERE encoder_id = ?",
            (enc.encoder_id,),
        ).fetchone()
        assert conn.execute(
            "SELECT 1 FROM embedding_inputs WHERE span_id = ? AND encoder_id = ?",
            (span, enc.encoder_id),
        ).fetchone()
    ev = _event_payloads(v2store, "embedded")
    assert ev and ev[0]["encoder_id"] == enc.encoder_id and ev[0]["encoded"] == 1


def test_embed_inference_never_inside_write_tx(v2store, scope, scope_id):
    guard = TxGuard(v2store)
    enc = TxGuardEncoder(guard)
    ing = Ingester(guard, _cfg(), encoder=enc)
    src, span, claim, review = _seed_pending(ing, scope)
    _enqueue(
        ing, scope_id, JobKind.EMBED,
        {"span_ids": [span], "encoder_id": enc.encoder_id},
    )
    ing.run_pending(scope=scope)
    assert enc.calls, "encoder never ran"
    assert enc.encode_inside_tx is False


def test_embed_gather_indexes_active_claim_spans(v2store, scope, scope_id):
    """span_ids omitted → gather primary evidence of ACTIVE claims lacking
    a row for this encoder."""
    enc = FakeEncoder()
    ing = Ingester(v2store, _cfg(), encoder=enc)
    src, span, claim, review = _seed_pending(ing, scope)
    _activate(v2store, claim)  # pending → active, evidence carries forward
    jid = _enqueue(ing, scope_id, JobKind.EMBED, {})
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid)["state"] == "succeeded"
    with v2store.read() as conn:
        assert conn.execute(
            "SELECT 1 FROM embeddings WHERE span_id = ? AND encoder_id = ?",
            (span, enc.encoder_id),
        ).fetchone()


def test_embed_pending_claims_are_not_gathered(v2store, scope, scope_id):
    """Gather mode authorizes ACTIVE claims only — pending work is skipped
    and the job completes with a recorded note, not a wedge."""
    enc = FakeEncoder()
    ing = Ingester(v2store, _cfg(), encoder=enc)
    _seed_pending(ing, scope)  # claim stays pending
    jid = _enqueue(ing, scope_id, JobKind.EMBED, {})
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid)["state"] == "succeeded"
    assert enc.calls == []  # nothing authorized to encode
    assert _event_payloads(v2store, "embed_skipped")


def test_embed_scopes_are_reverified(v2store, scope, scope_id):
    """An explicit span_id from a different scope is never encoded — the
    job completes with a note instead of crossing the boundary."""
    enc = FakeEncoder()
    ing = Ingester(v2store, _cfg(), encoder=enc)
    src, span, claim, review = _seed_pending(ing, scope)
    other = Scope(profile_id="prof", principal_id="bob", conversation_id="c9")
    r2 = ing.ingest(_env(other, "Bob uses emacs."))
    ing.run_pending(scope=other)
    with v2store.read() as conn:
        foreign_span = conn.execute(
            "SELECT span_id FROM spans WHERE source_id = ?",
            (r2.accepted[0],),
        ).fetchone()[0]
    jid = _enqueue(
        ing, scope_id, JobKind.EMBED,
        {"span_ids": [foreign_span], "encoder_id": enc.encoder_id},
    )
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid)["state"] == "succeeded"
    with v2store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE span_id = ?",
            (foreign_span,),
        ).fetchone()[0] == 0


def test_embed_without_encoder_completes_degraded(v2store, scope, scope_id):
    """Capability disabled is a visible outcome, not a wedge or a crash."""
    ing = Ingester(v2store, _cfg(), encoder=None)
    src, span, claim, review = _seed_pending(ing, scope)
    jid = _enqueue(
        ing, scope_id, JobKind.EMBED,
        {"span_ids": [span], "encoder_id": "fake:test:v1"},
    )
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid)["state"] == "succeeded"
    ev = _event_payloads(v2store, "embed_skipped")
    assert ev and ev[0]["reason"] == "encoder_unavailable"
    with v2store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0


def test_embed_wrong_encoder_id_fails(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg(), encoder=FakeEncoder())
    src, span, claim, review = _seed_pending(ing, scope)
    jid = _enqueue(
        ing, scope_id, JobKind.EMBED,
        {"span_ids": [span], "encoder_id": "other:encoder:v9"},
    )
    assert ing.run_pending(scope=scope) == 1
    job = _job(v2store, jid)
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.VALIDATION.value


def test_embed_bad_vectors_fail_without_partial_writes(v2store, scope, scope_id):
    enc = BadVectorEncoder()
    ing = Ingester(v2store, _cfg(), encoder=enc)
    src, span, claim, review = _seed_pending(ing, scope)
    jid = _enqueue(
        ing, scope_id, JobKind.EMBED,
        {"span_ids": [span], "encoder_id": enc.encoder_id},
    )
    assert ing.run_pending(scope=scope) == 1
    job = _job(v2store, jid)
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.VECTOR_INVALID.value
    with v2store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0


def test_embed_replay_does_not_reencode(v2store, scope, scope_id):
    enc = FakeEncoder()
    ing = Ingester(v2store, _cfg(), encoder=enc)
    src, span, claim, review = _seed_pending(ing, scope)
    refs = {"span_ids": [span], "encoder_id": enc.encoder_id}
    _enqueue(ing, scope_id, JobKind.EMBED, refs, operation_key="op:emb:x")
    ing.run_pending(scope=scope)
    assert len(enc.calls) == 1
    jid2 = _enqueue(ing, scope_id, JobKind.EMBED, refs, operation_key="op:emb:x")
    ing.run_pending(scope=scope)
    assert _job(v2store, jid2)["state"] == "succeeded"
    assert len(enc.calls) == 1  # receipt replay — inference did not re-run


# --------------------------------------------------------------------------
# review_apply
# --------------------------------------------------------------------------


def test_review_apply_admit_activates_claim(v2store, scope, scope_id):
    enc = FakeEncoder()
    ing = Ingester(v2store, _cfg(), encoder=enc)
    src, span, claim, review = _seed_pending(ing, scope)
    jid = _enqueue(
        ing, scope_id, JobKind.REVIEW_APPLY, {"review_id": review},
        operation_key=f"op:review:{review}",
    )
    # The apply also drains the embed obligation it enqueues atomically.
    assert ing.run_pending(scope=scope) >= 1
    assert _job(v2store, jid)["state"] == "succeeded"
    assert _head_state(v2store, claim) == "active"
    with v2store.read() as conn:
        assert conn.execute(
            "SELECT state FROM reviews WHERE review_id = ?", (review,)
        ).fetchone()[0] == "approved"
        # Activation indexed the claim under the current projection
        # generation — recall reads only indexed rows.
        assert conn.execute(
            "SELECT 1 FROM fts_rows WHERE claim_id = ?", (claim,)
        ).fetchone()
    kinds = _event_kinds(v2store)
    assert "claim_transition" in kinds and "review_applied" in kinds
    # The embed obligation was queued inside the same commit tx.
    assert any(j["kind"] == "embed" for j in _jobs(v2store))


def test_review_apply_stale_fence_leaves_review_open(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    with v2store.tx() as conn:
        stale_review = ReviewsRepo(v2store).create(
            conn, scope_id,
            {"effect": "admit", "claim_id": claim, "reason": "stale"},
            {claim: 999},  # expected revision can never match
        )
    jid = _enqueue(
        ing, scope_id, JobKind.REVIEW_APPLY, {"review_id": stale_review}
    )
    assert ing.run_pending(scope=scope) == 1
    # Completes: a permanently stale review is a recorded outcome, not a
    # retryable error — and the review stays open for re-triage.
    assert _job(v2store, jid)["state"] == "succeeded"
    with v2store.read() as conn:
        assert conn.execute(
            "SELECT state FROM reviews WHERE review_id = ?", (stale_review,)
        ).fetchone()[0] == "open"
    assert _event_payloads(v2store, "review_apply_stale")
    assert _head_state(v2store, claim) == "pending"


def test_review_apply_missing_review_fails(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    _seed_pending(ing, scope)
    jid = _enqueue(
        ing, scope_id, JobKind.REVIEW_APPLY, {"review_id": "no-such-review"}
    )
    assert ing.run_pending(scope=scope) == 1
    job = _job(v2store, jid)
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.NOT_FOUND_OR_FORBIDDEN.value


def test_review_apply_resolved_review_is_idempotent(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    _enqueue(ing, scope_id, JobKind.REVIEW_APPLY, {"review_id": review})
    ing.run_pending(scope=scope)
    assert _head_state(v2store, claim) == "active"
    # A duplicated apply job completes without re-transitioning.
    jid2 = _enqueue(ing, scope_id, JobKind.REVIEW_APPLY, {"review_id": review})
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid2)["state"] == "succeeded"
    with v2store.read() as conn:
        assert conn.execute(
            "SELECT state FROM reviews WHERE review_id = ?", (review,)
        ).fetchone()[0] == "approved"
    assert _event_kinds(v2store).count("claim_transition") == 1


# --------------------------------------------------------------------------
# purge
# --------------------------------------------------------------------------


def test_purge_executes_end_to_end(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    preview = plan_purge(
        v2store, scope,
        [("claim", claim), ("span", span), ("source", src)],
        "operator",
    )
    purge_id = preview["purge_id"]
    assert preview["state"] == "previewed"
    jid = _enqueue(ing, scope_id, JobKind.PURGE, {"purge_id": purge_id})
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid)["state"] == "succeeded"

    with v2store.read() as conn:
        assert conn.execute(
            "SELECT state FROM purges WHERE purge_id = ?", (purge_id,)
        ).fetchone()[0] == "completed"
        assert conn.execute(
            "SELECT payload FROM source_revisions WHERE source_id = ?",
            (src,),
        ).fetchone()[0] == b""
        kinds = {
            r[0]
            for r in conn.execute(
                "SELECT object_kind FROM erasure_ledger WHERE purge_id = ?",
                (purge_id,),
            )
        }
        assert {"claim", "span", "source"} <= kinds
        # The ledger stores digests — erased ids/bytes never appear raw.
        digests = [
            bytes(r[0])
            for r in conn.execute(
                "SELECT object_digest FROM erasure_ledger WHERE purge_id = ?",
                (purge_id,),
            )
        ]
        for raw in (claim.encode(), span.encode(), src.encode(), PAYLOAD.encode()):
            assert all(raw not in d for d in digests)
    assert _head_state(v2store, claim) == "erased"
    kinds = _event_kinds(v2store)
    assert "purged" in kinds and "purge_executed" in kinds


def test_purge_from_suppressed_state(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    preview = suppress(v2store, scope, [("claim", claim)], "operator")
    jid = _enqueue(ing, scope_id, JobKind.PURGE, {"purge_id": preview["purge_id"]})
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid)["state"] == "succeeded"
    assert _head_state(v2store, claim) == "erased"


def test_purge_cancels_jobs_referencing_erased_objects(v2store, scope, scope_id):
    """A queued job that names a purged span is cancelled by the erasure
    (V2-41.08) — it can never recreate erased derivatives."""
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    doomed = _enqueue(
        ing, scope_id, JobKind.COMPARE,
        {"span_id": span, "source_id": src, "revision": 1},
    )
    preview = plan_purge(v2store, scope, [("span", span)], "operator")
    jid = _enqueue(ing, scope_id, JobKind.PURGE, {"purge_id": preview["purge_id"]})
    ing.run_pending(scope=scope)
    assert _job(v2store, doomed)["state"] == "cancelled"
    assert _job(v2store, jid)["state"] == "succeeded"


def test_purge_completed_replay_is_safe(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    preview = plan_purge(v2store, scope, [("claim", claim)], "operator")
    _enqueue(ing, scope_id, JobKind.PURGE, {"purge_id": preview["purge_id"]})
    ing.run_pending(scope=scope)
    jid2 = _enqueue(ing, scope_id, JobKind.PURGE, {"purge_id": preview["purge_id"]})
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid2)["state"] == "succeeded"


def test_purge_missing_id_fails(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    _seed_pending(ing, scope)
    jid = _enqueue(ing, scope_id, JobKind.PURGE, {"purge_id": "nope"})
    assert ing.run_pending(scope=scope) == 1
    job = _job(v2store, jid)
    assert job["state"] == "failed"
    assert job["error_code"] == ErrorCode.NOT_FOUND_OR_FORBIDDEN.value


# --------------------------------------------------------------------------
# reindex
# --------------------------------------------------------------------------


def test_reindex_rebuilds_fts_under_new_generation(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    _activate(v2store, claim)
    before = v2store.projection_generation()
    jid = _enqueue(ing, scope_id, JobKind.REINDEX, {})
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, jid)["state"] == "succeeded"
    after = v2store.projection_generation()
    assert after == before + 1
    with v2store.read() as conn:
        row = conn.execute(
            "SELECT projection_generation FROM fts_rows WHERE claim_id = ?",
            (claim,),
        ).fetchone()
        assert row and row[0] == after
        assert conn.execute(
            "SELECT 1 FROM active_projections"
            " WHERE scope_partition = ? AND projection_kind = 'fts'",
            (scope_id,),
        ).fetchone()
    ev = _event_payloads(v2store, "reindexed")
    assert ev and ev[0]["generation"] == after and ev[0]["claims"] == 1


def test_reindex_drops_inactive_claims(v2store, scope, scope_id):
    """Rebuild keeps only live ACTIVE heads: the pending claim's stale
    projection row dies with the old generation."""
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    # Simulate a stale row for a claim that is NOT active.
    with v2store.tx() as conn:
        FtsRepo(v2store).index(
            conn, claim, 1, scope_id,
            v2store.projection_generation(), "stale text",
        )
    jid = _enqueue(ing, scope_id, JobKind.REINDEX, {})
    ing.run_pending(scope=scope)
    assert _job(v2store, jid)["state"] == "succeeded"
    with v2store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM fts_rows WHERE claim_id = ?", (claim,)
        ).fetchone()[0] == 0


# --------------------------------------------------------------------------
# lanes
# --------------------------------------------------------------------------


def test_control_lane_drains_before_ordinary(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg(), encoder=FakeEncoder())
    src, span, claim, review = _seed_pending(ing, scope)
    emb = _enqueue(
        ing, scope_id, JobKind.EMBED,
        {"span_ids": [span], "encoder_id": "fake:test:v1"},
    )
    rex = _enqueue(ing, scope_id, JobKind.REINDEX, {})
    assert ing.run_pending(scope=scope, limit=1) == 1
    assert _job(v2store, rex)["state"] == "succeeded"
    assert _job(v2store, emb)["state"] == "queued"
    assert ing.run_pending(scope=scope) == 1
    assert _job(v2store, emb)["state"] == "succeeded"


def test_lane_filter_drains_one_lane(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg(), encoder=FakeEncoder())
    src, span, claim, review = _seed_pending(ing, scope)
    emb = _enqueue(
        ing, scope_id, JobKind.EMBED,
        {"span_ids": [span], "encoder_id": "fake:test:v1"},
    )
    rex = _enqueue(ing, scope_id, JobKind.REINDEX, {})
    assert ing.run_pending(scope=scope, lane="ordinary") == 1
    assert _job(v2store, emb)["state"] == "succeeded"
    assert _job(v2store, rex)["state"] == "queued"
    assert ing.run_pending(scope=scope, lane="control") == 1
    assert _job(v2store, rex)["state"] == "succeeded"


def test_unknown_lane_rejected(v2store, scope, scope_id):
    ing = Ingester(v2store, _cfg())
    with pytest.raises(VerbatimError) as ei:
        ing.run_pending(scope=scope, lane="vip")
    assert ei.value.code == ErrorCode.VALIDATION


# --------------------------------------------------------------------------
# fencing
# --------------------------------------------------------------------------


def test_stale_worker_cannot_commit_admit(v2store, scope, scope_id):
    """Cancel a leased admit job → the staged handler raises LEASE_LOST
    before any domain write, and no claim is created."""
    ing = Ingester(v2store, _cfg())
    ing.ingest(_env(scope))
    ing.run_pending(scope=scope, kinds=[JobKind.HARVEST])
    leased = ing.jobs.lease(scope_id, [JobKind.ADMIT], owner="w1")[0]
    assert ing.jobs.cancel(leased["job_id"]) is True
    with pytest.raises(VerbatimError) as ei:
        ing._execute(leased, "w1")
    assert ei.value.code == ErrorCode.LEASE_LOST
    with v2store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0


def test_stale_worker_cannot_commit_purge(v2store, scope, scope_id):
    """The fenced commit rolls the whole effect tx back: a cancelled
    worker's purge leaves the preview and all payloads untouched."""
    ing = Ingester(v2store, _cfg())
    src, span, claim, review = _seed_pending(ing, scope)
    preview = plan_purge(v2store, scope, [("claim", claim)], "operator")
    jid = _enqueue(ing, scope_id, JobKind.PURGE, {"purge_id": preview["purge_id"]})
    leased = ing.jobs.lease(scope_id, [JobKind.PURGE], owner="w1")[0]
    assert ing.jobs.cancel(jid) is True
    with pytest.raises(VerbatimError) as ei:
        ing._execute(leased, "w1")
    assert ei.value.code == ErrorCode.LEASE_LOST
    with v2store.read() as conn:
        assert conn.execute(
            "SELECT state FROM purges WHERE purge_id = ?",
            (preview["purge_id"],),
        ).fetchone()[0] == "previewed"
        assert conn.execute(
            "SELECT COUNT(*) FROM erasure_ledger"
        ).fetchone()[0] == 0
    assert _head_state(v2store, claim) == "pending"


# --------------------------------------------------------------------------
# all kinds drain (v2 CHECK widened store)
# --------------------------------------------------------------------------


def test_every_registered_kind_drains_to_terminal(v2store, scope, scope_id):
    """All ten JobKinds execute through one drain — none wedge.

    PURGE runs in a second pass on purpose: it sorts first inside the
    control lane, so co-enqueueing it would erase the fixture before the
    other handlers execute.
    """
    enc = FakeEncoder()
    ing = Ingester(v2store, _cfg(), encoder=enc)
    src, span, claim, review = _seed_pending(ing, scope)

    _enqueue(ing, scope_id, JobKind.REVIEW_APPLY, {"review_id": review})
    _enqueue(ing, scope_id, JobKind.REINDEX, {})
    _enqueue(
        ing, scope_id, JobKind.EMBED,
        {"span_ids": [span], "encoder_id": enc.encoder_id},
    )
    _enqueue(ing, scope_id, JobKind.EPISODE_INDEX, {"episode_id": "ep-1"})
    _enqueue(ing, scope_id, JobKind.PROCEDURE_VALIDATE, {"procedure_id": "p-1"})
    _enqueue(ing, scope_id, JobKind.REPLAY, {"since_seq": 0})
    # COMPARE is a declared kind without a handler: it drains to a
    # terminal failure (capability_unavailable), never silently routed to
    # admission. Asserted explicitly below — "terminal" includes the
    # honest failure, not a wedge.
    _enqueue(
        ing, scope_id, JobKind.COMPARE,
        {"span_id": span, "source_id": src, "revision": 1},
        operation_key=f"compare:{span}",
    )
    # A second HARVEST under the committed op key replays — no dup spans.
    _enqueue(
        ing, scope_id, JobKind.HARVEST,
        {"source_id": src, "revision": 1},
        operation_key=f"harvest:{src}:1",
    )

    drained = ing.run_pending(scope=scope, limit=64)
    assert drained >= 8  # review_apply may enqueue an extra embed job
    for j in _jobs(v2store):
        if j["kind"] == JobKind.COMPARE.value:
            # Declared kind without a handler fails loudly and
            # terminally — not silently routed to admission, not wedged.
            assert j["state"] == "failed", j
            assert j["error_code"] == ErrorCode.CAPABILITY_UNAVAILABLE.value
            continue
        assert j["state"] == "succeeded", j

    kinds = _event_kinds(v2store)
    for expected in (
        "review_applied", "reindexed", "embedded",
        "episode_index_requested", "procedure_validate_requested",
        "replay_requested",
    ):
        assert expected in kinds, expected

    # Phase 2: the privacy obligation drains last here — erasure rewrites
    # the fixture the other handlers just exercised.
    preview = plan_purge(v2store, scope, [("claim", claim)], "operator")
    jid = _enqueue(ing, scope_id, JobKind.PURGE, {"purge_id": preview["purge_id"]})
    assert ing.run_pending(scope=scope) >= 1
    assert _job(v2store, jid)["state"] == "succeeded"
    kinds = _event_kinds(v2store)
    assert "purged" in kinds and "purge_executed" in kinds
    assert _head_state(v2store, claim) == "erased"

    with v2store.read() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) FROM jobs"
            " WHERE state IN ('queued','leased','retry_wait')"
        ).fetchone()[0]
    assert pending == 0


def test_record_only_kinds_emit_observable_events(v2store, scope, scope_id):
    """Deterministic handlers for kinds without domain services record the
    request — never a silent no-op."""
    ing = Ingester(v2store, _cfg())
    _seed_pending(ing, scope)
    for kind, refs in (
        (JobKind.EPISODE_INDEX, {"episode_id": "ep-9"}),
        (JobKind.PROCEDURE_VALIDATE, {"procedure_id": "proc-9"}),
        (JobKind.REPLAY, {"since_seq": 3}),
    ):
        _enqueue(ing, scope_id, kind, refs)
    assert ing.run_pending(scope=scope) == 3
    payloads = {
        k: _event_payloads(v2store, f"{k.value}_requested") for k in
        (JobKind.EPISODE_INDEX, JobKind.PROCEDURE_VALIDATE, JobKind.REPLAY)
    }
    assert payloads[JobKind.EPISODE_INDEX][0]["input_refs"] == {"episode_id": "ep-9"}
    assert payloads[JobKind.PROCEDURE_VALIDATE][0]["input_refs"] == {"procedure_id": "proc-9"}
    assert payloads[JobKind.REPLAY][0]["input_refs"] == {"since_seq": 3}


# --------------------------------------------------------------------------
# real store + v1-schema degradation
# --------------------------------------------------------------------------


def test_real_store_drains_with_durability(tmp_path):
    """The real on-disk schema carries operation_key/lane — receipts land."""
    from verbatim.api import open_store
    from verbatim.host import LocalHost

    eng = open_store(
        str(tmp_path), _cfg(),
        LocalHost(profile_id="p", principal_id="me", conversation_id="c1"),
        create=True, encoder=FakeEncoder(),
    )
    try:
        assert eng._ingester.jobs.supports_durability is True
        receipt = eng.ingest(_env(eng.host.default_scope()))
        assert eng.run_pending() >= 2
        store = eng.store
        with store.read() as conn:
            states = {r[0] for r in conn.execute("SELECT state FROM jobs")}
            assert states == {"succeeded"}
            n_ops = conn.execute("SELECT COUNT(*) FROM operations").fetchone()[0]
            assert n_ops >= 2
            assert conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 1
    finally:
        eng.close()


def test_ingester_degrades_on_v1_schema(store, scope, scope_id):
    """Pre-migration store: durability features flag off rather than write
    to missing tables; plain enqueue still works, v2 args are refused."""
    ing = Ingester(store, _cfg())
    assert ing.jobs.supports_durability is False
    assert ing.ops is None
    assert ing.embedding_inputs is None
    assert ing.projections is None
    with store.tx() as conn:
        jid = ing.jobs.enqueue(conn, scope_id, JobKind.COMPARE, {"a": 1})
    with store.read() as conn:
        row = conn.execute(
            "SELECT kind, state FROM jobs WHERE job_id = ?", (jid,)
        ).fetchone()
        assert row == ("compare", "queued")
    with store.tx() as conn, pytest.raises(VerbatimError) as ei:
        ing.jobs.enqueue(
            conn, scope_id, JobKind.COMPARE, {}, operation_key="op:x"
        )
    assert ei.value.code == ErrorCode.SCHEMA_UNSUPPORTED
