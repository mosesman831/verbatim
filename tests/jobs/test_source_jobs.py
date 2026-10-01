"""V5 source-projection jobs: real on-disk stores, no mocks.

Covers ``source_project`` (lexical projection + FTS shadow + postings +
enrichment + dedup links + update candidates), ``source_embed``
(hashing vector + codec gate), and ``source_backfill`` (bounded resumable
cursor scan) against ``Store.create`` databases. Dispatch wiring is the
integration session's seam — these tests lease jobs through the real
``JobQueue`` and invoke the registered handlers directly, which is the
same ``(job, owner, ingester)`` contract the dispatcher uses.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.time import now_us, rfc3339
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    Provenance,
    Scope,
    SourceKind,
    VerbatimError,
)
from verbatim.core.types_v4 import CapabilityName
from verbatim.embeddings.codec import Float32Codec
from verbatim.embeddings.hashing import HashingEncoder
from verbatim.enrichment import normalize_text, normalized_digest
from verbatim.ingest import Ingester, SourceEnvelope
from verbatim.jobs import source_jobs as sj
from verbatim.readiness import ReadinessEngine, ingest_receipt_id
from verbatim.security.quarantine import open_quarantine
from verbatim.sourcestate import state as src_state
from verbatim.sourcestate import transitions
from verbatim.storage.store import Store

SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")
SCOPE2 = Scope(profile_id="p", principal_id="bob", conversation_id="c2")


@pytest.fixture
def cfg() -> VerbatimConfig:
    c = VerbatimConfig()
    return replace(c, capture=replace(c.capture, enabled=True))


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v5.db"))
    yield s
    s.close()


@pytest.fixture
def encoder(cfg) -> HashingEncoder:
    return HashingEncoder(cfg.embedding)


@pytest.fixture
def ingester(store, cfg, encoder):
    return Ingester(store, cfg, encoder=encoder)


@pytest.fixture
def bare_ingester(store, cfg):
    """No encoder — ``source_embed`` must report the capability honestly."""
    return Ingester(store, cfg)


def _env(
    text: str,
    scope: Scope = SCOPE,
    ext: str | None = None,
    payload: bytes | None = None,
) -> SourceEnvelope:
    return SourceEnvelope(
        origin="test",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id=scope.principal_id,
        payload=payload if payload is not None else text.encode("utf-8"),
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
        external_id=ext,
    )


def _capture(ingester: Ingester, env: SourceEnvelope) -> tuple[str, int]:
    """Ingest one source and enqueue its v5 source jobs in-tx."""
    r = ingester.ingest(env)
    sid = r.accepted[0]
    rid = ingest_receipt_id(sid, 1)
    with ingester.store.tx() as conn:
        sj.enqueue_source_jobs(conn, ingester.store, receipt_id=rid)
    return sid, 1


def _lease(ingester: Ingester, *kinds: JobKind) -> list[dict]:
    return ingester.jobs.lease(None, list(kinds), owner="w1", limit=8)


def _drain_one(
    ingester: Ingester,
    kind: JobKind,
    handler,
    owner: str = "w1",
) -> dict:
    """Lease one job of ``kind`` and run its handler (the dispatcher's
    contract). Returns the leased job dict."""
    leased = _lease(ingester, kind)
    assert len(leased) == 1, f"expected one {kind.value} job"
    job = leased[0]
    handler(job, owner, ingester)
    return job


def _fail_like_drain(
    ingester: Ingester, job: dict, code: str, retryable: bool, owner="w1"
) -> None:
    """Mirror ``drain_report``'s failure path for a handler-raised error."""
    with ingester.store.tx() as conn:
        st = ingester.jobs.fail(
            conn, job["job_id"], owner, job["generation"], code, retryable
        )
        from verbatim.core.types import JobState

        if st is JobState.FAILED:
            ingester._fail_job_obligations(conn, job, code)


def _job_state(store: Store, job_id: str) -> str:
    with store.read() as conn:
        return conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()[0]


def _lexical_row(store: Store, sid: str, rev: int = 1):
    with store.read() as conn:
        return conn.execute(
            "SELECT * FROM source_lexical_projection"
            " WHERE source_id = ? AND revision = ?",
            (sid, rev),
        ).fetchone()


def _obligation_states(store: Store, receipt_id: str) -> dict[str, str]:
    snap = ReadinessEngine(store).receipt_state(receipt_id)
    return {k: v["state"] for k, v in snap["states"].items()}


def _seed_state(store: Store, sid: str, namespace: str, *, head: int = 1):
    """Create the source_state binding atomically (as the v5 facade does)."""
    with store.tx() as conn:
        return src_state.ensure_state(
            conn, sid, namespace, head=head, store=store
        )


# ----------------------------------------------------------------------
# source_project — lexical projection
# ----------------------------------------------------------------------


class TestSourceProject:
    def test_publishes_normalized_projection(self, store, ingester):
        sid, rev = _capture(
            ingester, _env("My Editor is  NEOVIM.\nSee  https://x.io/a")
        )
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)

        row = _lexical_row(store, sid, rev)
        assert row is not None
        (
            _sid,
            _rev,
            scope_id,
            generation,
            tokens,
            doc_len,
            digest,
        ) = row
        text = "My Editor is  NEOVIM.\nSee  https://x.io/a"
        assert tokens == normalize_text(text)
        assert tokens == " ".join(tokens.split())  # space-joined contract
        assert doc_len == len(normalize_text(text).split())
        assert digest == normalized_digest(text)
        assert generation == 1
        with store.read() as conn:
            scope = conn.execute(
                "SELECT scope_id FROM sources WHERE source_id = ?", (sid,)
            ).fetchone()[0]
        assert scope_id == scope

    def test_canonical_bytes_never_mutated(self, store, ingester):
        payload = "  Weird   SPACING\t\n".encode("utf-8")
        sid, _ = _capture(ingester, _env("", payload=payload))
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
        assert ingester.sources.payload(sid, 1) == payload

    def test_missing_source_fails_typed(self, store, ingester):
        with store.tx() as conn:
            jid = ingester.jobs.enqueue(
                conn,
                "scope:x",
                JobKind.SOURCE_PROJECT,
                {"source_id": "nope", "revision": 1},
            )
        job = _lease(ingester, JobKind.SOURCE_PROJECT)[0]
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_project(job, "w1", ingester)
        assert ei.value.code is ErrorCode.EVIDENCE_UNAVAILABLE
        _fail_like_drain(ingester, job, ei.value.code.value, False)
        assert _job_state(store, job["job_id"]) == "failed"

    def test_missing_revision_fails(self, store, ingester):
        sid, _ = _capture(ingester, _env("real bytes"))
        with store.tx() as conn:
            ingester.jobs.enqueue(
                conn,
                "scope:x",
                JobKind.SOURCE_PROJECT,
                {"source_id": sid, "revision": 99},
            )
        jobs = [
            j
            for j in _lease(ingester, JobKind.SOURCE_PROJECT)
            if j["input_refs"].get("revision") == 99
        ]
        assert len(jobs) == 1
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_project(jobs[0], "w1", ingester)
        assert ei.value.code is ErrorCode.EVIDENCE_UNAVAILABLE

    def test_strict_utf8_failure_is_permanent(self, store, ingester):
        """A malformed persisted payload fails VALIDATION — never a
        replacement-decoded projection (V4-13.12)."""
        sid, _ = _capture(ingester, _env("clean"))
        with store.tx() as conn:
            bad = b"\xff\xfe not utf8 \x80"
            conn.execute(
                "UPDATE source_revisions SET payload = ?, payload_hmac = ?"
                " WHERE source_id = ? AND revision = 1",
                (bad, store.hmac(bad), sid),
            )
        jobs = _lease(ingester, JobKind.SOURCE_PROJECT)
        job = jobs[0]
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_project(job, "w1", ingester)
        assert ei.value.code is ErrorCode.VALIDATION
        assert not ei.value.retryable
        _fail_like_drain(
            ingester, job, ei.value.code.value, ei.value.retryable
        )
        assert _lexical_row(store, sid) is None
        rid = ingest_receipt_id(sid, 1)
        states = _obligation_states(store, rid)
        assert states["source_lexical_ready"] == "failed"
        assert states["failed"] == "failed"

    def test_idempotent_replay(self, store, ingester):
        """A redelivered job replays the committed receipt — the job
        completes without re-applying effects (operation receipt)."""
        sid, rev = _capture(ingester, _env("replay me twice"))
        job = _drain_one(
            ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project
        )
        first = _lexical_row(store, sid, rev)
        # Force re-delivery under a fresh lease (the committed op receipt
        # replays; rows stay identical).
        with store.tx() as conn:
            conn.execute(
                "UPDATE jobs SET state = 'leased', lease_owner = 'w1',"
                " generation = generation + 1 WHERE job_id = ?",
                (job["job_id"],),
            )
            job["generation"] += 1
        sj.handle_source_project(job, "w1", ingester)
        assert _lexical_row(store, sid, rev) == first
        with store.read() as conn:
            n_fts = conn.execute(
                "SELECT COUNT(*) FROM source_fts_rows"
                " WHERE source_id = ? AND revision = ?",
                (sid, rev),
            ).fetchone()[0]
            n_idx = conn.execute(
                "SELECT COUNT(*) FROM source_fts_idx"
            ).fetchone()[0]
        assert n_fts == 1
        assert n_idx == 1

    def test_fts_shadow_consistent_and_replaced(self, store, ingester):
        sid, rev = _capture(ingester, _env("alpha bravo charlie"))
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
        with store.read() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM source_fts_rows WHERE source_id = ?",
                (sid,),
            ).fetchone()[0] == 1
            hit = conn.execute(
                "SELECT rowid FROM source_fts_idx"
                " WHERE source_fts_idx MATCH 'bravo'"
            ).fetchall()
            assert len(hit) == 1
        # Mutate the persisted bytes (re-sealed) and reproject — delete+
        # insert must retire the stale index terms. A fresh job row
        # (no dedup key) carries the bumped control_version fence.
        with store.tx() as conn:
            new = b"alpha delta echo"
            conn.execute(
                "UPDATE source_revisions SET payload = ?, payload_hmac = ?"
                " WHERE source_id = ? AND revision = 1",
                (new, store.hmac(new), sid),
            )
            conn.execute(
                "UPDATE source_state SET control_version = control_version + 1"
                " WHERE source_id = ?",
                (sid,),
            )
            ingester.jobs.enqueue(
                conn,
                "scope:x",
                JobKind.SOURCE_PROJECT,
                {
                    "source_id": sid,
                    "revision": rev,
                    "control_version": 1,
                },
            )
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
        with store.read() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM source_fts_rows WHERE source_id = ?",
                (sid,),
            ).fetchone()[0] == 1
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_fts_idx"
                    " WHERE source_fts_idx MATCH 'bravo'"
                ).fetchone()[0]
                == 0
            )
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_fts_idx"
                    " WHERE source_fts_idx MATCH 'delta'"
                ).fetchone()[0]
                == 1
            )
            assert (
                conn.execute("SELECT COUNT(*) FROM source_fts").fetchone()[0]
                == 1
            )

    def test_control_version_fence(self, store, ingester):
        """A pinned control_version that drifted since enqueue fails
        STALE_DEPENDENCY — stale output never overtakes a mutation
        (V5-14.14)."""
        sid, rev = _capture(ingester, _env("fenced bytes"))
        with store.tx() as conn:
            ingester.jobs.enqueue(
                conn,
                "scope:x",
                JobKind.SOURCE_PROJECT,
                {
                    "source_id": sid,
                    "revision": rev,
                    "control_version": 999,
                },
            )
        jobs = [
            j
            for j in _lease(ingester, JobKind.SOURCE_PROJECT)
            if j["input_refs"].get("control_version") == 999
        ]
        assert len(jobs) == 1
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_project(jobs[0], "w1", ingester)
        assert ei.value.code is ErrorCode.STALE_DEPENDENCY
        assert _lexical_row(store, sid, rev) is None

    def test_erased_source_refused(self, store, ingester):
        sid, rev = _capture(ingester, _env("to be erased"))
        scope_id = _scope_id_of(store, sid)
        with store.tx() as conn:
            transitions.apply_erasure(
                conn, sid, producer="test/erase",
                namespace=scope_id, store=store,
            )
            conn.execute(
                "UPDATE source_revisions SET payload = X'',"
                " payload_hmac = ? WHERE source_id = ?",
                (store.hmac(b""), sid),
            )
        job = _lease(ingester, JobKind.SOURCE_PROJECT)[0]
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_project(job, "w1", ingester)
        assert ei.value.code in (
            ErrorCode.INVALID_TRANSITION,
            ErrorCode.EVIDENCE_UNAVAILABLE,
        )
        assert _lexical_row(store, sid, rev) is None

    def test_held_source_refused(self, store, ingester):
        """A quarantine hold landing between enqueue and drain wins
        (V4-42.03 — the in-tx re-check, not the enqueue-time snapshot)."""
        sid, rev = _capture(ingester, _env("suspicious bytes"))
        with store.tx() as conn:
            scope_id = conn.execute(
                "SELECT scope_id FROM sources WHERE source_id = ?", (sid,)
            ).fetchone()[0]
            open_quarantine(
                conn,
                ("source", sid, rev),
                ["attack_risk:suspicious"],
                [{"rule_id": "test.hold"}],
                scope_id=scope_id,
            )
        job = _lease(ingester, JobKind.SOURCE_PROJECT)[0]
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_project(job, "w1", ingester)
        assert ei.value.code is ErrorCode.QUARANTINED
        assert _lexical_row(store, sid, rev) is None

    def test_suppressed_source_refused(self, store, ingester):
        sid, rev = _capture(ingester, _env("suppressed bytes"))
        with store.tx() as conn:
            scope_id = conn.execute(
                "SELECT scope_id FROM sources WHERE source_id = ?", (sid,)
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO purges (purge_id, selection_digest, scope_id,"
                " state, requested_us) VALUES ('pg1', X'00', ?,"
                " 'suppressed', ?)",
                (scope_id, now_us()),
            )
            conn.execute(
                "INSERT INTO purge_targets (purge_id, object_kind,"
                " object_id) VALUES ('pg1', 'source', ?)",
                (sid,),
            )
        job = _lease(ingester, JobKind.SOURCE_PROJECT)[0]
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_project(job, "w1", ingester)
        assert ei.value.code is ErrorCode.EVIDENCE_UNAVAILABLE
        assert _lexical_row(store, sid, rev) is None

    def test_stale_lease_blocks_writes(self, store, ingester):
        """Generation fencing: a superseded lease cannot commit."""
        sid, rev = _capture(ingester, _env("lease fenced"))
        job = _lease(ingester, JobKind.SOURCE_PROJECT)[0]
        with store.tx() as conn:
            # Reclaim-style generation bump: the lease token is dead.
            conn.execute(
                "UPDATE jobs SET generation = generation + 9,"
                " state = 'retry_wait' WHERE job_id = ?",
                (job["job_id"],),
            )
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_project(job, "w1", ingester)
        assert ei.value.code in (
            ErrorCode.LEASE_LOST,
            ErrorCode.STALE_DEPENDENCY,
            ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
            ErrorCode.INVALID_TRANSITION,
        )
        assert _lexical_row(store, sid, rev) is None

    def test_pinned_generation_ahead_of_store(self, store, ingester):
        sid, rev = _capture(ingester, _env("gen fenced"))
        with store.tx() as conn:
            ingester.jobs.enqueue(
                conn,
                "scope:x",
                JobKind.SOURCE_PROJECT,
                {"source_id": sid, "revision": rev, "generation": 99},
            )
        jobs = [
            j
            for j in _lease(ingester, JobKind.SOURCE_PROJECT)
            if j["input_refs"].get("generation") == 99
        ]
        assert len(jobs) == 1
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_project(jobs[0], "w1", ingester)
        assert ei.value.code is ErrorCode.STALE_DEPENDENCY
        assert ei.value.retryable

    def test_entity_postings_and_enrichment(self, store, ingester):
        sid, rev = _capture(
            ingester, _env("Ping alice@example.com about PROJ-1234")
        )
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
        with store.read() as conn:
            postings = conn.execute(
                "SELECT entity, entity_kind, offsets FROM entity_postings"
                " WHERE source_id = ? AND revision = ?",
                (sid, rev),
            ).fetchall()
            enr = conn.execute(
                "SELECT producer, type, polarity, fields_json"
                " FROM enrichment WHERE source_id = ? AND revision = ?",
                (sid, rev),
            ).fetchone()
        values = {p[0] for p in postings}
        assert "alice@example.com" in values
        assert "PROJ-1234" in values
        assert enr is not None
        assert enr[0] == "enrich/v1"
        import json as _json

        fields = _json.loads(enr[3])
        assert any(
            i["value"] == "alice@example.com" for i in fields["identifiers"]
        )

    def test_namespace_isolation_no_cross_links(self, store, ingester):
        """Identical payloads in different namespaces must not link."""
        sid1, _ = _capture(ingester, _env("identical body", scope=SCOPE))
        sid2, _ = _capture(ingester, _env("identical body", scope=SCOPE2))
        for j in _lease(ingester, JobKind.SOURCE_PROJECT):
            sj.handle_source_project(j, "w1", ingester)
        with store.read() as conn:
            links = conn.execute(
                "SELECT source_id, method FROM duplicate_links"
            ).fetchall()
            ns = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT source_id, namespace FROM entity_postings"
                ).fetchall()
            }
            lex_ns = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT source_id, scope_id FROM source_lexical_projection"
                ).fetchall()
            }
        assert links == []
        assert len(set(lex_ns.values())) == 2  # projection rows partitioned
        if ns:
            assert len(set(ns.values())) == 2  # postings stayed partitioned

    def test_exact_duplicate_links_in_namespace(self, store, ingester):
        sid1, _ = _capture(ingester, _env("same bytes here", ext="a"))
        sid2, _ = _capture(ingester, _env("same bytes here", ext="b"))
        for j in _lease(ingester, JobKind.SOURCE_PROJECT):
            sj.handle_source_project(j, "w1", ingester)
        with store.read() as conn:
            rows = conn.execute(
                "SELECT source_id, revision, group_id, method, score"
                " FROM duplicate_links ORDER BY method"
            ).fetchall()
        methods = {r[3] for r in rows}
        assert "exact_digest" in methods
        assert "normalized" in methods
        # The later source links into the earliest member's group.
        groups = {r[0]: r[2] for r in rows}
        assert groups.get(sid2) == sid1

    def test_readiness_fulfillment(self, store, ingester):
        sid, rev = _capture(ingester, _env("ready me"))
        rid = ingest_receipt_id(sid, rev)
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
        states = _obligation_states(store, rid)
        assert states["screened"] == "succeeded"
        assert states["source_lexical_ready"] == "succeeded"
        # Claim-lane capabilities are untouched by the source handler.
        assert states["lexical_ready"] == "pending"
        assert states["semantic_ready"] == "pending"

    def test_utf8_ok_false_rejected(self, store, ingester):
        sid, rev = _capture(ingester, _env("bytes"))
        with store.tx() as conn:
            ingester.jobs.enqueue(
                conn,
                "scope:x",
                JobKind.SOURCE_PROJECT,
                {"source_id": sid, "revision": rev, "utf8_ok": False},
            )
        jobs = [
            j
            for j in _lease(ingester, JobKind.SOURCE_PROJECT)
            if j["input_refs"].get("utf8_ok") is False
        ]
        assert len(jobs) == 1
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_project(jobs[0], "w1", ingester)
        assert ei.value.code is ErrorCode.VALIDATION

    def test_stale_plan_aborts_and_replans(self, store, ingester, monkeypatch):
        """A fingerprint drift between prescan and commit aborts the
        fenced commit — rollback, never partial state — and the job
        replans on a fresh snapshot instead of scanning the live
        namespace under the write lock."""
        sid, rev = _capture(ingester, _env("project alpha rollout notes"))
        job = _lease(ingester, JobKind.SOURCE_PROJECT)[0]

        # The retry budget only exists when the fused scan is expensive
        # — force the cost gate open so this small fixture exercises the
        # replan path.
        monkeypatch.setattr(sj, "_REPLAN_MIN_SCAN_MS", 0.0)

        real_fp = sj._deps_fingerprint
        real_prescan = sj._prescan
        prescans = {"n": 0}
        forced = {"done": False}

        def flaky_fp(conn, mode="fold"):
            # Prescan fingerprinting runs on the snapshot read conn; the
            # commit-time check runs on the writer conn inside the tx.
            # Drift exactly once — only on that first in-tx fingerprint —
            # so the commit aborts and the replan sees a stable world.
            if conn is store._writer and not forced["done"]:
                forced["done"] = True
                return ("counter", "forced-drift")
            return real_fp(conn, mode)

        def counting_prescan(*a, **kw):
            prescans["n"] += 1
            return real_prescan(*a, **kw)

        monkeypatch.setattr(sj, "_deps_fingerprint", flaky_fp)
        monkeypatch.setattr(sj, "_prescan", counting_prescan)

        sj.handle_source_project(job, "w1", ingester)

        assert prescans["n"] == 2, "expected one replan after the abort"
        assert _job_state(store, job["job_id"]) == "succeeded"
        assert _lexical_row(store, sid, rev) is not None

    def test_persistent_drift_falls_back_to_fused(
        self, store, ingester, monkeypatch
    ):
        """When the dependency fingerprint never stabilizes the bounded
        retries exhaust to the fused in-transaction path — the honest
        bound the job always had — and the commit still completes
        atomically."""
        sid, rev = _capture(ingester, _env("another rollout note"))
        job = _lease(ingester, JobKind.SOURCE_PROJECT)[0]
        monkeypatch.setattr(sj, "_REPLAN_MIN_SCAN_MS", 0.0)
        monkeypatch.setattr(sj, "_PLAN_STALE_RETRIES", 2)

        seq = {"n": 0}
        fused = {"n": 0}
        planned = {"n": 0}
        real_run = sj._run_dedup

        def drifting_fp(conn, mode="fold"):
            seq["n"] += 1
            return ("counter", f"fp-{seq['n']}")  # never equal twice

        def counting_run(conn, **kw):
            bucket = planned if kw.get("near_plan") is not None else fused
            bucket["n"] += 1
            return real_run(conn, **kw)

        monkeypatch.setattr(sj, "_deps_fingerprint", drifting_fp)
        monkeypatch.setattr(sj, "_run_dedup", counting_run)

        sj.handle_source_project(job, "w1", ingester)

        assert fused["n"] == 1 and planned["n"] == 0, (
            f"expected exactly one fused fallback, got "
            f"planned={planned['n']} fused={fused['n']}"
        )
        assert _job_state(store, job["job_id"]) == "succeeded"
        assert _lexical_row(store, sid, rev) is not None


# ----------------------------------------------------------------------
# source_embed — vector projection
# ----------------------------------------------------------------------


class TestSourceEmbed:
    def test_publishes_validated_vector(self, store, ingester, encoder):
        sid, rev = _capture(ingester, _env("embed me please"))
        _drain_one(ingester, JobKind.SOURCE_EMBED, sj.handle_source_embed)
        with store.read() as conn:
            row = conn.execute(
                "SELECT source_id, revision, namespace, encoder,"
                " generation, vector, digest FROM source_vectors"
                " WHERE source_id = ? AND revision = ?",
                (sid, rev),
            ).fetchone()
        assert row is not None
        assert row[3] == encoder.encoder_id == "hashing:subword-ngram:v1"
        assert row[4] == 1
        blob = bytes(row[5])
        # Codec validation: exactly 384 float32 lanes, finite, unit norm.
        vec = Float32Codec.validate_blob(blob, encoder.dimensions)
        import math

        assert abs(math.fsum(x * x for x in vec) - 1.0) < 1e-5
        assert row[6] == sj._vector_digest(encoder.encoder_id, blob)

    def test_encoder_unprovisioned_defers_honestly(
        self, store, bare_ingester
    ):
        """No configured encoder → recorded deferral, not a failure and
        not a pending lie (V2-27.02 convention)."""
        sid, rev = _capture(bare_ingester, _env("no encoder here"))
        rid = ingest_receipt_id(sid, rev)
        _drain_one(
            bare_ingester, JobKind.SOURCE_EMBED, sj.handle_source_embed
        )
        states = _obligation_states(store, rid)
        assert states["source_vector_ready"] == "deferred"
        assert states["failed"] != "failed"
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_vectors WHERE source_id = ?",
                    (sid,),
                ).fetchone()[0]
                == 0
            )
            ev = conn.execute(
                "SELECT payload_json FROM events"
                " WHERE kind = 'source_embed_skipped'"
            ).fetchone()
        assert "encoder_unavailable" in ev[0]

    def test_unavailable_backend_retryable(self, store, cfg):
        class _Down:
            encoder_id = "stub:down:v1"
            dimensions = 8

            def available(self):
                return False

            def encode(self, texts):  # pragma: no cover - never reached
                raise AssertionError("unavailable encoder must not encode")

        ing = Ingester(store, cfg, encoder=_Down())
        sid, rev = _capture(ing, _env("backend down"))
        job = _lease(ing, JobKind.SOURCE_EMBED)[0]
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_embed(job, "w1", ing)
        assert ei.value.code is ErrorCode.ENCODER_UNAVAILABLE
        assert ei.value.retryable

    def test_invalid_blob_is_permanent_vector_failure(self, store, cfg):
        class _BadBlob:
            encoder_id = "stub:bad:v1"
            dimensions = 8

            def available(self):
                return True

            def encode(self, texts):
                return [b"\x00" * 12]  # not 8*4 bytes

        ing = Ingester(store, cfg, encoder=_BadBlob())
        sid, rev = _capture(ing, _env("bad vector"))
        rid = ingest_receipt_id(sid, rev)
        job = _lease(ing, JobKind.SOURCE_EMBED)[0]
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_embed(job, "w1", ing)
        assert ei.value.code is ErrorCode.VECTOR_INVALID
        assert not ei.value.retryable
        _fail_like_drain(ing, job, ei.value.code.value, False)
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_vectors WHERE source_id = ?",
                    (sid,),
                ).fetchone()[0]
                == 0
            )
        states = _obligation_states(store, rid)
        assert states["source_vector_ready"] == "failed"
        assert states["failed"] == "failed"

    def test_encoder_identity_mismatch_rejected(self, store, ingester):
        sid, rev = _capture(ingester, _env("pin me"))
        with store.tx() as conn:
            ingester.jobs.enqueue(
                conn,
                "scope:x",
                JobKind.SOURCE_EMBED,
                {
                    "source_id": sid,
                    "revision": rev,
                    "encoder": "other:enc:v9",
                },
            )
        jobs = [
            j
            for j in _lease(ingester, JobKind.SOURCE_EMBED)
            if j["input_refs"].get("encoder") == "other:enc:v9"
        ]
        assert len(jobs) == 1
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_embed(jobs[0], "w1", ingester)
        assert ei.value.code is ErrorCode.VALIDATION

    def test_held_source_refused(self, store, ingester):
        sid, rev = _capture(ingester, _env("held vector"))
        with store.tx() as conn:
            scope_id = conn.execute(
                "SELECT scope_id FROM sources WHERE source_id = ?", (sid,)
            ).fetchone()[0]
            open_quarantine(
                conn,
                ("source", sid, rev),
                ["attack_risk:suspicious"],
                [{"rule_id": "test.hold"}],
                scope_id=scope_id,
            )
        job = _lease(ingester, JobKind.SOURCE_EMBED)[0]
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_embed(job, "w1", ingester)
        assert ei.value.code is ErrorCode.QUARANTINED
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_vectors WHERE source_id = ?",
                    (sid,),
                ).fetchone()[0]
                == 0
            )

    def test_readiness_fulfillment(self, store, ingester):
        sid, rev = _capture(ingester, _env("vector ready"))
        rid = ingest_receipt_id(sid, rev)
        _drain_one(ingester, JobKind.SOURCE_EMBED, sj.handle_source_embed)
        states = _obligation_states(store, rid)
        assert states["source_vector_ready"] == "succeeded"
        assert states["source_lexical_ready"] != "succeeded"  # not claimed


# ----------------------------------------------------------------------
# source_backfill — bounded resumable scan
# ----------------------------------------------------------------------


def _capture_many(ingester: Ingester, n: int, scope: Scope = SCOPE):
    sids = []
    for i in range(n):
        r = ingester.ingest(
            _env(f"backfill body {i} text", scope=scope, ext=f"bf-{i}")
        )
        sids.append(r.accepted[0])
    return sids


def _scope_id_of(store: Store, sid: str) -> str:
    with store.read() as conn:
        return conn.execute(
            "SELECT scope_id FROM sources WHERE source_id = ?", (sid,)
        ).fetchone()[0]


def _cursor(store: Store, job_key: str):
    with store.read() as conn:
        return conn.execute(
            "SELECT * FROM backfill_cursor WHERE job_key = ?", (job_key,)
        ).fetchone()


def _drain_backfill(ingester: Ingester, rounds: int = 1) -> list:
    """Lease + handle up to ``rounds`` queued backfill jobs."""
    ran = []
    for _ in range(rounds):
        leased = _lease(ingester, JobKind.SOURCE_BACKFILL)
        if not leased:
            break
        for job in leased:
            sj.handle_source_backfill(job, "w1", ingester)
            ran.append(job)
    return ran


class TestSourceBackfill:
    def test_bounded_resumable_coverage(self, store, ingester):
        sids = _capture_many(ingester, 5)
        scope_id = _scope_id_of(store, sids[0])
        with store.tx() as conn:
            sj.enqueue_source_backfill(
                conn, ingester, scope_id=scope_id, batch_size=2
            )
        # Round 1: exactly two sources projected, cursor advanced, a
        # continuation job committed atomically.
        _drain_backfill(ingester, 1)
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_lexical_projection"
                ).fetchone()[0]
                == 2
            )
            live = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind = 'source_backfill'"
                " AND state IN ('queued','leased','retry_wait')"
            ).fetchone()[0]
        cur = _cursor(store, "source_backfill:all")
        assert cur is not None and cur[3] == 0  # done = 0
        assert live == 1
        _drain_backfill(ingester, 10)
        cur = _cursor(store, "source_backfill:all")
        assert cur[3] == 1  # done
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_lexical_projection"
                ).fetchone()[0]
                == 5
            )
            assert (
                conn.execute("SELECT COUNT(*) FROM source_vectors").fetchone()[
                    0
                ]
                == 5
            )
            states = conn.execute(
                "SELECT disposition, mutation_head FROM source_state"
            ).fetchall()
        # Governed adoption: recorded + unresolved head, never inferred
        # approval (V5-08.18).
        assert all(d == "recorded" for d, _ in states)
        assert all(h == "unresolved" for _, h in states)

    def test_cursor_commits_with_batch_only(self, store, ingester):
        """Crash before the batch commit leaves the cursor untouched —
        the retried job re-scans the same window (no skipped sources)."""
        sids = _capture_many(ingester, 3)
        scope_id = _scope_id_of(store, sids[0])
        with store.tx() as conn:
            sj.enqueue_source_backfill(
                conn, ingester, scope_id=scope_id, batch_size=2
            )
        job = _lease(ingester, JobKind.SOURCE_BACKFILL)[0]

        # Crash simulation: kill the tx mid-apply by forcing an error —
        # rollback restores both the projections AND the cursor.
        original = sj._write_lexical

        def _boom(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("simulated crash mid-batch")

        # A non-VerbatimError propagates: tx rolls back entirely.
        import verbatim.jobs.source_jobs as mod

        held = mod._write_lexical
        try:
            mod._write_lexical = _boom
            with pytest.raises(RuntimeError):
                sj.handle_source_backfill(job, "w1", ingester)
        finally:
            mod._write_lexical = held
        assert _cursor(store, "source_backfill:all") is None
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_lexical_projection"
                ).fetchone()[0]
                == 0
            )
        # Re-drive the same job row (lease reclaimed): it redoes the
        # identical window — the cursor never advanced past unprojected
        # work.
        with store.tx() as conn:
            conn.execute(
                "UPDATE jobs SET state = 'queued', lease_owner = NULL,"
                " lease_until_us = NULL, generation = generation + 1"
                " WHERE job_id = ?",
                (job["job_id"],),
            )
        _drain_backfill(ingester, 10)
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_lexical_projection"
                ).fetchone()[0]
                == 3
            )
        assert _cursor(store, "source_backfill:all")[3] == 1

    def test_namespace_scoped_scan(self, store, ingester):
        a = _capture_many(ingester, 2, scope=SCOPE)
        b = _capture_many(ingester, 2, scope=SCOPE2)
        ns_a = _scope_id_of(store, a[0])
        with store.tx() as conn:
            sj.enqueue_source_backfill(
                conn,
                ingester,
                scope_id=ns_a,
                namespace=ns_a,
                batch_size=8,
            )
        _drain_backfill(ingester, 10)
        with store.read() as conn:
            projected = {
                r[0]
                for r in conn.execute(
                    "SELECT source_id FROM source_lexical_projection"
                ).fetchall()
            }
        assert set(a) <= projected
        assert not (set(b) & projected)
        # And the scan never wrote postings into the wrong partition.
        with store.read() as conn:
            bad = conn.execute(
                "SELECT COUNT(*) FROM entity_postings WHERE namespace != ?",
                (ns_a,),
            ).fetchone()[0]
        assert bad == 0

    def test_held_source_disposition_not_projected(self, store, ingester):
        sids = _capture_many(ingester, 2)
        scope_id = _scope_id_of(store, sids[0])
        with store.tx() as conn:
            open_quarantine(
                conn,
                ("source", sids[0], 1),
                ["attack_risk:suspicious"],
                [{"rule_id": "test.hold"}],
                scope_id=scope_id,
            )
            sj.enqueue_source_backfill(
                conn, ingester, scope_id=scope_id, batch_size=8
            )
        _drain_backfill(ingester, 10)
        with store.read() as conn:
            projected = {
                r[0]
                for r in conn.execute(
                    "SELECT source_id FROM source_lexical_projection"
                ).fetchall()
            }
            ev = conn.execute(
                "SELECT payload_json FROM events"
                " WHERE kind = 'source_backfill_batch'"
            ).fetchall()
        assert sids[0] not in projected
        assert sids[1] in projected
        assert ev and "held_partial" in ev[-1][0]

    def test_done_cursor_is_idempotent(self, store, ingester):
        sids = _capture_many(ingester, 1)
        scope_id = _scope_id_of(store, sids[0])
        with store.tx() as conn:
            sj.enqueue_source_backfill(
                conn, ingester, scope_id=scope_id, batch_size=8
            )
        _drain_backfill(ingester, 10)
        assert _cursor(store, "source_backfill:all")[3] == 1
        # A second plan over the same key converges on done — it must not
        # rewind the cursor or fabricate new work.
        with store.tx() as conn:
            sj.enqueue_source_backfill(
                conn, ingester, scope_id=scope_id, batch_size=8
            )
        _drain_backfill(ingester, 10)
        assert _cursor(store, "source_backfill:all")[3] == 1
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_lexical_projection"
                ).fetchone()[0]
                == 1
            )

    def test_backfill_never_fabricates_obligations(self, store, ingester):
        """V5-08.17: no source_* obligation appears for a capture that
        never declared the branch — coverage is the cursor's, not a
        readiness lie."""
        sids = _capture_many(ingester, 2)
        scope_id = _scope_id_of(store, sids[0])
        with store.tx() as conn:
            sj.enqueue_source_backfill(
                conn, ingester, scope_id=scope_id, batch_size=8
            )
        _drain_backfill(ingester, 10)
        with store.read() as conn:
            src_caps = conn.execute(
                "SELECT COUNT(*) FROM readiness_obligations"
                " WHERE capability LIKE 'source_%'"
            ).fetchone()[0]
        # The v2 capture receipts never declared source caps and no
        # per-source job evidence exists — nothing may appear.
        assert src_caps == 0

    def test_stale_plan_aborts_and_replans(self, store, ingester, monkeypatch):
        """A fingerprint drift between the batch prescan and the fenced
        commit aborts the whole batch transaction — rollback, never
        partial state (cursor, dispositions, continuation all roll back
        together) — and the handler replans on a fresh snapshot instead
        of running the fused namespace scans under the write lock."""
        sids = _capture_many(ingester, 2)
        scope_id = _scope_id_of(store, sids[0])
        with store.tx() as conn:
            sj.enqueue_source_backfill(
                conn, ingester, scope_id=scope_id, batch_size=8
            )
        job = _lease(ingester, JobKind.SOURCE_BACKFILL)[0]

        # The retry budget only exists when the fused scans are
        # expensive — force the cost gate open so this small fixture
        # exercises the replan path.
        monkeypatch.setattr(sj, "_REPLAN_MIN_SCAN_MS", 0.0)

        real_fp = sj._deps_fingerprint
        real_prescan = sj._backfill_prescan
        real_run = sj._run_dedup
        prescans = {"n": 0}
        planned = {"n": 0}
        forced = {"done": False}

        def flaky_fp(conn, mode="fold"):
            # Prescan fingerprinting runs on the snapshot read conn; the
            # commit-time check runs on the writer conn inside the tx.
            # Drift exactly once — only on that first in-tx fingerprint —
            # so the batch aborts and the replan sees a stable world.
            if conn is store._writer and not forced["done"]:
                forced["done"] = True
                return ("counter", "forced-drift")
            return real_fp(conn, mode)

        def counting_prescan(*a, **kw):
            prescans["n"] += 1
            return real_prescan(*a, **kw)

        def counting_run(conn, **kw):
            if kw.get("near_plan") is not None:
                planned["n"] += 1
            return real_run(conn, **kw)

        monkeypatch.setattr(sj, "_deps_fingerprint", flaky_fp)
        monkeypatch.setattr(sj, "_backfill_prescan", counting_prescan)
        monkeypatch.setattr(sj, "_run_dedup", counting_run)

        sj.handle_source_backfill(job, "w1", ingester)

        assert prescans["n"] == 2, "expected one replan after the abort"
        # Both revisions committed from the replanned snapshot plans.
        assert planned["n"] == 2
        assert _job_state(store, job["job_id"]) == "succeeded"
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_lexical_projection"
                ).fetchone()[0]
                == 2
            )
        assert _cursor(store, "source_backfill:all")[3] == 1

    def test_persistent_drift_falls_back_to_fused(
        self, store, ingester, monkeypatch
    ):
        """When the dependency fingerprint never stabilizes the bounded
        retries exhaust to the fused in-transaction path — the honest
        bound the batch always had — and the commit still completes
        atomically (cursor + dispositions + projections together)."""
        sids = _capture_many(ingester, 2)
        scope_id = _scope_id_of(store, sids[0])
        with store.tx() as conn:
            sj.enqueue_source_backfill(
                conn, ingester, scope_id=scope_id, batch_size=8
            )
        job = _lease(ingester, JobKind.SOURCE_BACKFILL)[0]
        monkeypatch.setattr(sj, "_REPLAN_MIN_SCAN_MS", 0.0)
        monkeypatch.setattr(sj, "_PLAN_STALE_RETRIES", 2)

        seq = {"n": 0}
        fused = {"n": 0}
        planned = {"n": 0}
        real_run = sj._run_dedup

        def drifting_fp(conn, mode="fold"):
            seq["n"] += 1
            return ("counter", f"fp-{seq['n']}")  # never equal twice

        def counting_run(conn, **kw):
            bucket = planned if kw.get("near_plan") is not None else fused
            bucket["n"] += 1
            return real_run(conn, **kw)

        monkeypatch.setattr(sj, "_deps_fingerprint", drifting_fp)
        monkeypatch.setattr(sj, "_run_dedup", counting_run)

        sj.handle_source_backfill(job, "w1", ingester)

        assert planned["n"] == 0 and fused["n"] == 2, (
            f"expected one fused fallback per revision, got "
            f"planned={planned['n']} fused={fused['n']}"
        )
        assert _job_state(store, job["job_id"]) == "succeeded"
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM source_lexical_projection"
                ).fetchone()[0]
                == 2
            )
        assert _cursor(store, "source_backfill:all")[3] == 1


# ----------------------------------------------------------------------
# enqueue seam
# ----------------------------------------------------------------------


class TestEnqueueSeam:
    def test_enqueue_resolves_receipt_and_dedups(self, store, ingester):
        r = ingester.ingest(_env("seam test"))
        sid = r.accepted[0]
        rid = ingest_receipt_id(sid, 1)
        with store.tx() as conn:
            out1 = sj.enqueue_source_jobs(conn, store, receipt_id=rid)
            out2 = sj.enqueue_source_jobs(conn, store, receipt_id=rid)
        assert out1["job_ids"] == out2["job_ids"]
        assert len(out1["job_ids"]) == 2
        assert out1["source_id"] == sid and out1["revision"] == 1
        with store.read() as conn:
            kinds = sorted(
                r[0]
                for r in conn.execute(
                    "SELECT kind FROM jobs"
                    " WHERE kind IN ('source_project','source_embed')"
                ).fetchall()
            )
        assert kinds == ["source_embed", "source_project"]

    def test_enqueue_unknown_receipt_fails(self, store, ingester):
        with store.tx() as conn:
            with pytest.raises(VerbatimError) as ei:
                sj.enqueue_source_jobs(
                    conn, store, receipt_id="rc_ingest:ghost:1"
                )
        assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_cr_receipt_resolves(self, store, ingester):
        """cr_* ids reverse through source_envelopes."""
        r = ingester.ingest(_env("cr path"))
        sid = r.accepted[0]
        scope_id = _scope_id_of(store, sid)
        with store.tx() as conn:
            conn.execute(
                "INSERT INTO source_envelopes"
                " (envelope_id, source_id, revision, envelope_kind,"
                "  scope_id, trust_class, receipt_us)"
                " VALUES ('env1', ?, 1, 'user_message', ?, 'unknown', ?)",
                (sid, scope_id, now_us()),
            )
            from verbatim.evidence.receipts import _receipt_id

            rid = _receipt_id(sid, 1, "user_message")
            out = sj.enqueue_source_jobs(conn, store, receipt_id=rid)
        assert out["source_id"] == sid
        with store.read() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM jobs"
                    " WHERE kind = 'source_project'"
                ).fetchone()[0]
                == 1
            )
