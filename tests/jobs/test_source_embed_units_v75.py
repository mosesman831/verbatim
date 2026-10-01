"""V75-03.02 — ``source_embed`` produces the unit-vector block matrix.

Covers the V7.5 acceptance scenarios:

* **J04** — ``source_embed`` writes ``unit_vectors_block`` rows at the
  pinned generation (``encoder_id = <model>:hdr:v1`` — the V7-07.09
  contextual-header version rides inside the vector-space id), mirrors
  every pinned unit into the f32 ``unit_vectors`` oracle, and the dense
  lane resolves the headered space and returns the units.
* **J05** — no configured encoder keeps the ``source_embed_skipped``
  deferral; dense reports ``partial(n_pending)``/uncovered or
  unavailable, never a successful zero-block answer while work is owed.
* **J06** — source erasure removes the source's block rows (byte-exact
  rowmap surgery) and its oracle rows; a post-erasure dense scan can
  never emit an erased unit.

Plus the mechanical invariants: the ``hdr:v1`` header text is
deterministic and byte-pinned, block_no appends at ``MAX+1`` per
``(encoder_id, scope_id, generation)``, the int8 threshold switches
quant, and a unit-slice drift between encode snapshot and fenced commit
aborts (``_PlanStale``) rather than writing stale vectors.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.types import (
    JobKind,
    Provenance,
    Scope,
    SourceKind,
)
from verbatim.core.types_v7 import (
    BudgetClass,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneSlice,
    LaneStatus,
    NormAnalysis,
    QueryViewV7,
    RetrievalPolicyV7,
)
from verbatim.embeddings import matrix as mx
from verbatim.embeddings.codec import Float32Codec
from verbatim.embeddings.hashing import HashingEncoder
from verbatim.ingest import Ingester, SourceEnvelope
from verbatim.jobs import source_jobs as sj
from verbatim.privacy import closure_v7
from verbatim.readiness import ingest_receipt_id
from verbatim.retrieval.v7.dense import lane_dense
from verbatim.storage.store import Store

SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")

# Fixed capture instant — 2024-03-05 UTC — so the hdr:v1 date field is
# deterministic in tests.
_EVENT_US = int(
    datetime(2024, 3, 5, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1e6
)


@pytest.fixture
def cfg() -> VerbatimConfig:
    c = VerbatimConfig()
    return replace(c, capture=replace(c.capture, enabled=True))


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v75.db"))
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
    return Ingester(store, cfg)  # no encoder capability


def _env(
    text: str,
    *,
    scope: Scope = SCOPE,
    speaker: str | None = "Alice",
    metadata: dict | None = None,
    event_us: int = _EVENT_US,
) -> SourceEnvelope:
    return SourceEnvelope(
        origin="test",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id=speaker,
        payload=text.encode("utf-8"),
        event_us=event_us,
        captured_us=event_us,
        provenance=Provenance.DIRECT_USER,
        metadata=metadata or {},
    )


def _capture(ingester: Ingester, env: SourceEnvelope) -> tuple[str, int]:
    r = ingester.ingest(env)
    sid = r.accepted[0]
    rid = ingest_receipt_id(sid, 1)
    with ingester.store.tx() as conn:
        sj.enqueue_source_jobs(conn, ingester.store, receipt_id=rid)
    return sid, 1


def _drain_one(ingester: Ingester, kind: JobKind, handler) -> dict:
    leased = ingester.jobs.lease(None, [kind], owner="w1", limit=8)
    assert len(leased) == 1, f"expected one {kind.value} job"
    job = leased[0]
    handler(job, "w1", ingester)
    return job


def _project_then_embed(ingester: Ingester) -> None:
    _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
    _drain_one(ingester, JobKind.SOURCE_EMBED, sj.handle_source_embed)


def _units(conn, source_id: str, revision: int = 1) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM units WHERE source_id = ? AND revision = ?"
        " ORDER BY unit_id",
        (source_id, revision),
    ).fetchall()
    conn.row_factory = None
    return rows


def _blocks(conn, encoder_id: str) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM unit_vectors_block WHERE encoder_id = ?"
        " ORDER BY scope_id, generation, block_no",
        (encoder_id,),
    ).fetchall()
    conn.row_factory = None
    return rows


def _rowmap_keys(row) -> list[str]:
    return list(mx._unpack_rowmap(bytes(row["rowmap_blob"]), row["n_rows"]))


def _qv(text: str) -> QueryViewV7:
    return QueryViewV7(
        query=text,
        norm=NormAnalysis(analyzer_id="norm/v2", terms=(), identifiers=()),
        intent=IntentResult(
            primary=IntentClass.LOOKUP, classes=(IntentClass.LOOKUP,)
        ),
    )


def _slice(cap: int = 40) -> LaneSlice:
    return LaneSlice(deadline_ms=10_000.0, cap=cap)


def _ctx(
    store: Store,
    *,
    scope_id: str,
    generation: int,
    encoder,
    pin_encoder_id: str | None = None,
) -> LaneContextV7:
    manifest: dict = {"query_encoder": encoder}
    if pin_encoder_id is not None:
        manifest["encoder_id"] = pin_encoder_id
    return LaneContextV7(
        store=store,
        scope_id=scope_id,
        generation=generation,
        eligible=None,
        query_time_us=_EVENT_US,
        profile="local_memory",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7",
            profile="local_memory",
            lanes=(),
            lane_weights={},
        ),
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# J04 — producer writes blocks; dense returns them under the hdr:v1 space
# ---------------------------------------------------------------------------


class TestJ04Production:
    def test_blocks_written_at_pinned_generation_hdr_space(
        self, store, ingester, encoder
    ):
        sid, rev = _capture(
            ingester,
            _env(
                "Alice moved to Berlin in March with her dog.",
                metadata={"session_id": "sess-42"},
            ),
        )
        _project_then_embed(ingester)

        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        assert space == f"{encoder.encoder_id}:hdr:v1"

        with store.read() as conn:
            units = _units(conn, sid, rev)
            assert units, "projection produced no units"
            pinned = [
                u for u in units
                if u["byte_start"] is not None and u["byte_end"] is not None
            ]
            assert pinned, "expected at least one byte-pinned unit"
            gen = pinned[0]["generation"]

            blocks = _blocks(conn, space)
            assert blocks, "no unit_vectors_block rows written"
            for b in blocks:
                assert b["generation"] == gen
                assert b["dims"] == encoder.dimensions
                assert b["quant"] == "f32"  # far below the int8 threshold
            covered = {k for b in blocks for k in _rowmap_keys(b)}
            assert covered == {u["unit_id"] for u in pinned}

            # Per-row f32 oracle mirrors every embedded unit.
            oracle = conn.execute(
                "SELECT unit_key, vector FROM unit_vectors"
                " WHERE encoder_id = ?",
                (space,),
            ).fetchall()
            assert {r[0] for r in oracle} == covered
            assert len(oracle) == len(pinned)
            # The bare model id owns no unit blocks — hdr:v1 is a
            # distinct vector space (V7-07.13: never mixed).
            assert _blocks(conn, encoder.encoder_id) == []

            # Event receipt carries the unit-vector stats.
            ev = conn.execute(
                "SELECT payload_json FROM events"
                " WHERE kind = 'source_embedded'"
            ).fetchone()
        assert '"unit_encoder"' in ev[0] and "hdr:v1" in ev[0]

    def test_dense_returns_units_through_hdr_space(
        self, store, ingester, encoder
    ):
        sid, rev = _capture(
            ingester,
            _env(
                "Alice moved to Berlin in March with her dog.",
                metadata={"session_id": "sess-42"},
            ),
        )
        _project_then_embed(ingester)
        with store.read() as conn:
            units = _units(conn, sid, rev)
            scope_id = units[0]["scope_id"]
            gen = units[0]["generation"]
            pinned_ids = {
                u["unit_id"] for u in units if u["byte_start"] is not None
            }

        query_text = "Alice moved to Berlin in March with her dog."
        # Caller pins the bare model id; the lane derives :hdr:v1.
        out = lane_dense(
            _ctx(
                store,
                scope_id=scope_id,
                generation=gen,
                encoder=encoder,
                pin_encoder_id=encoder.encoder_id,
            ),
            _qv(query_text),
            _slice(cap=10),
        )
        assert out.status == LaneStatus.OK
        assert out.stats["encoder_id"] == mx.unit_vector_encoder_id(
            encoder.encoder_id
        )
        assert out.stats["vector_generation"] == gen
        got = {c.unit_id for c in out.candidates}
        assert got and got <= pinned_ids
        # The unit whose payload slice IS the query text scores strongly
        # (not ~1.0 — the embedded text carries the hdr:v1 header too,
        # which dilutes cosine vs a bare query — by design).
        assert out.candidates[0].raw_score > 0.5
        # And it is exactly the score of the headered text — no hidden
        # query/document asymmetry.
        u0 = next(
            u for u in units
            if u["unit_id"] == out.candidates[0].unit_id
        )
        with store.read() as conn:
            payload = conn.execute(
                "SELECT payload FROM source_revisions"
                " WHERE source_id = ? AND revision = ?",
                (sid, rev),
            ).fetchone()[0]
        doc_text = sj._unit_embed_text(
            {k: u0[k] for k in u0.keys()}, bytes(payload)
        )
        qv = Float32Codec.unpack(
            encoder.encode([query_text])[0], encoder.dimensions
        )
        dv = Float32Codec.unpack(
            encoder.encode([doc_text])[0], encoder.dimensions
        )
        qn = math.sqrt(sum(x * x for x in qv))
        dn = math.sqrt(sum(x * x for x in dv))
        want = sum(a * b for a, b in zip(qv, dv)) / (qn * dn)
        assert out.candidates[0].raw_score == pytest.approx(want, abs=1e-6)

    def test_dense_auto_resolves_single_hdr_space(
        self, store, ingester, encoder
    ):
        """No encoder_id pin: the single stored space (the hdr:v1 id)
        resolves unambiguously."""
        sid, rev = _capture(ingester, _env("plain text payload"))
        _project_then_embed(ingester)
        with store.read() as conn:
            units = _units(conn, sid, rev)
            scope_id, gen = units[0]["scope_id"], units[0]["generation"]
        out = lane_dense(
            _ctx(store, scope_id=scope_id, generation=gen, encoder=encoder),
            _qv("plain text payload"),
            _slice(cap=5),
        )
        assert out.status == LaneStatus.OK
        assert out.stats["encoder_id"].endswith(":hdr:v1")
        assert out.candidates

    def test_header_text_is_deterministic_and_byte_pinned(
        self, store, ingester, encoder
    ):
        """V7-07.09: each embedded text is ``<date> | <speaker> |
        <session>: <payload[byte_start:byte_end]>`` — proven by
        re-deriving the exact string and comparing encoder output
        byte-for-byte against the stored oracle row."""
        text = "Alice moved to Berlin in March with her dog."
        sid, rev = _capture(
            ingester,
            _env(
                text,
                speaker="Alice",
                metadata={"session_id": "sess-42"},
            ),
        )
        _project_then_embed(ingester)
        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        with store.read() as conn:
            units = _units(conn, sid, rev)
            for u in units:
                if u["byte_start"] is None:
                    continue
                rebuilt = {
                    "session_id": u["session_id"],
                    "speaker_canon": u["speaker_canon"],
                    "recorded_at_us": u["recorded_at_us"],
                    "occurred_start_us": u["occurred_start_us"],
                    "byte_start": u["byte_start"],
                    "byte_end": u["byte_end"],
                }
                expected = sj._unit_embed_text(rebuilt, text.encode("utf-8"))
                assert expected is not None
                # Header shape: "YYYY-MM-DD | <speaker> | <session>: ".
                header, _, body = expected.partition(": ")
                assert body == text.encode("utf-8")[
                    u["byte_start"] : u["byte_end"]
                ].decode("utf-8")
                date, speaker, session = header.split(" | ")
                assert date == "2024-03-05"
                assert speaker == "alice"
                assert session == "sess-42"
                want = encoder.encode([expected])[0]
                row = conn.execute(
                    "SELECT vector FROM unit_vectors"
                    " WHERE unit_key = ? AND encoder_id = ?",
                    (u["unit_id"], space),
                ).fetchone()
                assert row is not None
                assert bytes(row[0]) == bytes(want)

    def test_block_no_appends_at_max_plus_one(
        self, store, ingester, encoder
    ):
        """A pre-existing block at block_no=5 in the space pushes the
        embed batch to block_no 6+ (MAX+1 per (enc, scope, gen))."""
        sid, rev = _capture(ingester, _env("append-order source text"))
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
        with store.read() as conn:
            units = _units(conn, sid, rev)
            scope_id, gen = units[0]["scope_id"], units[0]["generation"]
        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        with store.tx() as conn:
            mx.write_block(
                conn, space, scope_id, gen, 5,
                [("preexisting", [0.5] * encoder.dimensions)],
            )
        _drain_one(ingester, JobKind.SOURCE_EMBED, sj.handle_source_embed)
        with store.read() as conn:
            nos = [
                b["block_no"]
                for b in _blocks(conn, space)
                if "preexisting" not in _rowmap_keys(b)
            ]
        assert nos and min(nos) == 6

    def test_int8_threshold_switches_quant_and_oracle_rescores(
        self, store, ingester, encoder, monkeypatch
    ):
        """V7-07.06: the space flips to int8 once its row count crosses
        the threshold (patched low here); the f32 oracle still backs the
        rescore so dense hits are ``int8_rescored``."""
        monkeypatch.setattr(mx, "INT8_ROW_THRESHOLD", 0)
        # units v1.1: a single-turn session no longer mints its
        # byte-identical session unit, so this source's lone pinned turn
        # is the space's only row — a zero threshold flips the space to
        # int8 on the write that pushes the count past it.
        sid, rev = _capture(
            ingester,
            _env(
                "quantized unit vectors",
                metadata={"session_id": "sess-q"},
            ),
        )
        _project_then_embed(ingester)
        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        with store.read() as conn:
            blocks = _blocks(conn, space)
            assert blocks and all(b["quant"] == "int8" for b in blocks)
            assert conn.execute(
                "SELECT COUNT(*) FROM unit_vectors WHERE encoder_id = ?",
                (space,),
            ).fetchone()[0] == sum(b["n_rows"] for b in blocks)
            units = _units(conn, sid, rev)
            scope_id, gen = units[0]["scope_id"], units[0]["generation"]
        out = lane_dense(
            _ctx(
                store,
                scope_id=scope_id,
                generation=gen,
                encoder=encoder,
                pin_encoder_id=encoder.encoder_id,
            ),
            _qv("quantized unit vectors"),
            _slice(cap=10),
        )
        assert out.status == LaneStatus.OK
        assert out.candidates
        kinds = {c.signals.get("score_kind") for c in out.candidates}
        assert kinds <= {"int8_rescored", "int8_approx"}
        assert "int8_rescored" in kinds

    def test_generation_coexistence_and_fence(
        self, store, ingester, encoder
    ):
        """V7-30.02: a generation bump re-derives new unit ids (the id is
        content-addressed with its generation) and new blocks; a reader
        pinned at the old generation still scans the old snapshot, a
        reader at the new pin sees only the new slice."""
        sid, rev = _capture(ingester, _env("coexisting generations text"))
        _project_then_embed(ingester)
        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        with store.read() as conn:
            u1 = _units(conn, sid, rev)
            g1 = u1[0]["generation"]
            ids1 = {u["unit_id"] for u in u1}
            scope_id = u1[0]["scope_id"]

        with store.tx() as conn:
            store.bump_generation(conn)
        rid = ingest_receipt_id(sid, rev)
        with store.tx() as conn:
            sj.enqueue_source_jobs(conn, store, receipt_id=rid)
        _project_then_embed(ingester)

        with store.read() as conn:
            u2 = [
                u for u in _units(conn, sid, rev) if u["generation"] > g1
            ]
            assert u2, "no new-generation unit slice"
            g2 = u2[0]["generation"]
            ids2 = {u["unit_id"] for u in u2}
            assert ids2.isdisjoint(ids1)  # generation is inside the id
            gens = {b["generation"] for b in _blocks(conn, space)}
        assert {g1, g2} <= gens

        # Pin below the bump → the old snapshot.
        out_old = lane_dense(
            _ctx(store, scope_id=scope_id, generation=g1, encoder=encoder),
            _qv("coexisting generations text"),
            _slice(cap=50),
        )
        assert out_old.status == LaneStatus.OK
        assert out_old.stats["vector_generation"] == g1
        assert {c.unit_id for c in out_old.candidates} <= ids1

        # Pin at the bump → only the newest block generation is scanned.
        out_new = lane_dense(
            _ctx(store, scope_id=scope_id, generation=g2, encoder=encoder),
            _qv("coexisting generations text"),
            _slice(cap=50),
        )
        assert out_new.status == LaneStatus.OK
        assert out_new.stats["vector_generation"] == g2
        assert {c.unit_id for c in out_new.candidates} <= ids2
        assert out_new.candidates  # the re-embedded slice is live


# ---------------------------------------------------------------------------
# J05 — owed work is never reported as a successful empty lane
# ---------------------------------------------------------------------------


class TestJ05HonestLag:
    def test_no_encoder_skips_and_dense_reports_hole(
        self, store, bare_ingester, encoder
    ):
        sid, rev = _capture(bare_ingester, _env("never embedded"))
        _drain_one(
            bare_ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project
        )
        _drain_one(
            bare_ingester, JobKind.SOURCE_EMBED, sj.handle_source_embed
        )
        with store.read() as conn:
            ev = conn.execute(
                "SELECT payload_json FROM events"
                " WHERE kind = 'source_embed_skipped'"
            ).fetchone()
            assert ev is not None and "encoder_unavailable" in ev[0]
            assert _blocks(conn, mx.unit_vector_encoder_id(
                encoder.encoder_id
            )) == []
            units = _units(conn, sid, rev)
            scope_id, gen = units[0]["scope_id"], units[0]["generation"]

        out = lane_dense(
            _ctx(
                store,
                scope_id=scope_id,
                generation=gen,
                encoder=encoder,
                pin_encoder_id=encoder.encoder_id,
            ),
            _qv("never embedded"),
            _slice(cap=5),
        )
        # Pinned units with no vectors anywhere → partial, never ok.
        assert out.status == LaneStatus.PARTIAL
        assert out.reason == "units_uncovered"
        assert out.stats["units_uncovered"] >= 1
        assert out.candidates == []

    def test_pending_embed_job_is_partial_not_ok(
        self, store, ingester, encoder
    ):
        """V7-07.12: an owed source_embed job shows up as
        ``coverage.dense = partial(n_pending)`` — the lane never blocks
        and never pretends the empty index is complete."""
        sid, rev = _capture(ingester, _env("awaiting the encoder"))
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
        # The source_embed job is still queued — never leased.
        with store.read() as conn:
            units = _units(conn, sid, rev)
            scope_id, gen = units[0]["scope_id"], units[0]["generation"]
            n_jobs = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind = 'source_embed'"
                " AND state IN ('queued','leased','retry_wait')"
            ).fetchone()[0]
        assert n_jobs == 1

        out = lane_dense(
            _ctx(
                store,
                scope_id=scope_id,
                generation=gen,
                encoder=encoder,
                pin_encoder_id=encoder.encoder_id,
            ),
            _qv("awaiting the encoder"),
            _slice(cap=5),
        )
        assert out.status == LaneStatus.PARTIAL
        assert out.reason == "embed_pending"
        assert out.stats["n_pending"] >= 1
        assert out.candidates == []

    def test_uncovered_units_after_permanent_embed_failure(
        self, store, ingester, encoder
    ):
        """A permanently-failed embed leaves the hole visible even with
        no job pending (committed work can't be claimed)."""
        sid, rev = _capture(ingester, _env("doomed embedding"))
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
        job = ingester.jobs.lease(
            None, [JobKind.SOURCE_EMBED], owner="w1", limit=1
        )[0]
        with store.tx() as conn:
            ingester.jobs.fail(
                conn, job["job_id"], "w1", job["generation"],
                "vector_invalid", False,
            )
        with store.read() as conn:
            units = _units(conn, sid, rev)
            scope_id, gen = units[0]["scope_id"], units[0]["generation"]
        out = lane_dense(
            _ctx(
                store,
                scope_id=scope_id,
                generation=gen,
                encoder=encoder,
                pin_encoder_id=encoder.encoder_id,
            ),
            _qv("doomed embedding"),
            _slice(cap=5),
        )
        assert out.status == LaneStatus.PARTIAL
        assert out.reason == "units_uncovered"
        assert "n_pending" not in out.stats


# ---------------------------------------------------------------------------
# J06 — erasure removes block + oracle rows; dense never emits erased units
# ---------------------------------------------------------------------------


class TestJ06Erasure:
    def test_delete_source_removes_blocks_and_oracle(
        self, store, ingester, encoder
    ):
        sid, rev = _capture(ingester, _env("erase me completely"))
        _project_then_embed(ingester)
        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        with store.read() as conn:
            units = _units(conn, sid, rev)
            ids = {u["unit_id"] for u in units}
            scope_id, gen = units[0]["scope_id"], units[0]["generation"]
            assert ids
            assert conn.execute(
                "SELECT COUNT(*) FROM unit_vectors WHERE encoder_id = ?",
                (space,),
            ).fetchone()[0] > 0

        with store.tx() as conn:
            receipt = closure_v7.delete_source_v7(conn, sid)

        with store.read() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM units WHERE unit_id IN"
                f" ({','.join('?' * len(ids))})",
                sorted(ids),
            ).fetchone()[0] == 0
            # Block rows for the scope are gone or rewritten without the
            # erased keys.
            for b in _blocks(conn, space):
                assert not (set(_rowmap_keys(b)) & ids)
            assert conn.execute(
                "SELECT COUNT(*) FROM unit_vectors WHERE unit_key IN"
                f" ({','.join('?' * len(ids))})",
                sorted(ids),
            ).fetchone()[0] == 0
        n_pinned = len(
            {u["unit_id"] for u in units if u["byte_start"] is not None}
        )
        assert receipt["deleted"].get("unit_vectors", 0) == n_pinned

        out = lane_dense(
            _ctx(store, scope_id=scope_id, generation=gen, encoder=encoder),
            _qv("erase me completely"),
            _slice(cap=10),
        )
        assert not ({c.unit_id for c in out.candidates} & ids)

    def test_erasure_rewrites_shared_block_not_sibling_keys(
        self, store, ingester, encoder
    ):
        """Two sources sharing one block space: erasing one rewrites the
        block minus its keys; the sibling's rows and oracle survive."""
        a, _ = _capture(ingester, _env("first source stays"))
        _project_then_embed(ingester)
        b, _ = _capture(ingester, _env("second source erased"))
        _project_then_embed(ingester)
        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        with store.read() as conn:
            ua = _units(conn, a, 1)
            ub = _units(conn, b, 1)
            ids_a = {u["unit_id"] for u in ua}
            ids_b = {u["unit_id"] for u in ub}
            scope_id, gen = ua[0]["scope_id"], ua[0]["generation"]

        with store.tx() as conn:
            closure_v7.delete_source_v7(conn, b)

        with store.read() as conn:
            remaining = {
                k for blk in _blocks(conn, space) for k in _rowmap_keys(blk)
            }
            assert not (remaining & ids_b)
            assert ids_a - {
                u["unit_id"] for u in ua if u["byte_start"] is None
            } <= remaining
            # Oracle: only the erased units' rows are gone.
            orow = conn.execute(
                "SELECT unit_key FROM unit_vectors WHERE encoder_id = ?",
                (space,),
            ).fetchall()
            okeys = {r[0] for r in orow}
            assert not (okeys & ids_b)
            assert okeys & ids_a

        out = lane_dense(
            _ctx(store, scope_id=scope_id, generation=gen, encoder=encoder),
            _qv("first source stays"),
            _slice(cap=20),
        )
        got = {c.unit_id for c in out.candidates}
        assert not (got & ids_b)
        assert got & ids_a


# ---------------------------------------------------------------------------
# drift fence — encode snapshot vs fenced commit
# ---------------------------------------------------------------------------


class TestDriftFence:
    def test_commit_aborts_on_unit_slice_drift(
        self, store, ingester, encoder
    ):
        """_PlanStale: a reprojected slice between snapshot and commit
        must not let stale vectors land."""
        sid, rev = _capture(ingester, _env("drift-sensitive payload"))
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
        payload = bytes(ingester.sources.payload(sid, rev) or b"")
        plan = sj._prepare_unit_vectors(
            ingester, encoder, sid, rev, payload
        )
        assert plan["units"] >= 1

        # Re-project at a bumped generation — the live slice changes.
        with store.tx() as conn:
            store.bump_generation(conn)
        rid = ingest_receipt_id(sid, rev)
        with store.tx() as conn:
            sj.enqueue_source_jobs(conn, store, receipt_id=rid)
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)

        gen = store.projection_generation()
        with pytest.raises(sj._PlanStale):
            with store.tx() as conn:
                sj._commit_unit_vectors(
                    conn,
                    space_id=mx.unit_vector_encoder_id(encoder.encoder_id),
                    plan=plan,
                    source_id=sid,
                    revision=rev,
                    generation=gen,
                )

        # And the fresh drain embeds the NEW slice only — stale-gen keys
        # never enter the new generation's blocks (the earlier plan was
        # aborted before any of its vectors could land). Two embed jobs
        # are queued (the original g1 plus the bumped g2 re-enqueue) —
        # drain both; each resolves the live slice at commit time.
        for job in ingester.jobs.lease(
            None, [JobKind.SOURCE_EMBED], owner="w1", limit=8
        ):
            sj.handle_source_embed(job, "w1", ingester)
        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        with store.read() as conn:
            live_ids = {
                r[0]
                for r in conn.execute(
                    "SELECT unit_id FROM units WHERE source_id = ?"
                    " AND revision = ? AND generation = ?",
                    (sid, rev, gen),
                ).fetchall()
            }
            new_keys = {
                k
                for b in _blocks(conn, space)
                if b["generation"] == gen
                for k in _rowmap_keys(b)
            }
        assert new_keys and new_keys <= live_ids


# ---------------------------------------------------------------------------
# generation-strand healing + idempotent append + stale-sweep surgery
# ---------------------------------------------------------------------------


class TestGenerationHealing:
    def _requeue(self, store, sid: str, rev: int) -> None:
        rid = ingest_receipt_id(sid, rev)
        with store.tx() as conn:
            sj.enqueue_source_jobs(conn, store, receipt_id=rid)

    def test_reembed_at_newer_generation_heals_dense_space(
        self, store, ingester, encoder
    ):
        """Commit-generation stamping (V75-03.02 "at the pinned
        generation"): a non-rebuild bump strands source A's blocks below
        the newest snapshot — dense reports the hole as PARTIAL, and a
        re-embed of A at the new generation heals it (the same unit ids
        land in the scanned snapshot)."""
        a, ra = _capture(ingester, _env("stranded source alpha"))
        _project_then_embed(ingester)
        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        with store.read() as conn:
            ua = _units(conn, a, ra)
            scope_id, g1 = ua[0]["scope_id"], ua[0]["generation"]
            ids_a = {u["unit_id"] for u in ua if u["byte_start"] is not None}

        # A non-rebuild generation bump (no reprojection owed) — then a
        # second source embeds at the new generation, splitting the space.
        with store.tx() as conn:
            store.bump_generation(conn)
        b, rb = _capture(ingester, _env("newest snapshot beta"))
        _project_then_embed(ingester)
        with store.read() as conn:
            g2 = store.projection_generation()
            assert g2 > g1
            gens = {b_["generation"] for b_ in _blocks(conn, space)}
            assert gens == {g1, g2}

        # The newest snapshot covers only B — A's live units are
        # stranded below it: honest PARTIAL, never silent.
        out = lane_dense(
            _ctx(
                store, scope_id=scope_id, generation=g2, encoder=encoder,
                pin_encoder_id=encoder.encoder_id,
            ),
            _qv("stranded source alpha"),
            _slice(cap=20),
        )
        assert out.status == LaneStatus.PARTIAL
        assert out.reason == "units_uncovered"
        assert out.stats["units_uncovered"] == len(ids_a)

        # Re-queue A's jobs; drain ONLY source_embed — A's units were
        # never reprojected, so the live slice is still the gen-1 ids.
        # The commit stamps the resolved (new) generation, landing those
        # ids inside the scanned snapshot.
        self._requeue(store, a, ra)
        _drain_one(ingester, JobKind.SOURCE_EMBED, sj.handle_source_embed)
        out2 = lane_dense(
            _ctx(
                store, scope_id=scope_id, generation=g2, encoder=encoder,
                pin_encoder_id=encoder.encoder_id,
            ),
            _qv("stranded source alpha"),
            _slice(cap=20),
        )
        assert out2.status == LaneStatus.OK
        assert out2.stats["vector_generation"] == g2
        assert {c.unit_id for c in out2.candidates} & ids_a

    def test_duplicate_embed_jobs_do_not_double_append(
        self, store, ingester, encoder
    ):
        """Two live embed jobs for the same revision (a stale sibling
        minted before a generation bump) converge on one copy of each
        unit vector — the commit skips rowmap keys already present."""
        sid, rev = _capture(ingester, _env("deduplicated embed"))
        _drain_one(ingester, JobKind.SOURCE_PROJECT, sj.handle_source_project)
        self._requeue(store, sid, rev)
        with store.tx() as conn:
            store.bump_generation(conn)
        self._requeue(store, sid, rev)
        jobs = ingester.jobs.lease(
            None, [JobKind.SOURCE_EMBED], owner="w1", limit=8
        )
        assert len(jobs) == 2
        for job in jobs:
            sj.handle_source_embed(job, "w1", ingester)

        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        with store.read() as conn:
            units = _units(conn, sid, rev)
            live = {
                u["unit_id"] for u in units if u["byte_start"] is not None
            }
            keys = [
                k for b in _blocks(conn, space) for k in _rowmap_keys(b)
            ]
        assert sorted(keys) == sorted(live)  # exactly once each

    def test_stale_sweep_scrubs_surviving_generation_blocks(
        self, store, ingester, encoder
    ):
        """A block stamped at a newer commit generation can carry unit
        ids minted at a swept generation — the stale-generation sweep
        must rewrite those blocks, not just delete same-generation
        ones."""
        sid, rev = _capture(ingester, _env("swept stale vectors"))
        _project_then_embed(ingester)
        space = mx.unit_vector_encoder_id(encoder.encoder_id)
        with store.read() as conn:
            units = _units(conn, sid, rev)
            ids1 = {u["unit_id"] for u in units}
            g1 = units[0]["generation"]

        with store.tx() as conn:
            store.bump_generation(conn)
        self._requeue(store, sid, rev)
        # Embed without reproject: the gen-1 unit ids land in a block
        # stamped at the new commit generation.
        _drain_one(ingester, JobKind.SOURCE_EMBED, sj.handle_source_embed)
        with store.read() as conn:
            g2 = store.projection_generation()
            gen2_keys = {
                k
                for b in _blocks(conn, space)
                if b["generation"] == g2
                for k in _rowmap_keys(b)
            }
            assert gen2_keys & ids1

        with store.tx() as conn:
            closure_v7.sweep_stale_generations_v7(conn, keep_generation=g2)
        with store.read() as conn:
            remaining = {
                k for b in _blocks(conn, space) for k in _rowmap_keys(b)
            }
            assert not (remaining & ids1)
            assert conn.execute(
                "SELECT COUNT(*) FROM unit_vectors WHERE unit_key IN"
                f" ({','.join('?' * len(ids1))})",
                sorted(ids1),
            ).fetchone()[0] == 0
