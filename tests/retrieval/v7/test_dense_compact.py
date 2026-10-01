"""V8 dense compaction + dense-slot emit-side tests (SPEC_V8 §08, §25.5).

Scenarios K43–K46 + K48:

- K43: a fragmented space (micro-blocks, one row each) compacts to
  ``ceil(rows / MAX_BLOCK_ROWS)`` contiguous blocks; ``matrix.scan``
  top-k is identical before and after.
- K44: a unit erased between plan and commit is absent from the
  compacted blocks — live keys re-read under the write lock (V8-19.03
  closure invariant).
- K45: scan stats show the vectorized path engaged per block
  (``vectorized_blocks == blocks_scanned`` when numpy is present;
  honest-zero otherwise) and the scan envelope is recorded.
- K46: candidates clearing the null-model floor are emitted marked
  ``signals["dense_slot"]`` (emit side — slot enforcement is fusion's);
  below-floor items never appear.
- K48: spaces are keyed ``(encoder_id, scope_id, generation)`` — a
  header-version change mints a separate space; compaction never merges
  spaces.

Also covered: ``dense_compaction_epoch/<space>`` meta recording +
monotonic bump, idempotent re-runs, ``coverage_dense`` stats block,
``dense.N_d`` arm override, and the ``dense_compact`` job handler.
"""

from __future__ import annotations

import contextlib
import math
import random
import sqlite3
import threading
import types as _types

import pytest

from verbatim.core.types import VerbatimError
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
from verbatim.embeddings.hashing import HashingEncoder
from verbatim.retrieval.v7 import dense_compact as dc
from verbatim.retrieval.v7.dense import lane_dense
from verbatim.storage.schema import DDL_V1
from verbatim.storage.schema_v7 import ensure_v7_additive

_ORACLE_DDL = """
CREATE TABLE unit_vectors (
    unit_key TEXT NOT NULL,
    encoder_id TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (unit_key, encoder_id)
);
"""

_SCOPE = "scope-a"
_ENC = "test:enc:v1"
_DIMS = 64


class _TxStore:
    """tx()/read() shim over one explicit-transaction in-memory conn.

    ``isolation_level=None`` gives manual BEGIN/COMMIT — the same
    BEGIN IMMEDIATE semantics ``Store._write_tx`` enforces — so the
    compaction commit runs under a real write transaction.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()
        self._event_us = 1_000_000

    @contextlib.contextmanager
    def read(self):
        yield self._conn

    @contextlib.contextmanager
    def tx(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def next_event_us(self) -> int:
        self._event_us += 1
        return self._event_us


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(DDL_V1)
    ensure_v7_additive(c)
    c.executescript(_ORACLE_DDL)
    yield c
    c.close()


@pytest.fixture()
def store(conn):
    return _TxStore(conn)


def _rand_vec(rng: random.Random, dims: int = _DIMS) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(dims)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _micro_blocks(conn, vecs, *, encoder_id=_ENC, scope=_SCOPE, gen=1):
    """One block per unit — the D8-09 fragmentation shape."""
    for bno, key in enumerate(sorted(vecs)):
        mx.write_block(conn, encoder_id, scope, gen, bno, [(key, vecs[key])])


def _scan_topk(conn, q, k=25, *, encoder_id=_ENC, scope=_SCOPE, gen=1):
    res = mx.scan(conn, encoder_id, scope, gen, q, k=k, use_numpy=False)
    return res


# ---------------------------------------------------------------------------
# K43 — fragmentation coalesces, scan identical
# ---------------------------------------------------------------------------


def test_k43_micro_blocks_compact_to_one(conn, store):
    rng = random.Random(7)
    vecs = {f"u{i:04d}": _rand_vec(rng) for i in range(300)}
    _micro_blocks(conn, vecs)
    assert dc.plan_space(conn, _ENC, _SCOPE, 1).needed
    q = _rand_vec(random.Random(99))
    before = list(_scan_topk(conn, q, k=25))

    rep = dc.compact_space(store, _ENC, _SCOPE, 1)
    assert rep["compacted"] and rep["rows"] == 300
    assert rep["blocks_before"] == 300 and rep["blocks_after"] == 1
    assert rep["epoch"] == 1

    blocks = list(mx.iter_blocks(conn, _ENC, _SCOPE, 1))
    assert len(blocks) == 1 and blocks[0].n_rows == 300
    assert list(_scan_topk(conn, q, k=25)) == before


def test_k43_multiblock_rewrite_ordered(conn, store, monkeypatch):
    """Rows > MAX_BLOCK_ROWS → ``ceil(N/B)`` blocks; keys ordered by
    unit_id across the fresh ``block_no`` range."""
    monkeypatch.setattr(mx, "MAX_BLOCK_ROWS", 100)
    rng = random.Random(11)
    vecs = {f"u{i:04d}": _rand_vec(rng) for i in range(250)}
    _micro_blocks(conn, vecs)
    q = _rand_vec(random.Random(5))
    before = list(_scan_topk(conn, q, k=40))
    rep = dc.compact_space(store, _ENC, _SCOPE, 1)
    assert rep["blocks_after"] == 3  # ceil(250/100)
    keys = [k for b in mx.iter_blocks(conn, _ENC, _SCOPE, 1) for k in b.keys]
    assert keys == sorted(vecs)
    assert list(_scan_topk(conn, q, k=40)) == before


def test_compaction_idempotent(conn, store):
    rng = random.Random(3)
    vecs = {f"u{i}": _rand_vec(rng) for i in range(50)}
    _micro_blocks(conn, vecs)
    rep1 = dc.compact_space(store, _ENC, _SCOPE, 1)
    assert rep1["compacted"] and rep1["epoch"] == 1
    rep2 = dc.compact_space(store, _ENC, _SCOPE, 1)
    assert rep2["compacted"] is False and rep2["reason"] == "below_threshold"
    assert dc.compaction_epoch(conn, _ENC, _SCOPE, 1) == 1
    # New fragmentation re-triggers and bumps the epoch again.
    extra = {f"w{i}": _rand_vec(random.Random(100 + i)) for i in range(30)}
    base = max(b.block_no for b in mx.iter_blocks(conn, _ENC, _SCOPE, 1)) + 1
    for bno, key in enumerate(sorted(extra)):
        mx.write_block(conn, _ENC, _SCOPE, 1, base + bno, [(key, extra[key])])
    rep3 = dc.compact_space(store, _ENC, _SCOPE, 1)
    assert rep3["compacted"] and rep3["epoch"] == 2
    assert mx.block_row_keys(conn, _ENC, _SCOPE, 1) == set(vecs) | set(extra)


def test_plan_thresholds(conn):
    rng = random.Random(2)
    # Few fat blocks: under B_max with healthy mean → no compaction.
    for bno in range(3):
        rows = [
            (f"u{bno}-{i}", _rand_vec(rng)) for i in range(300)
        ]
        mx.write_block(conn, _ENC, _SCOPE, 1, bno, rows)
    plan = dc.plan_space(conn, _ENC, _SCOPE, 1)
    assert not plan.needed and plan.reason == "below_threshold"
    # Same block count, starved rows → mean_rows trigger.
    for bno in range(3, 6):
        mx.write_block(
            conn, "sparse:enc:v1", _SCOPE, 1, bno - 3,
            [(f"s{bno}-{i}", _rand_vec(rng)) for i in range(10)],
        )
    plan2 = dc.plan_space(conn, "sparse:enc:v1", _SCOPE, 1)
    assert plan2.needed and plan2.mean_rows == 10.0
    # b_max override: three fat blocks still exceed b_max=2.
    assert dc.plan_space(conn, _ENC, _SCOPE, 1, b_max=2).needed
    # Empty space plans a clean no-op.
    empty = dc.plan_space(conn, "none:enc:v0", _SCOPE, 1)
    assert not empty.needed and empty.reason == "empty_space"


def test_compact_validation(conn, store):
    with pytest.raises(VerbatimError):
        dc.compact_space(store, _ENC, _SCOPE, -1)
    with pytest.raises(VerbatimError):
        dc.compact_space(store, _ENC, _SCOPE, 1, b_max=-1)
    with pytest.raises(VerbatimError):
        dc.compact_space(store, _ENC, _SCOPE, 1, min_mean_rows=0)
    # Missing block table → honest no-op report, never a crash.
    conn.execute("DROP TABLE unit_vectors_block")
    rep = dc.compact_space(store, _ENC, _SCOPE, 1)
    assert rep["compacted"] is False and rep["reason"] == "no_block_table"


# ---------------------------------------------------------------------------
# K44 — purged key never resurrected
# ---------------------------------------------------------------------------


def test_k44_erased_between_plan_and_commit(conn, store):
    rng = random.Random(17)
    vecs = {f"u{i:03d}": _rand_vec(rng) for i in range(40)}
    _micro_blocks(conn, vecs)
    plan = dc.plan_space(conn, _ENC, _SCOPE, 1)
    assert plan.needed and "u-purged" not in vecs

    # A unit minted into the space, then erased between plan and commit —
    # the purge path's own tx already swept its block row (micro-block →
    # whole-row DELETE, exactly _sweep_vector_blocks' empty-keep case).
    mx.write_block(
        conn, _ENC, _SCOPE, 1, len(vecs), [("u-gone", _rand_vec(rng))]
    )
    conn.execute(
        "DELETE FROM unit_vectors_block"
        " WHERE encoder_id = ? AND scope_id = ? AND generation = ?"
        "   AND block_no = ?",
        (_ENC, _SCOPE, 1, len(vecs)),
    )
    # The stale plan saw u-gone; the commit re-reads live keys under the
    # write lock — the swept key is absent from the compacted blocks.
    assert plan.n_rows == 40  # plan snapshot predates u-gone anyway
    with store.tx() as c:
        rep = dc.compact_space_in_tx(c, _ENC, _SCOPE, 1)
    assert rep["compacted"] and rep["rows"] == 40
    keys = mx.block_row_keys(conn, _ENC, _SCOPE, 1)
    assert "u-gone" not in keys and keys == set(vecs)


def test_k44_surgical_rowmap_sweep(conn, store):
    """Erase inside a shared block (the purge path's UPDATE case): the
    compacted space must not carry the swept key's row."""
    rng = random.Random(23)
    vecs = {f"u{i:02d}": _rand_vec(rng) for i in range(20)}
    mx.write_block(
        conn, _ENC, _SCOPE, 1, 0,
        [(k, vecs[k]) for k in sorted(vecs)],
    )
    for bno in range(1, 5):  # fragment: 4 micro-blocks + the shared one
        mx.write_block(
            conn, _ENC, _SCOPE, 1, bno,
            [(f"x{bno}", _rand_vec(rng))],
        )
    # Purge rewrites the shared block minus u-07 (closure_v7 surgery).
    blk = mx.read_block(conn, _ENC, _SCOPE, 1, 0)
    keep = [i for i, k in enumerate(blk.keys) if k != "u-07"]
    data_b = blk.data.tobytes()
    import struct as _st

    row_bytes = blk.dims * 4
    new_data = b"".join(
        data_b[i * row_bytes : (i + 1) * row_bytes] for i in keep
    )
    scales = conn.execute(
        "SELECT scale_blob FROM unit_vectors_block"
        " WHERE encoder_id=? AND scope_id=? AND generation=? AND block_no=?",
        (_ENC, _SCOPE, 1, 0),
    ).fetchone()[0]
    new_scale = b"".join(
        bytes(scales)[i * 8 : (i + 1) * 8] for i in keep
    )
    new_rowmap = _st.pack("<I", len(keep)) + b"".join(
        _st.pack("<I", len(blk.keys[i].encode())) + blk.keys[i].encode()
        for i in keep
    )
    conn.execute(
        "UPDATE unit_vectors_block SET n_rows=?, scale_blob=?,"
        " data_blob=?, rowmap_blob=?"
        " WHERE encoder_id=? AND scope_id=? AND generation=? AND block_no=?",
        (len(keep), new_scale, new_data, new_rowmap, _ENC, _SCOPE, 1, 0),
    )
    rep = dc.compact_space(store, _ENC, _SCOPE, 1)
    assert rep["compacted"]
    keys = mx.block_row_keys(conn, _ENC, _SCOPE, 1)
    assert "u-07" not in keys
    assert keys == (set(vecs) - {"u-07"}) | {f"x{b}" for b in range(1, 5)}


# ---------------------------------------------------------------------------
# V8-08.03 / K48 — spaces never merge
# ---------------------------------------------------------------------------


def test_k48_header_spaces_never_merge(conn, store):
    rng = random.Random(31)
    enc_v1 = "hashing:subword-ngram:v1:hdr:v1"
    enc_v2 = "hashing:subword-ngram:v1:hdr:v2"
    a = {f"a{i:03d}": _rand_vec(rng) for i in range(30)}
    b = {f"b{i:03d}": _rand_vec(rng) for i in range(30)}
    _micro_blocks(conn, a, encoder_id=enc_v1)
    _micro_blocks(conn, b, encoder_id=enc_v2)
    rep = dc.compact_space(store, enc_v1, _SCOPE, 1)
    assert rep["compacted"]
    # The hdr:v2 space is untouched — count and keys intact.
    b_blocks = list(mx.iter_blocks(conn, enc_v2, _SCOPE, 1))
    assert len(b_blocks) == 30
    assert mx.block_row_keys(conn, enc_v2, _SCOPE, 1) == set(b)
    # And the compacted v1 space carries only its own keys.
    assert mx.block_row_keys(conn, enc_v1, _SCOPE, 1) == set(a)
    a_blocks = list(mx.iter_blocks(conn, enc_v1, _SCOPE, 1))
    assert len(a_blocks) == 1
    q = _rand_vec(random.Random(3))
    res = mx.scan(conn, enc_v1, _SCOPE, 1, q, k=100, use_numpy=False)
    assert all(k.startswith("a") for k, _ in res)


def test_list_spaces_filters(conn):
    rng = random.Random(5)
    _micro_blocks(conn, {f"a{i}": _rand_vec(rng) for i in range(5)})
    _micro_blocks(conn, {f"b{i}": _rand_vec(rng) for i in range(5)},
                  encoder_id="other:enc:v2")
    _micro_blocks(conn, {f"c{i}": _rand_vec(rng) for i in range(5)},
                  scope="scope-b")
    assert len(dc.list_spaces(conn)) == 3
    assert dc.list_spaces(conn, scope_id=_SCOPE) == [
        dc.SpaceRef("other:enc:v2", _SCOPE, 1),
        dc.SpaceRef(_ENC, _SCOPE, 1),
    ]
    assert dc.list_spaces(conn, encoder_id="other:enc:v2") == [
        dc.SpaceRef("other:enc:v2", _SCOPE, 1),
    ]
    assert dc.list_spaces(conn, generation=2) == []


def test_compact_spaces_multi(conn, store):
    rng = random.Random(9)
    _micro_blocks(conn, {f"a{i}": _rand_vec(rng) for i in range(10)})
    _micro_blocks(conn, {f"b{i}": _rand_vec(rng) for i in range(10)},
                  encoder_id="other:enc:v2")
    reports = dc.compact_spaces(store, iter(dc.list_spaces(conn)))
    assert len(reports) == 2 and all(r["compacted"] for r in reports)
    for enc in (_ENC, "other:enc:v2"):
        assert len(list(mx.iter_blocks(conn, enc, _SCOPE, 1))) == 1
        assert dc.compaction_epoch(conn, enc, _SCOPE, 1) == 1


# ---------------------------------------------------------------------------
# K45 — vectorized path engaged; envelope recorded
# ---------------------------------------------------------------------------


def _units(conn, rows):
    for r in rows:
        conn.execute(
            "INSERT INTO units (unit_id, source_id, revision, scope_id,"
            " kind, speaker_canon, generation) VALUES (?,?,?,?,?,?,?)", r
        )


def _hashing_encoder():
    return HashingEncoder(_types.SimpleNamespace(artifact_revision=None))


def _ctx(conn, *, encoder=None, eligible=None, manifest_extra=None):
    manifest = {}
    if encoder is not None:
        manifest["query_encoder"] = encoder
        manifest["encoder_id"] = getattr(encoder, "encoder_id", None)
    if manifest_extra:
        manifest.update(manifest_extra)
    return LaneContextV7(
        store=_TxStore(conn),
        scope_id=_SCOPE,
        generation=1,
        eligible=eligible,
        query_time_us=1_000_000,
        profile="local_memory",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="local_memory",
            lanes=(), lane_weights={},
        ),
        manifest=manifest,
    )


def _qv(text):
    return QueryViewV7(
        query=text,
        norm=NormAnalysis(analyzer_id="norm/v2", terms=(), identifiers=()),
        intent=IntentResult(primary=IntentClass.LOOKUP,
                            classes=(IntentClass.LOOKUP,)),
    )


def _slice(cap=40):
    return LaneSlice(deadline_ms=10_000.0, cap=cap)


def _text_space(conn, enc):
    """Real-encoder corpus: blocks fragmented one-per-unit."""
    texts = {
        "u-target": "alice moved to berlin in march 2024 with her dog",
        "u-close": "alice moved to berlin in 2024",
        "u-held": "alice moved to berlin in march 2024 with her dog",
        **{f"u-{i:02d}": f"unrelated note {i} about topic {i}" for i in range(28)},
    }
    _units(
        conn,
        [
            (k, f"src-{k}", 1, _SCOPE, "turn",
             "held" if k == "u-held" else "alice", 1)
            for k in texts
        ],
    )
    blobs = enc.encode(list(texts.values()))
    for bno, (key, blob) in enumerate(zip(texts, blobs)):
        mx.write_block(conn, enc.encoder_id, _SCOPE, 1, bno, [(key, blob)])
    return texts


def test_k45_vectorized_stats_and_epoch(conn, store):
    enc = _hashing_encoder()
    _text_space(conn, enc)
    dc.compact_space(store, enc.encoder_id, _SCOPE, 1)
    out = lane_dense(
        _ctx(conn, encoder=enc),
        _qv("alice moved to berlin in march 2024 with her dog"),
        _slice(cap=10),
    )
    assert out.status == LaneStatus.OK
    assert out.stats["blocks_scanned"] == 1
    np = pytest.importorskip("numpy")
    assert out.stats["numpy"] is True
    assert out.stats["vectorized_blocks"] == out.stats["blocks_scanned"]
    assert out.stats["scan_ms"] >= 0.0  # envelope recorded
    assert out.stats["compaction_epoch"] == 1
    cov = out.stats["coverage_dense"]
    assert cov["slots"] == 3
    assert cov["floor"] == pytest.approx(out.stats["noise_floor"])
    assert cov["blocks_scanned"] == 1
    assert cov["compaction_epoch"] == 1


def test_scan_matches_pre_post_compaction_both_paths(conn, store):
    """Vectorized and scalar scans produce identical rankings on the
    compacted space — and identical to the fragmented pre-image."""
    pytest.importorskip("numpy")
    rng = random.Random(41)
    vecs = {f"u{i:04d}": _rand_vec(rng) for i in range(200)}
    _micro_blocks(conn, vecs)
    q = _rand_vec(random.Random(13))
    pre_np = list(mx.scan(conn, _ENC, _SCOPE, 1, q, k=40, use_numpy=True))
    pre_py = list(mx.scan(conn, _ENC, _SCOPE, 1, q, k=40, use_numpy=False))
    dc.compact_space(store, _ENC, _SCOPE, 1)
    post_np = list(mx.scan(conn, _ENC, _SCOPE, 1, q, k=40, use_numpy=True))
    post_py = list(mx.scan(conn, _ENC, _SCOPE, 1, q, k=40, use_numpy=False))
    # Identical rankings across paths and across the compaction boundary;
    # scores agree to float noise (matmul vs sequential sum ordering).
    assert (
        [k for k, _ in pre_np]
        == [k for k, _ in pre_py]
        == [k for k, _ in post_np]
        == [k for k, _ in post_py]
    )
    for (k1, s1), (k2, s2), (k3, s3), (k4, s4) in zip(
        pre_np, pre_py, post_np, post_py
    ):
        assert k1 == k2 == k3 == k4
        assert s1 == pytest.approx(s2, abs=1e-9)
        assert s1 == pytest.approx(s3, abs=1e-9)
        assert s1 == pytest.approx(s4, abs=1e-9)


# ---------------------------------------------------------------------------
# K46 — dense_slot emit-side marking
# ---------------------------------------------------------------------------


def test_k46_dense_slot_marks_floor_clearing_candidates(conn, store):
    enc = _hashing_encoder()
    _text_space(conn, enc)
    out = lane_dense(
        _ctx(conn, encoder=enc, manifest_extra={"dense_N_d": 10}),
        _qv("alice moved to berlin in march 2024 with her dog"),
        _slice(cap=10),
    )
    assert out.status == LaneStatus.OK
    # Three candidates clear the floor; all three emitted under slot
    # reservation (dense.N_d=10 covers them all).
    assert len(out.candidates) == 3
    assert all(c.signals.get("dense_slot") is True for c in out.candidates)
    assert out.stats["dense_slots"] == 3
    assert out.stats["dense_N_d"] == 10
    # Below-floor items are never emitted — nothing to mark.
    assert out.stats["below_floor"] == 7
    cov = out.stats["coverage_dense"]
    assert cov["slots"] == 3 and cov["floor"] == out.stats["noise_floor"]
    assert cov["blocks_scanned"] >= 1
    assert cov["compaction_epoch"] == 0  # never compacted


def test_k46_dense_slot_arm_override(conn):
    """``dense.N_d`` override caps the marked prefix; 0 disables it."""
    enc = _hashing_encoder()
    _text_space(conn, enc)
    out = lane_dense(
        _ctx(conn, encoder=enc, manifest_extra={"dense_N_d": 2}),
        _qv("alice moved to berlin in march 2024 with her dog"),
        _slice(cap=10),
    )
    marked = [c for c in out.candidates if c.signals.get("dense_slot")]
    assert len(out.candidates) == 3 and len(marked) == 2
    assert [c.unit_id for c in marked] == ["u-held", "u-target"]
    assert out.stats["dense_slots"] == 2 and out.stats["dense_N_d"] == 2
    out0 = lane_dense(
        _ctx(conn, encoder=enc, manifest_extra={"dense_N_d": 0}),
        _qv("alice moved to berlin in march 2024 with her dog"),
        _slice(cap=10),
    )
    assert len(out0.candidates) == 3
    assert not any(c.signals.get("dense_slot") for c in out0.candidates)
    assert out0.stats["dense_slots"] == 0


def test_k46_dense_slot_default_off(conn):
    """SPEC_V8_5 §06 don't-ship: ``dense.N_d`` ships at 0 — the pack
    measured 0/137 unique gold rescues under slot reservation."""
    enc = _hashing_encoder()
    _text_space(conn, enc)
    out = lane_dense(
        _ctx(conn, encoder=enc),
        _qv("alice moved to berlin in march 2024 with her dog"),
        _slice(cap=10),
    )
    assert out.status == LaneStatus.OK
    assert out.stats["dense_N_d"] == 0
    assert out.stats["dense_slots"] == 0
    assert not any(c.signals.get("dense_slot") for c in out.candidates)


def test_k46_below_floor_no_slots(conn):
    enc = _hashing_encoder()
    _text_space(conn, enc)
    out = lane_dense(
        _ctx(conn, encoder=enc),
        _qv("quantum chromodynamics lattice gauge coupling"),
        _slice(cap=10),
    )
    assert out.candidates == []
    assert out.stats["coverage_dense"]["slots"] == 0


def test_lane_coverage_dense_early_exits(conn):
    enc = _hashing_encoder()
    # No blocks at all → zeros + floor + null epoch, never a missing key.
    _units(conn, [("u-a", "s", 1, _SCOPE, "turn", "alice", 1)])
    out = lane_dense(_ctx(conn, encoder=enc), _qv("x"), _slice())
    cov = out.stats["coverage_dense"]
    assert cov["slots"] == 0
    assert cov["floor"] == pytest.approx(3.0 / math.sqrt(enc.dimensions))
    assert cov["blocks_scanned"] == 0
    assert cov["compaction_epoch"] is None


def test_lane_coverage_dense_no_index():
    enc = _hashing_encoder()
    # A fresh store that never provisioned the block table → unavailable
    # with the coverage block still emitted (blocks_scanned = 0).
    bare = sqlite3.connect(":memory:", isolation_level=None)
    bare.executescript(DDL_V1)
    ensure_v7_additive(bare)
    bare.execute("DROP TABLE unit_vectors_block")
    out = lane_dense(_ctx(bare, encoder=enc), _qv("x"), _slice())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_vector_index"
    assert out.stats["coverage_dense"]["blocks_scanned"] == 0
    assert out.stats["coverage_dense"]["compaction_epoch"] is None
    bare.close()


def test_dense_compaction_derivation_status(conn, store):
    rng = random.Random(61)
    # No spaces yet → ok (evaluated, zero debt).
    assert dc.dense_compaction_derivation(conn, _SCOPE) == "ok"
    _micro_blocks(conn, {f"a{i:02d}": _rand_vec(rng) for i in range(12)})
    _micro_blocks(
        conn, {f"b{i:02d}": _rand_vec(rng) for i in range(12)},
        encoder_id="other:enc:v2",
    )
    assert dc.dense_compaction_derivation(conn, _SCOPE) == "partial(2)"
    # Compacting one space discharges its debt only.
    dc.compact_space(store, _ENC, _SCOPE, 1)
    assert dc.dense_compaction_derivation(conn, _SCOPE) == "partial(1)"
    # Generation pin hides debt above it; foreign scope unaffected.
    assert dc.dense_compaction_derivation(conn, _SCOPE, generation=0) == "ok"
    assert dc.dense_compaction_derivation(conn, "scope-none") == "ok"
    # Absent block table → None (status cannot be evaluated honestly).
    conn.execute("DROP TABLE unit_vectors_block")
    assert dc.dense_compaction_derivation(conn, _SCOPE) is None


# ---------------------------------------------------------------------------
# job handler
# ---------------------------------------------------------------------------


def test_handle_dense_compact_job(conn, store):
    rng = random.Random(53)
    _micro_blocks(conn, {f"u{i:03d}": _rand_vec(rng) for i in range(40)})
    ingester = _types.SimpleNamespace(store=store)
    job = {
        "job_id": "j-1",
        "kind": dc.DENSE_COMPACT_KIND,
        "scope_id": _SCOPE,
        "input_refs": {"scope_id": _SCOPE},
    }
    dc.handle_dense_compact(job, "owner", ingester)
    assert len(list(mx.iter_blocks(conn, _ENC, _SCOPE, 1))) == 1
    assert dc.compaction_epoch(conn, _ENC, _SCOPE, 1) == 1
    # Receipt event landed (real events table + event clock on the shim).
    row = conn.execute(
        "SELECT kind, payload_json FROM events WHERE kind = 'dense_compacted'"
    ).fetchone()
    assert row is not None
    # Re-run is a no-op — the handler stays silent.
    dc.handle_dense_compact(job, "owner", ingester)
    assert dc.compaction_epoch(conn, _ENC, _SCOPE, 1) == 1
    row2 = conn.execute(
        "SELECT COUNT(*) FROM events WHERE kind = 'dense_compacted'"
    ).fetchone()
    assert row2[0] == 1


def test_handle_dense_compact_foreign_scope_untouched(conn, store):
    rng = random.Random(59)
    _micro_blocks(conn, {f"a{i}": _rand_vec(rng) for i in range(12)})
    _micro_blocks(conn, {f"b{i}": _rand_vec(rng) for i in range(12)},
                  scope="scope-b")
    ingester = _types.SimpleNamespace(store=store)
    dc.handle_dense_compact(
        {"job_id": "j-2", "scope_id": _SCOPE, "input_refs": {}},
        "owner", ingester,
    )
    assert len(list(mx.iter_blocks(conn, _ENC, _SCOPE, 1))) == 1
    assert len(list(mx.iter_blocks(conn, _ENC, "scope-b", 1))) == 12
    assert dc.compaction_epoch(conn, _ENC, "scope-b", 1) == 0
