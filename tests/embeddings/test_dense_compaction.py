"""V8 dense-space verification (SPEC_V8 §08, §24 Dense row, §25 K42–K48).

The §24 verification row this module owns:

  "atomic compaction; erased key excluded; scan results equal before and
  after; int8 threshold on post-compaction count"

plus the §25 dense scenarios:

- K42: ``PoolProfile.prf_round`` is gone (V8-07.07) — declaration,
  POOLS values, and the static flag-reader sweep over every production
  source (a declared-but-unexecuted flag is the D8-25 defect).
- K43: micro-block fragmentation rewrites into contiguous
  ``ceil(rows/MAX_BLOCK_ROWS)`` blocks inside ONE transaction — a
  mid-commit failure rolls the space back byte-identically — and
  ``matrix.scan`` output is identical before and after. The post-
  compaction row count selects the quant mode (``> INT8_ROW_THRESHOLD``
  → int8, strict boundary).
- K44: a unit erased between ``plan_space`` and the write transaction
  is absent from the compacted blocks — commit re-reads live keys under
  ``BEGIN IMMEDIATE`` (V8-19.03 closure invariant).
- K45: scan stats honestly report the per-block execution path
  (``numpy`` engaged → vectorized counters; absent → honest-zero) and
  the lane records the scan envelope (``scan_ms``) + ``coverage_dense``.
- K46: candidates clearing the null-model floor are emitted marked
  ``signals["dense_slot"]``; below-floor items never appear.
- K47: no resolvable encoder/artifact → ``LaneStatus.UNAVAILABLE``,
  never ``ok`` — the ``local_memory_quality`` S1 gate shape; the
  provisioned hashing encoder (``local_memory``) stays ``ok``.
- K48: ``encoder_id`` header versions mint separate spaces; scans never
  mix them and compaction of one leaves the other untouched.

Sibling coverage lives in ``tests/retrieval/v7/test_dense_compact.py``
(lane-level emit/coverage detail); this file is self-contained on real
SQLite + real ``matrix``/``dense_compact`` machinery — no mocks.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import random
import sqlite3
import threading
import types as _types

import pytest

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

_SCOPE = "scope-a"
_ENC = "test:enc:v1"
_DIMS = 32
_NUMPY_PRESENT = importlib.util.find_spec("numpy") is not None


class _TxStore:
    """``read()``/``tx()`` shim over one explicit-transaction conn.

    ``isolation_level=None`` gives manual BEGIN/COMMIT — the same
    BEGIN IMMEDIATE semantics ``Store._write_tx`` enforces — so the
    compaction commit runs under a real SQLite write transaction and a
    raised error exercises a real ROLLBACK.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()

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


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(DDL_V1)
    ensure_v7_additive(c)
    mx.ensure_oracle_table(c)
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
    """One block per unit — the fragmentation shape compaction owns."""
    for bno, key in enumerate(sorted(vecs)):
        mx.write_block(conn, encoder_id, scope, gen, bno, [(key, vecs[key])])


def _space_dump(conn, encoder_id=_ENC, scope=_SCOPE, gen=1):
    """Full row tuples of one space — byte-exact pre/post comparison."""
    return conn.execute(
        f"SELECT encoder_id, scope_id, generation, block_no, n_rows, dims,"
        f" quant, scale_blob, data_blob, rowmap_blob FROM {mx.BLOCK_TABLE}"
        " WHERE encoder_id=? AND scope_id=? AND generation=?"
        " ORDER BY block_no",
        (encoder_id, scope, gen),
    ).fetchall()


# ---------------------------------------------------------------------------
# K42 — prf_round removal (V8-07.07, D8-25)
# ---------------------------------------------------------------------------


def test_k42_prf_round_static_flag_reader_check():
    """``prf_round`` is gone from ``PoolProfile``/``POOLS`` and no
    production module names the dead flag — the static flag-reader
    check the verification row requires."""
    import dataclasses

    from verbatim.core import types_v7
    fields = {f.name for f in dataclasses.fields(types_v7.PoolProfile)}
    assert "prf_round" not in fields
    for pool in types_v7.POOLS.values():
        assert not hasattr(pool, "prf_round")
    # Static flag-reader sweep: the flag must not appear in any
    # production source (readers included).
    from pathlib import Path
    root = Path(types_v7.__file__).resolve().parents[1]
    offenders = [
        str(p)
        for p in root.rglob("*.py")
        if "prf_round" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# K43 — atomic compaction; scan identical; int8 post-compaction threshold
# ---------------------------------------------------------------------------


def test_k43_micro_blocks_compact_scan_identical(conn, store):
    rng = random.Random(7)
    vecs = {f"u{i:04d}": _rand_vec(rng) for i in range(120)}
    _micro_blocks(conn, vecs)
    assert dc.plan_space(conn, _ENC, _SCOPE, 1).needed
    q = _rand_vec(random.Random(99))
    before = list(mx.scan(conn, _ENC, _SCOPE, 1, q, k=25, use_numpy=False))
    before_keys = mx.block_row_keys(conn, _ENC, _SCOPE, 1)

    rep = dc.compact_space(store, _ENC, _SCOPE, 1)
    assert rep["compacted"] and rep["reason"] == "ok"
    assert rep["blocks_before"] == 120 and rep["blocks_after"] == 1
    assert rep["rows"] == 120 and rep["quant"] == "f32"
    assert rep["epoch"] == 1

    blocks = list(mx.iter_blocks(conn, _ENC, _SCOPE, 1))
    assert len(blocks) == 1 and blocks[0].n_rows == 120
    assert mx.block_row_keys(conn, _ENC, _SCOPE, 1) == before_keys
    # Scan results identical before and after — same (key, score) pairs
    # exactly (f32 blocks preserve the stored bytes).
    after = list(mx.scan(conn, _ENC, _SCOPE, 1, q, k=25, use_numpy=False))
    assert after == before


def test_k43_compaction_commit_is_atomic(conn, store, monkeypatch):
    """A failure mid-rewrite rolls the whole commit back — the pre-image
    is byte-identical, no orphan blocks at the fresh range, no epoch."""
    rng = random.Random(13)
    vecs = {f"u{i:04d}": _rand_vec(rng) for i in range(90)}
    _micro_blocks(conn, vecs)
    pre_image = _space_dump(conn)

    monkeypatch.setattr(mx, "MAX_BLOCK_ROWS", 40)  # → 3 output blocks
    real_write = mx.write_block
    calls = {"n": 0}

    def boom(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:  # fail after two new blocks already landed
            raise RuntimeError("injected commit failure")
        return real_write(*a, **kw)

    monkeypatch.setattr(mx, "write_block", boom)
    with pytest.raises(RuntimeError, match="injected commit failure"):
        dc.compact_space(store, _ENC, _SCOPE, 1)
    assert calls["n"] == 3

    # Rollback is total: byte-identical pre-image, epoch untouched.
    assert _space_dump(conn) == pre_image
    assert dc.compaction_epoch(conn, _ENC, _SCOPE, 1) == 0
    # And a retry without the fault still compacts cleanly.
    monkeypatch.undo()
    rep = dc.compact_space(store, _ENC, _SCOPE, 1)
    assert rep["compacted"] and rep["blocks_after"] == 1
    assert rep["epoch"] == 1


def test_k43_int8_threshold_on_post_compaction_count(
    conn, store, monkeypatch
):
    """The V7-07.06 quant decision reads the POST-compaction row count
    (``> INT8_ROW_THRESHOLD`` → int8, strict)."""
    monkeypatch.setattr(mx, "INT8_ROW_THRESHOLD", 40)
    rng = random.Random(19)

    # 50 rows → post-compaction count crosses the threshold → int8.
    over = {f"o{i:03d}": _rand_vec(rng) for i in range(50)}
    _micro_blocks(conn, over, encoder_id="enc:over")
    rep = dc.compact_space(store, "enc:over", _SCOPE, 1)
    assert rep["compacted"] and rep["quant"] == "int8"
    blocks = list(mx.iter_blocks(conn, "enc:over", _SCOPE, 1))
    assert len(blocks) == 1 and blocks[0].quant == "int8"
    assert mx.block_row_keys(conn, "enc:over", _SCOPE, 1) == set(over)

    # int8 rows are byte-stable through compaction AND rescore through
    # the real f32 oracle (write_oracle_row → scan rescore phase).
    for key in over:
        mx.write_oracle_row(conn, "enc:over", key, over[key])
    q = _rand_vec(random.Random(5))
    res = mx.scan(conn, "enc:over", _SCOPE, 1, q, k=10, use_numpy=False)
    assert res.stats["quant"] == ["int8"]
    assert res.stats["rescore"] == "exact"
    assert set(res.kinds.values()) == {"int8_rescored"}
    res_nr = mx.scan(
        conn, "enc:over", _SCOPE, 1, q, k=10,
        use_numpy=False, rescore=False,
    )
    assert set(res_nr.kinds.values()) == {"int8_approx"}

    # Exactly AT the threshold → stays f32 (strict ``>`` boundary).
    at = {f"a{i:03d}": _rand_vec(rng) for i in range(40)}
    _micro_blocks(conn, at, encoder_id="enc:at")
    rep_at = dc.compact_space(store, "enc:at", _SCOPE, 1)
    assert rep_at["compacted"] and rep_at["quant"] == "f32"
    assert [b.quant for b in mx.iter_blocks(conn, "enc:at", _SCOPE, 1)] == [
        "f32"
    ]


# ---------------------------------------------------------------------------
# K44 — erased between plan and commit
# ---------------------------------------------------------------------------


def test_k44_erased_between_plan_and_commit(conn, store):
    """Plan on the snapshot; purge sweeps land before the commit tx;
    the rewrite re-reads live keys under the write lock — swept keys
    are absent from the compacted blocks."""
    rng = random.Random(17)
    vecs = {f"u{i:03d}": _rand_vec(rng) for i in range(40)}
    _micro_blocks(conn, vecs)
    # One shared block holding 3 more units (the purge path's UPDATE case).
    shared = {f"s{i}": _rand_vec(rng) for i in range(3)}
    mx.write_block(
        conn, _ENC, _SCOPE, 1, len(vecs),
        [(k, shared[k]) for k in sorted(shared)],
    )
    plan = dc.plan_space(conn, _ENC, _SCOPE, 1)
    assert plan.needed and plan.n_rows == 43

    # Erase 1: whole micro-block row DELETE (purge's empty-keep case).
    victim_block = sorted(vecs).index("u010")
    conn.execute(
        f"DELETE FROM {mx.BLOCK_TABLE} WHERE encoder_id=? AND scope_id=?"
        " AND generation=? AND block_no=?",
        (_ENC, _SCOPE, 1, victim_block),
    )
    # Erase 2: surgical rowmap rewrite inside the shared block — s1 out.
    blk = mx.read_block(conn, _ENC, _SCOPE, 1, len(vecs))
    keep = [i for i, k in enumerate(blk.keys) if k != "s1"]
    row_bytes = blk.dims * 4
    data_b = blk.data.tobytes()
    new_data = b"".join(
        data_b[i * row_bytes:(i + 1) * row_bytes] for i in keep
    )
    scales = conn.execute(
        f"SELECT scale_blob FROM {mx.BLOCK_TABLE} WHERE encoder_id=?"
        " AND scope_id=? AND generation=? AND block_no=?",
        (_ENC, _SCOPE, 1, len(vecs)),
    ).fetchone()[0]
    new_scale = b"".join(bytes(scales)[i * 8:(i + 1) * 8] for i in keep)
    import struct as _st
    new_rowmap = _st.pack("<I", len(keep)) + b"".join(
        _st.pack("<I", len(blk.keys[i].encode())) + blk.keys[i].encode()
        for i in keep
    )
    conn.execute(
        f"UPDATE {mx.BLOCK_TABLE} SET n_rows=?, scale_blob=?,"
        " data_blob=?, rowmap_blob=? WHERE encoder_id=? AND scope_id=?"
        " AND generation=? AND block_no=?",
        (len(keep), new_scale, new_data, new_rowmap,
         _ENC, _SCOPE, 1, len(vecs)),
    )

    # Commit re-plans and re-reads live keys under the write lock: the
    # stale plan counted 43 rows, the in-tx re-plan sees the post-sweep
    # 41 — both swept keys are absent from the compacted blocks.
    assert plan.n_rows == 43
    with store.tx() as c:
        rep = dc.compact_space_in_tx(c, _ENC, _SCOPE, 1)
    assert rep["compacted"] and rep["reason"] == "ok"
    assert rep["rows_before"] == 41 and rep["rows"] == 41
    keys = mx.block_row_keys(conn, _ENC, _SCOPE, 1)
    assert "u010" not in keys and "s1" not in keys
    assert keys == (set(vecs) | set(shared)) - {"u010", "s1"}


def test_k44_compact_space_public_path(conn, store):
    """``compact_space`` (read-plan → write-commit) excludes a unit the
    purge swept between its own read and tx phases — same invariant on
    the public entry point."""
    rng = random.Random(29)
    vecs = {f"u{i:03d}": _rand_vec(rng) for i in range(30)}
    _micro_blocks(conn, vecs)
    # Purge lands before compact_space is invoked at all — the in-tx
    # re-plan + re-read sees only live keys.
    dead = sorted(vecs).index("u007")
    conn.execute(
        f"DELETE FROM {mx.BLOCK_TABLE} WHERE encoder_id=? AND scope_id=?"
        " AND generation=? AND block_no=?",
        (_ENC, _SCOPE, 1, dead),
    )
    rep = dc.compact_space(store, _ENC, _SCOPE, 1)
    assert rep["compacted"] and rep["rows"] == 29
    assert "u007" not in mx.block_row_keys(conn, _ENC, _SCOPE, 1)


# ---------------------------------------------------------------------------
# K45 — vectorized path stats + envelope
# ---------------------------------------------------------------------------


def test_k45_scan_stats_envelope(conn, store):
    """``matrix.scan`` honestly reports the per-block path: with numpy
    the matmul counters engage; without it the counters read zero/
    honest-false — never fabricated."""
    rng = random.Random(41)
    vecs = {f"u{i:04d}": _rand_vec(rng) for i in range(80)}
    _micro_blocks(conn, vecs)
    dc.compact_space(store, _ENC, _SCOPE, 1)

    q = _rand_vec(random.Random(11))
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, k=25)
    assert res.stats["blocks_total"] == 1
    assert res.stats["blocks_scanned"] == 1
    assert res.examined == 80 and res.eligible == 80
    assert res.scored == 80 and len(res) == 25
    assert res.stats["quant"] == ["f32"]
    assert res.stats["numpy"] is _NUMPY_PRESENT

    # A caller forcing the vectorized path gets an honest degradation
    # note when numpy is absent — never a silent pretend-vectorized run.
    forced = mx.scan(conn, _ENC, _SCOPE, 1, q, k=25, use_numpy=True)
    if _NUMPY_PRESENT:
        assert forced.stats["numpy"] is True
    else:
        assert forced.stats["numpy"] is False
        assert forced.stats["numpy_note"] == "requested_but_absent"
    # Both paths score the same ranking on the compacted space.
    assert [k for k, _ in forced] == [k for k, _ in res]


def test_k45_lane_stats_envelope(conn, store):
    """Lane-level: ``vectorized_blocks`` (honest-zero without numpy),
    ``scan_ms`` envelope recorded, ``coverage_dense`` block emitted."""
    enc = HashingEncoder(_types.SimpleNamespace(artifact_revision=None))
    texts = {
        "u-target": "alice moved to berlin in march 2024 with her dog",
        "u-close": "alice moved to berlin in 2024",
        **{f"u-{i:02d}": f"unrelated note {i} about topic {i}"
           for i in range(20)},
    }
    for key in texts:
        conn.execute(
            "INSERT INTO units (unit_id, source_id, revision, scope_id,"
            " kind, speaker_canon, generation) VALUES (?,?,?,?,?,?,?)",
            (key, f"src-{key}", 1, _SCOPE, "turn", "alice", 1),
        )
    blobs = enc.encode(list(texts.values()))
    for bno, (key, blob) in enumerate(zip(texts, blobs)):
        mx.write_block(conn, enc.encoder_id, _SCOPE, 1, bno, [(key, blob)])
    dc.compact_space(store, enc.encoder_id, _SCOPE, 1)

    ctx = LaneContextV7(
        store=_TxStore(conn),
        scope_id=_SCOPE,
        generation=1,
        eligible=None,
        query_time_us=1_000_000,
        profile="local_memory",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="local_memory",
            lanes=(), lane_weights={},
        ),
        manifest={"query_encoder": enc, "encoder_id": enc.encoder_id},
    )
    qv = QueryViewV7(
        query="alice moved to berlin in march 2024 with her dog",
        norm=NormAnalysis(analyzer_id="norm/v2", terms=(), identifiers=()),
        intent=IntentResult(primary=IntentClass.LOOKUP,
                            classes=(IntentClass.LOOKUP,)),
    )
    out = lane_dense(ctx, qv, LaneSlice(deadline_ms=10_000.0, cap=10))
    assert out.status == LaneStatus.OK
    assert out.stats["blocks_scanned"] == 1
    # Honest path accounting: vectorized only when numpy actually ran.
    assert out.stats["numpy"] is _NUMPY_PRESENT
    assert out.stats["vectorized_blocks"] == (
        out.stats["blocks_scanned"] if _NUMPY_PRESENT else 0
    )
    assert out.stats["scan_ms"] >= 0.0  # dense envelope recorded
    assert out.stats["compaction_epoch"] == 1
    cov = out.stats["coverage_dense"]
    assert cov["blocks_scanned"] == 1
    assert cov["compaction_epoch"] == 1
    assert cov["floor"] == pytest.approx(out.stats["noise_floor"])


# ---------------------------------------------------------------------------
# K46 — dense_slot emit-side marking
# ---------------------------------------------------------------------------


def _mk_ctx(conn, enc, manifest_extra=None):
    manifest = {"query_encoder": enc, "encoder_id": enc.encoder_id}
    if manifest_extra:
        manifest.update(manifest_extra)
    return LaneContextV7(
        store=_TxStore(conn),
        scope_id=_SCOPE,
        generation=1,
        eligible=None,
        query_time_us=1_000_000,
        profile="local_memory",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="local_memory",
            lanes=(), lane_weights={},
        ),
        manifest=manifest,
    )


def _mk_qv(text):
    return QueryViewV7(
        query=text,
        norm=NormAnalysis(analyzer_id="norm/v2", terms=(), identifiers=()),
        intent=IntentResult(primary=IntentClass.LOOKUP,
                            classes=(IntentClass.LOOKUP,)),
    )


def test_k46_dense_slot_floor_marking(conn):
    enc = HashingEncoder(_types.SimpleNamespace(artifact_revision=None))
    texts = {
        "u-target": "alice moved to berlin in march 2024 with her dog",
        "u-close": "alice moved to berlin in 2024",
        "u-held": "alice moved to berlin in march 2024 with her dog",
        **{f"u-{i:02d}": f"unrelated note {i} about topic {i}"
           for i in range(28)},
    }
    for key in texts:
        conn.execute(
            "INSERT INTO units (unit_id, source_id, revision, scope_id,"
            " kind, speaker_canon, generation) VALUES (?,?,?,?,?,?,?)",
            (key, f"src-{key}", 1, _SCOPE, "turn",
             "held" if key == "u-held" else "alice", 1),
        )
    blobs = enc.encode(list(texts.values()))
    for bno, (key, blob) in enumerate(zip(texts, blobs)):
        mx.write_block(conn, enc.encoder_id, _SCOPE, 1, bno, [(key, blob)])

    out = lane_dense(
        _mk_ctx(conn, enc, manifest_extra={"dense_N_d": 10}),
        _mk_qv("alice moved to berlin in march 2024 with her dog"),
        LaneSlice(deadline_ms=10_000.0, cap=10),
    )
    assert out.status == LaneStatus.OK
    # Floor-clearing candidates enter the pool marked dense_slot;
    # below-floor rows are never emitted.
    assert len(out.candidates) == 3
    assert all(c.signals.get("dense_slot") is True for c in out.candidates)
    assert out.stats["dense_slots"] == 3
    # below_floor counts within the scan's returned top-cap (cap=10).
    assert out.stats["below_floor"] == 10 - 3
    floor = out.stats["noise_floor"]
    assert all(c.raw_score > floor for c in out.candidates)

    # A query with no lexical overlap anywhere still emits only
    # floor-clearers — and may honestly emit none.
    out_none = lane_dense(
        _mk_ctx(conn, enc),
        _mk_qv("quantum chromodynamics lattice gauge coupling"),
        LaneSlice(deadline_ms=10_000.0, cap=10),
    )
    assert out_none.candidates == []
    assert out_none.stats["coverage_dense"]["slots"] == 0


# ---------------------------------------------------------------------------
# K47 — artifact-gated availability
# ---------------------------------------------------------------------------


def test_k47_no_artifact_dense_unavailable(conn):
    """No resolvable encoder → ``UNAVAILABLE``/``no_encoder``, never
    ``ok`` — the shape ``local_memory_quality`` takes without its S1
    artifact; the provisioned hashing encoder (``local_memory``) is
    unaffected."""
    conn.execute(
        "INSERT INTO units (unit_id, source_id, revision, scope_id,"
        " kind, speaker_canon, generation) VALUES (?,?,?,?,?,?,?)",
        ("u-a", "s", 1, _SCOPE, "turn", "alice", 1),
    )
    ctx = _mk_ctx(conn, enc=None) if False else LaneContextV7(
        store=_TxStore(conn),
        scope_id=_SCOPE,
        generation=1,
        eligible=None,
        query_time_us=1_000_000,
        profile="local_memory_quality",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7",
            profile="local_memory_quality", lanes=(), lane_weights={},
        ),
        manifest={},  # no query_encoder, no encoder_id pin — no artifact
    )
    out = lane_dense(ctx, _mk_qv("anything"),
                     LaneSlice(deadline_ms=10_000.0, cap=10))
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_encoder"

    # local_memory with the provisioned hashing encoder resolves and
    # runs — the S1 tier's absence never bleeds into the base profile.
    enc = HashingEncoder(_types.SimpleNamespace(artifact_revision=None))
    mx.write_block(conn, enc.encoder_id, _SCOPE, 1, 0,
                   [("u-a", enc.encode(["alice moved to berlin"])[0])])
    out2 = lane_dense(
        _mk_ctx(conn, enc), _mk_qv("alice moved to berlin"),
        LaneSlice(deadline_ms=10_000.0, cap=10),
    )
    assert out2.status == LaneStatus.OK


def test_k47_ambiguous_space_unavailable(conn):
    """Two encoder spaces in one scope with no pin → ``UNAVAILABLE``
    ``encoder_id_ambiguous`` — never a silently mixed scan."""
    enc = HashingEncoder(_types.SimpleNamespace(artifact_revision=None))
    conn.execute(
        "INSERT INTO units (unit_id, source_id, revision, scope_id,"
        " kind, speaker_canon, generation) VALUES (?,?,?,?,?,?,?)",
        ("u-a", "s", 1, _SCOPE, "turn", "alice", 1),
    )
    mx.write_block(conn, "enc:v1", _SCOPE, 1, 0,
                   [("u-a", _rand_vec(random.Random(1), enc.dimensions))])
    mx.write_block(conn, "enc:v2", _SCOPE, 1, 0,
                   [("u-b", _rand_vec(random.Random(2), enc.dimensions))])
    ctx = LaneContextV7(
        store=_TxStore(conn),
        scope_id=_SCOPE,
        generation=1,
        eligible=None,
        query_time_us=1_000_000,
        profile="local_memory",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="local_memory",
            lanes=(), lane_weights={},
        ),
        # Callable adapter over the real encoder — carries no
        # ``encoder_id`` attribute, so the space pin stays unresolved
        # and the distinct-ids probe must decide honestly.
        manifest={"query_encoder": lambda t: enc.encode([t])[0]},
    )
    out = lane_dense(ctx, _mk_qv("x"), LaneSlice(deadline_ms=10_000.0,
                                                 cap=10))
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "encoder_id_ambiguous"
    assert out.stats["encoder_ids"] == ["enc:v1", "enc:v2"]


# ---------------------------------------------------------------------------
# K48 — header-version spaces never mix
# ---------------------------------------------------------------------------


def test_k48_header_version_spaces_isolated(conn, store):
    rng = random.Random(31)
    enc_v1 = "hashing:subword-ngram:v1:hdr:v1"
    enc_v2 = "hashing:subword-ngram:v1:hdr:v2"
    a = {f"a{i:03d}": _rand_vec(rng) for i in range(30)}
    b = {f"b{i:03d}": _rand_vec(rng) for i in range(30)}
    _micro_blocks(conn, a, encoder_id=enc_v1)
    _micro_blocks(conn, b, encoder_id=enc_v2)

    rep = dc.compact_space(store, enc_v1, _SCOPE, 1)
    assert rep["compacted"]
    # v2 space untouched — count, keys, and epoch intact.
    assert len(list(mx.iter_blocks(conn, enc_v2, _SCOPE, 1))) == 30
    assert mx.block_row_keys(conn, enc_v2, _SCOPE, 1) == set(b)
    assert dc.compaction_epoch(conn, enc_v2, _SCOPE, 1) == 0

    # Scans are keyed per space — each returns only its own keys.
    q = _rand_vec(random.Random(3))
    ra = mx.scan(conn, enc_v1, _SCOPE, 1, q, k=100, use_numpy=False)
    rb = mx.scan(conn, enc_v2, _SCOPE, 1, q, k=100, use_numpy=False)
    assert all(k.startswith("a") for k, _ in ra)
    assert all(k.startswith("b") for k, _ in rb)
    assert len(ra) == 30 and len(rb) == 30


def test_k48_lane_bare_pin_resolves_header_space(conn):
    """``unit_vector_encoder_id`` maps a bare model id to ``:hdr:v1`` —
    the lane resolves the headered space and stays on it when a v2
    space also exists (no cross-version scan)."""
    enc = HashingEncoder(_types.SimpleNamespace(artifact_revision=None))
    base = enc.encoder_id  # bare id — no :hdr: segment
    hdr = mx.unit_vector_encoder_id(base)
    assert hdr.endswith(":hdr:v1") and hdr != base
    assert mx.unit_vector_encoder_id(hdr) == hdr  # idempotent

    conn.execute(
        "INSERT INTO units (unit_id, source_id, revision, scope_id,"
        " kind, speaker_canon, generation) VALUES (?,?,?,?,?,?,?)",
        ("u-a", "s", 1, _SCOPE, "turn", "alice", 1),
    )
    conn.execute(
        "INSERT INTO units (unit_id, source_id, revision, scope_id,"
        " kind, speaker_canon, generation) VALUES (?,?,?,?,?,?,?)",
        ("u-b", "s", 1, _SCOPE, "turn", "alice", 1),
    )
    blob_a = enc.encode(["alice moved to berlin in march 2024"])[0]
    mx.write_block(conn, hdr, _SCOPE, 1, 0, [("u-a", blob_a)])
    # A foreign header version lives in the same scope — must not mix.
    mx.write_block(
        conn, f"{base}:hdr:v2", _SCOPE, 1, 0,
        [("u-b", enc.encode(["totally different words here now"])[0])],
    )
    out = lane_dense(
        _mk_ctx(conn, enc),  # manifest pins the BARE encoder_id
        _mk_qv("alice moved to berlin in march 2024"),
        LaneSlice(deadline_ms=10_000.0, cap=10),
    )
    assert out.status == LaneStatus.OK
    assert out.stats["encoder_id"] == hdr  # resolved to :hdr:v1
    assert {c.unit_id for c in out.candidates} == {"u-a"}
