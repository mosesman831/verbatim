"""w-dense wave-A tests: contiguous-matrix store + dense lane skeleton.

Covers V7-07.04/05/06/14 and D7-17: packed-block round-trip, blocked scan
== per-row-BLOB oracle ordering (pure-Python always, numpy when it
imports), int8 quantization with measured recall loss, eligibility-before-
rank, honest unavailability, deadline partials, and deterministic
tie-breaking.

The ``units`` / ``unit_vectors_block`` schema comes from the landed
``schema_v7.ensure_v7_additive`` (the real §30 DDL — notably ``units`` is
keyed ``(unit_id, generation)`` so rebuild generations coexist). The
``unit_vectors`` per-row oracle table is provisional — the durable per-row
BLOB home is a schema_v7 integration decision (V7-07.04), marked for swap.
"""

from __future__ import annotations

import contextlib
import math
import random
import sqlite3
import types as _types

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
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
from verbatim.retrieval.v7 import dense as dense_mod
from verbatim.retrieval.v7.dense import lane_dense
from verbatim.storage.schema_v7 import ensure_v7_additive

# Provisional per-row f32 oracle (V7-07.04 durable oracle). Not in §30 —
# the schema_v7 integration owns the final name; marked for swap.
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


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    ensure_v7_additive(c)
    c.executescript(_ORACLE_DDL)
    yield c
    c.close()


def _rand_vec(rng: random.Random, dims: int = _DIMS) -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(dims)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _write_corpus(conn, vecs: dict[str, list[float]], *, block_size=256,
                  quant="f32", encoder_id=_ENC, scope=_SCOPE, gen=1,
                  block_start=0):
    keys = sorted(vecs)
    n = 0
    for off, i in enumerate(range(0, len(keys), block_size)):
        rows = [(k, vecs[k]) for k in keys[i : i + block_size]]
        n += mx.write_block(
            conn, encoder_id, scope, gen, block_start + off, rows,
            quant=quant,
        )
    return n


def _oracle_topk(vecs, query, k, eligible=None):
    """Brute-force per-row-BLOB cosine oracle — the pre-D7-17 semantics:
    decode each row's own float32 blob, cosine in float64, sort
    ``(-score, key)``. Independent of the block machinery under test."""
    qn = math.sqrt(sum(x * x for x in query))
    out = []
    for key, v in vecs.items():
        if eligible is not None and key not in eligible:
            continue
        blob = Float32Codec.pack(v)
        w = Float32Codec.unpack(blob, len(blob) // 4)
        vn = math.sqrt(sum(x * x for x in w))
        if vn <= 0.0:
            continue
        s = sum(a * b for a, b in zip(query, w)) / (qn * vn)
        out.append((key, s))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out[:k]


# ---------------------------------------------------------------------------
# block write/read round-trip
# ---------------------------------------------------------------------------


def test_block_roundtrip_f32(conn):
    rng = random.Random(7)
    vecs = {f"u{i:03d}": _rand_vec(rng) for i in range(10)}
    keys = sorted(vecs)
    written = mx.write_block(
        conn, _ENC, _SCOPE, 1, 0, [(k, vecs[k]) for k in keys]
    )
    assert written == 10

    blk = mx.read_block(conn, _ENC, _SCOPE, 1, 0)
    assert blk is not None
    assert blk.n_rows == 10 and blk.dims == _DIMS and blk.quant == "f32"
    assert blk.keys == tuple(keys)
    assert blk.scales is None
    for i, k in enumerate(keys):
        # f32 round-trip is bit-exact at the packed-byte level.
        assert Float32Codec.pack(blk.row(i)) == Float32Codec.pack(vecs[k])
        assert blk.norms[i] == pytest.approx(1.0, abs=1e-6)
    assert mx.read_block(conn, _ENC, _SCOPE, 1, 99) is None

    # iter_blocks streams in block_no order across a multi-block corpus.
    mx.write_block(conn, _ENC, _SCOPE, 1, 2, [("z1", vecs["u000"])])
    mx.write_block(conn, _ENC, _SCOPE, 1, 1, [("z0", vecs["u001"])])
    ordered = [b.block_no for b in mx.iter_blocks(conn, _ENC, _SCOPE, 1)]
    assert ordered == [0, 1, 2]
    all_keys = [k for b in mx.iter_blocks(conn, _ENC, _SCOPE, 1) for k in b.keys]
    assert all_keys == keys + ["z0", "z1"]


def test_block_roundtrip_int8(conn):
    rng = random.Random(11)
    vecs = {f"u{i:03d}": _rand_vec(rng) for i in range(8)}
    keys = sorted(vecs)
    mx.write_block(
        conn, _ENC, _SCOPE, 1, 0, [(k, vecs[k]) for k in keys], quant="int8"
    )
    blk = mx.read_block(conn, _ENC, _SCOPE, 1, 0)
    assert blk.quant == "int8" and blk.scales is not None
    assert len(blk.scales) == len(blk.norms) == 8
    for i, k in enumerate(keys):
        s = blk.scales[i]
        assert s == pytest.approx(max(abs(x) for x in vecs[k]) / 127.0, rel=1e-6)
        # dequantized row approximates the original within quant error.
        for got, want in zip(blk.row(i), vecs[k]):
            assert got == pytest.approx(want, abs=s + 1e-6)


def test_write_block_validates(conn):
    v = _rand_vec(random.Random(3))
    with pytest.raises(VerbatimError) as ei:
        mx.write_block(conn, _ENC, _SCOPE, 1, 0, [("a", v), ("a", v)])
    assert ei.value.code == ErrorCode.VALIDATION
    with pytest.raises(VerbatimError) as ei:
        mx.write_block(conn, _ENC, _SCOPE, 1, 0, [("a", v), ("b", v[:8])])
    assert ei.value.code == ErrorCode.VECTOR_INVALID
    with pytest.raises(VerbatimError) as ei:
        mx.write_block(conn, _ENC, _SCOPE, 1, 0, [("a", v)], quant="bit")
    assert ei.value.code == ErrorCode.CAPABILITY_UNAVAILABLE
    with pytest.raises(VerbatimError):
        mx.write_block(conn, _ENC, _SCOPE, 1, 0, [("a", [0.0, float("nan")] + v[2:])])
    assert mx.write_block(conn, _ENC, _SCOPE, 1, 5, []) == 0
    assert mx.read_block(conn, _ENC, _SCOPE, 1, 5) is None


# ---------------------------------------------------------------------------
# blocked scan == per-row oracle
# ---------------------------------------------------------------------------


def _corpus(conn, n=600, dims=_DIMS, seed=42, block_size=256):
    rng = random.Random(seed)
    vecs = {f"u{i:04d}": _rand_vec(rng, dims) for i in range(n)}
    _write_corpus(conn, vecs, block_size=block_size)
    return vecs


def test_scan_pure_python_matches_oracle(conn):
    vecs = _corpus(conn)
    rng = random.Random(99)
    q = _rand_vec(rng)
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, k=40, use_numpy=False)
    oracle = _oracle_topk(vecs, q, 40)
    assert [k for k, _ in res] == [k for k, _ in oracle]
    for (_, s1), (_, s2) in zip(res, oracle):
        assert s1 == pytest.approx(s2, abs=1e-9)
    assert res.examined == 600 and res.eligible == 600 and res.scored == 600
    assert not res.partial
    assert all(res.kinds[k] == "f32" for k, _ in res)


def test_scan_numpy_matches_oracle_and_pure(conn):
    np = pytest.importorskip("numpy")
    vecs = _corpus(conn)
    rng = random.Random(99)
    q = _rand_vec(rng)
    res_np = mx.scan(conn, _ENC, _SCOPE, 1, q, k=40, use_numpy=True)
    res_py = mx.scan(conn, _ENC, _SCOPE, 1, q, k=40, use_numpy=False)
    oracle = _oracle_topk(vecs, q, 40)
    assert res_np.stats["numpy"] is True
    # V7-07.05: identical top-k ordering across both paths and the oracle.
    assert [k for k, _ in res_np] == [k for k, _ in oracle]
    assert [k for k, _ in res_np] == [k for k, _ in res_py]
    for (_, s1), (_, s2) in zip(res_np, oracle):
        assert s1 == pytest.approx(s2, abs=1e-9)


def test_scan_multi_encoder_isolation(conn):
    vecs = _corpus(conn, n=40)
    rng = random.Random(5)
    other = {f"w{i}": _rand_vec(rng) for i in range(20)}
    _write_corpus(conn, other, encoder_id="other:enc:v9", gen=1)
    _write_corpus(conn, other, scope="scope-b", gen=1)
    _write_corpus(conn, other, gen=2)  # same scope+encoder, other generation
    q = _rand_vec(rng)
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, k=100, use_numpy=False)
    assert res.examined == 40
    assert all(k in vecs for k, _ in res)


# ---------------------------------------------------------------------------
# int8 quantization — measured recall loss
# ---------------------------------------------------------------------------


def _cluster_corpus(n_clusters=5, per=200, dims=_DIMS, noise=0.05, seed=1234):
    rng = random.Random(seed)
    centers = [_rand_vec(rng, dims) for _ in range(n_clusters)]
    vecs: dict[str, list[float]] = {}
    for c in range(n_clusters):
        for i in range(per):
            v = [centers[c][d] + noise * rng.gauss(0, 1) for d in range(dims)]
            n = math.sqrt(sum(x * x for x in v))
            vecs[f"c{c}u{i:04d}"] = [x / n for x in v]
    return vecs, centers


def test_int8_recall_loss_bounded(conn):
    """V7-07.06: int8 + f32 rescore recall@50 vs exact oracle < 2% loss on
    separable clusters (loss > 1pp fails the release gate — we assert the
    tighter bound on this fixture)."""
    vecs, centers = _cluster_corpus()
    _write_corpus(conn, vecs, block_size=256, quant="int8")
    for key, v in vecs.items():
        mx.write_oracle_row(conn, _ENC, key, v)
    rng = random.Random(77)
    q = [centers[0][d] + 0.01 * rng.gauss(0, 1) for d in range(_DIMS)]
    qn = math.sqrt(sum(x * x for x in q))
    q = [x / qn for x in q]

    exact = {k for k, _ in _oracle_topk(vecs, q, 50)}
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, k=50, use_numpy=False)
    got = {k for k, _ in res}
    recall = len(got & exact) / 50.0
    res.stats["measured_recall_at_50"] = recall
    print(f"\n[int8] measured recall@50 vs exact oracle: {recall:.4f}")
    assert res.stats["rescore"] == "exact"
    assert all(res.kinds[k] == "int8_rescored" for k, _ in res)
    assert recall >= 0.98  # <2% loss on the separable-cluster fixture


def test_int8_recall_numpy_path(conn):
    pytest.importorskip("numpy")
    vecs, centers = _cluster_corpus()
    _write_corpus(conn, vecs, block_size=256, quant="int8")
    for key, v in vecs.items():
        mx.write_oracle_row(conn, _ENC, key, v)
    rng = random.Random(77)
    q = [centers[0][d] + 0.01 * rng.gauss(0, 1) for d in range(_DIMS)]
    qn = math.sqrt(sum(x * x for x in q))
    q = [x / qn for x in q]
    res_np = mx.scan(conn, _ENC, _SCOPE, 1, q, k=50, use_numpy=True)
    res_py = mx.scan(conn, _ENC, _SCOPE, 1, q, k=50, use_numpy=False)
    assert [k for k, _ in res_np] == [k for k, _ in res_py]
    exact = {k for k, _ in _oracle_topk(vecs, q, 50)}
    assert len({k for k, _ in res_np} & exact) / 50.0 >= 0.98


def test_int8_without_oracle_is_labeled_approx(conn):
    vecs, _centers = _cluster_corpus(per=40)
    _write_corpus(conn, vecs, block_size=256, quant="int8")
    rng = random.Random(1)
    q = _rand_vec(rng)
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, k=20, use_numpy=False)
    assert res.stats["rescore"] == "unavailable"
    assert all(res.kinds[k] == "int8_approx" for k, _ in res)


def test_mixed_quant_blocks(conn):
    rng = random.Random(31)
    exact_vecs = {f"e{i:03d}": _rand_vec(rng) for i in range(60)}
    quant_vecs = {f"q{i:03d}": _rand_vec(rng) for i in range(60)}
    _write_corpus(conn, exact_vecs, block_size=256, quant="f32")
    blocks = list(mx.iter_blocks(conn, _ENC, _SCOPE, 1))
    mx.write_block(
        conn, _ENC, _SCOPE, 1, len(blocks),
        [(k, v) for k, v in sorted(quant_vecs.items())], quant="int8",
    )
    for k, v in quant_vecs.items():
        mx.write_oracle_row(conn, _ENC, k, v)
    q = _rand_vec(rng)
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, k=30, use_numpy=False)
    both = {**exact_vecs, **quant_vecs}
    oracle = _oracle_topk(both, q, 30)
    # f32 hits are exact; int8 hits rescored — ordering vs oracle may lose
    # recall only through quantization (asserted loosely here, tightly in
    # the cluster fixture above).
    assert len({k for k, _ in res} & {k for k, _ in oracle}) >= 25
    assert set(res.stats["quant"]) == {"f32", "int8"}


# ---------------------------------------------------------------------------
# eligibility before rank
# ---------------------------------------------------------------------------


def test_scan_eligibility_set(conn):
    vecs = _corpus(conn, n=200)
    rng = random.Random(4)
    q = _rand_vec(rng)
    allowed = frozenset(sorted(vecs)[::3])  # every third key
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, eligible_keys=allowed, k=50,
                  use_numpy=False)
    assert res.examined == 200
    assert res.eligible == len(allowed)
    assert all(k in allowed for k, _ in res)
    oracle = _oracle_topk(vecs, q, 50, eligible=allowed)
    assert [k for k, _ in res] == [k for k, _ in oracle]


def test_scan_eligibility_callable(conn):
    vecs = _corpus(conn, n=120)
    rng = random.Random(4)
    q = _rand_vec(rng)
    bad = set(sorted(vecs)[:17])
    res = mx.scan(
        conn, _ENC, _SCOPE, 1, q,
        eligible_keys=lambda k: k not in bad, k=50, use_numpy=False,
    )
    assert all(k not in bad for k, _ in res)
    assert res.eligible == 120 - 17
    oracle = _oracle_topk(vecs, q, 50, eligible=set(vecs) - bad)
    assert [k for k, _ in res] == [k for k, _ in oracle]


def test_scan_eligibility_empty_set(conn):
    _corpus(conn, n=50)
    q = _rand_vec(random.Random(4))
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, eligible_keys=frozenset(), k=10)
    assert list(res) == [] and res.eligible == 0
    assert res.stats["stop_reason"] == "eligible_empty"


# ---------------------------------------------------------------------------
# deadline / partial
# ---------------------------------------------------------------------------


def test_scan_deadline_partial_fake_clock(conn):
    _corpus(conn, n=250, block_size=50)  # 5 blocks
    q = _rand_vec(random.Random(4))
    calls = iter([0.0, 0.0, 0.0, 1e9] + [1e9] * 64)  # t0 + per-block checks
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, k=10,
                  deadline_ms=1000.0, clock=lambda: next(calls),
                  use_numpy=False)
    assert res.partial is True
    assert res.stats["stop_reason"] == "deadline"
    assert res.examined == 100  # blocks 0-1 fully scanned; cut before block 2
    assert len(res) <= 10


def test_scan_deadline_zero(conn):
    _corpus(conn, n=50)
    q = _rand_vec(random.Random(4))
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, k=10, deadline_ms=0.0,
                  use_numpy=False)
    assert res.partial is True and res.examined == 0 and list(res) == []


# ---------------------------------------------------------------------------
# determinism + edge cases
# ---------------------------------------------------------------------------


def test_determinism_and_tie_break(conn):
    rng = random.Random(8)
    shared = _rand_vec(rng)
    vecs = {"z-dup": shared, "a-dup": list(shared), "m-dup": list(shared)}
    for i in range(20):
        vecs[f"r{i:02d}"] = _rand_vec(rng)
    _write_corpus(conn, vecs, block_size=8)
    q = list(shared)
    r1 = mx.scan(conn, _ENC, _SCOPE, 1, q, k=25, use_numpy=False)
    r2 = mx.scan(conn, _ENC, _SCOPE, 1, q, k=25, use_numpy=False)
    assert list(r1) == list(r2)
    top3 = [k for k, _ in r1[:3]]
    assert top3 == ["a-dup", "m-dup", "z-dup"]  # (-score, key) tie-break
    assert all(s == pytest.approx(1.0, abs=1e-6) for _, s in r1[:3])


def test_zero_norm_and_dims_mismatch(conn):
    rng = random.Random(9)
    vecs = {f"u{i:02d}": _rand_vec(rng) for i in range(30)}
    vecs["zero-vec"] = [0.0] * _DIMS
    _write_corpus(conn, vecs, block_size=64)
    other_dims = {f"d{i}": _rand_vec(rng, dims=16) for i in range(10)}
    # dims=16 block shares the (encoder, scope, generation) space — scanned
    # but never scored (different vector space).
    _write_corpus(conn, other_dims, block_size=64, block_start=1)
    q = _rand_vec(rng)
    res = mx.scan(conn, _ENC, _SCOPE, 1, q, k=40, use_numpy=False)
    assert "zero-vec" not in {k for k, _ in res}
    assert not any(k.startswith("d") for k, _ in res)
    assert res.stats["excluded_zero_norm"] == 1
    assert res.stats["blocks_dims_mismatch"] == 1
    # zero-vec passed eligibility but is unscored; the dims-16 rows are in
    # a different vector space and never counted eligible.
    assert res.eligible == 31 and res.scored == 30


def test_scan_input_validation(conn):
    _corpus(conn, n=20)
    with pytest.raises(VerbatimError) as ei:
        mx.scan(conn, _ENC, _SCOPE, 1, _rand_vec(random.Random(1)), k=0)
    assert ei.value.code == ErrorCode.VALIDATION
    with pytest.raises(VerbatimError) as ei:
        mx.scan(conn, _ENC, _SCOPE, 1, b"\x00\x01", k=5)
    assert ei.value.code == ErrorCode.VECTOR_INVALID
    res = mx.scan(conn, _ENC, _SCOPE, 1, [0.0] * _DIMS, k=5)
    assert list(res) == [] and res.stats["stop_reason"] == "zero_norm_query"


# ---------------------------------------------------------------------------
# lane: lane_dense over LaneContextV7
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal store stand-in — the lane only needs ``read()``."""

    def __init__(self, conn):
        self._conn = conn

    @contextlib.contextmanager
    def read(self):
        yield self._conn


def _ctx(conn, *, encoder=None, eligible=None, encoder_id="auto"):
    manifest = {}
    if encoder is not None:
        manifest["query_encoder"] = encoder
    if encoder_id != "auto":
        manifest["encoder_id"] = encoder_id
    elif encoder is not None:
        manifest["encoder_id"] = getattr(encoder, "encoder_id", None)
    return LaneContextV7(
        store=_FakeStore(conn),
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


def _slice(cap=40, deadline_ms=None):
    return LaneSlice(deadline_ms=deadline_ms if deadline_ms is not None else 10_000.0,
                     cap=cap)


def _hashing_encoder():
    return HashingEncoder(_types.SimpleNamespace(artifact_revision=None))


def _units(conn, rows):
    for r in rows:
        conn.execute(
            "INSERT INTO units (unit_id, source_id, revision, scope_id, kind,"
            " speaker_canon, generation) VALUES (?,?,?,?,?,?,?)", r
        )


def _text_corpus(conn, enc, *, block_size=None):
    """Real-encoder corpus: 30 units incl. one held + one target text."""
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
    pairs = list(zip(texts, blobs))
    bs = block_size or len(pairs)
    for bno, i in enumerate(range(0, len(pairs), bs)):
        mx.write_block(conn, enc.encoder_id, _SCOPE, 1, bno, pairs[i : i + bs])
    return texts


def test_lane_dense_end_to_end(conn):
    enc = _hashing_encoder()
    _text_corpus(conn, enc)
    # Query text identical to the u-target/u-held unit text -> cosine ~1.0;
    # the two identical vectors tie and break on ascending key.
    out = lane_dense(
        _ctx(conn, encoder=enc),
        _qv("alice moved to berlin in march 2024 with her dog"),
        _slice(cap=10),
    )
    assert out.status == LaneStatus.OK
    assert out.examined == 31 and out.eligible == 31
    # Null-model floor (3/sqrt(dim)): only the three real matches emit —
    # the 28 unrelated texts sit within noise of the null cosine and are
    # disclosed via stats, never delivered.
    assert len(out.candidates) == 3
    assert [c.unit_id for c in out.candidates] == [
        "u-held",
        "u-target",
        "u-close",
    ]
    assert out.candidates[0].raw_score == pytest.approx(1.0, abs=1e-6)
    tgt = out.candidates[1]
    assert tgt.source_id == "src-u-target" and tgt.revision == 1
    assert [c.rank for c in out.candidates] == list(range(1, 4))
    assert all(c.lane == "dense" for c in out.candidates)
    assert all("cosine" in c.signals for c in out.candidates)
    assert out.stats["encoder_id"] == enc.encoder_id
    assert out.stats["noise_floor"] > 0
    assert out.stats["below_floor"] == 7


def test_lane_dense_noise_floor_refuses_unrelated(conn):
    """A query sharing nothing with the indexed units emits no
    candidates — the lane abstains honestly rather than delivering the
    nearest noise (V7-11.01 trigger (b) stays reachable)."""
    enc = _hashing_encoder()
    _text_corpus(conn, enc)
    out = lane_dense(
        _ctx(conn, encoder=enc),
        _qv("quantum chromodynamics lattice gauge coupling"),
        _slice(cap=10),
    )
    assert out.status == LaneStatus.OK
    assert out.candidates == []
    assert out.stats["note"] == "all_below_noise_floor"
    assert out.stats["below_floor"] == 10  # the whole scanned cap


def test_lane_unavailable_without_encoder(conn):
    out = lane_dense(_ctx(conn, encoder=None), _qv("anything"), _slice())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_encoder"
    assert out.candidates == []


def test_lane_unavailable_no_index(conn):
    enc = _hashing_encoder()
    conn.execute("DROP TABLE unit_vectors_block")
    out = lane_dense(_ctx(conn, encoder=enc), _qv("x"), _slice())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_vector_index"


def test_lane_unavailable_no_units(conn):
    enc = _hashing_encoder()
    _text_corpus(conn, enc)
    conn.execute("DROP TABLE units")
    out = lane_dense(_ctx(conn, encoder=enc), _qv("x"), _slice())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_units_index"


def test_lane_unavailable_encoder_fails(conn):
    class _Boom:
        encoder_id = _ENC

        def encode(self, texts):
            raise RuntimeError("backend exploded")

    enc = _Boom()
    _text_corpus(conn, _hashing_encoder())
    ctx = _ctx(conn, encoder=enc)
    ctx.manifest["encoder_id"] = _ENC  # blocks exist under _ENC
    out = lane_dense(ctx, _qv("x"), _slice())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "encoder_failed"


def test_lane_unavailable_ambiguous_encoder(conn):
    enc = _hashing_encoder()
    _text_corpus(conn, enc)
    other = {f"w{i}": _rand_vec(random.Random(2)) for i in range(5)}
    _write_corpus(conn, other, encoder_id="other:enc:v9", block_start=1)
    # A bare callable carries no encoder_id and the manifest pins none —
    # two distinct vector spaces in storage make the space ambiguous.
    blob = enc.encode(["x"])[0]
    ctx = _ctx(conn, encoder=lambda _text: blob, encoder_id=None)
    ctx.manifest.pop("encoder_id", None)
    out = lane_dense(ctx, _qv("x"), _slice())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "encoder_id_ambiguous"


def test_lane_encoder_id_from_encoder_attr(conn):
    """Encoder-supplied identity pins the space — a foreign encoder's
    blocks in the same scope+generation are a different space, not
    ambiguity."""
    enc = _hashing_encoder()
    _text_corpus(conn, enc)
    other = {f"w{i}": _rand_vec(random.Random(2)) for i in range(5)}
    _write_corpus(conn, other, encoder_id="other:enc:v9", block_start=1)
    ctx = _ctx(conn, encoder=enc, encoder_id=None)  # manifest unpinned
    ctx.manifest.pop("encoder_id", None)
    out = lane_dense(ctx, _qv("alice moved to berlin in march 2024 "
                            "with her dog"), _slice(cap=5))
    assert out.status == LaneStatus.OK
    assert out.stats["encoder_id"] == enc.encoder_id
    assert all(c.unit_id.startswith("u-") for c in out.candidates)


def test_lane_eligibility_set_and_callable(conn):
    enc = _hashing_encoder()
    texts = _text_corpus(conn, enc)
    allowed = set(texts) - {"u-target", "u-close"}
    out = lane_dense(_ctx(conn, encoder=enc, eligible=allowed),
                     _qv("alice moved to berlin"), _slice(cap=10))
    assert out.status == LaneStatus.OK
    assert all(c.unit_id in allowed for c in out.candidates)
    assert out.eligible == len(allowed)

    out2 = lane_dense(
        _ctx(conn, encoder=enc,
             eligible=lambda row: row["speaker_canon"] != "held"),
        _qv("alice moved to berlin in march 2024 with her dog"),
        _slice(cap=10),
    )
    assert out2.status == LaneStatus.OK
    assert all(c.unit_id != "u-held" for c in out2.candidates)
    assert out2.eligible == len(texts) - 1
    assert out2.stats["eligibility_evaluated"] == len(texts)


def test_lane_units_generation_fence(conn):
    """``units`` is keyed (unit_id, generation) — the fence is
    ``generation <= pinned`` (V7-30.02 rebuild coexistence): an
    older-generation row still resolves and supplies metadata, while a
    rowmap key whose unit only exists ABOVE the pin — or not at all — is
    orphaned and never emitted."""
    enc = _hashing_encoder()
    _units(
        conn,
        [
            ("u-old", "src-old", 1, _SCOPE, "turn", "alice", 0),
            ("u-future", "src-new", 2, _SCOPE, "turn", "alice", 5),
        ],
    )
    blob = enc.encode(["stale generation unit"])[0]
    mx.write_block(
        conn, enc.encoder_id, _SCOPE, 1, 0,
        [("u-old", blob), ("u-future", blob), ("u-ghost", blob)],
    )
    out = lane_dense(_ctx(conn, encoder=enc), _qv("stale"), _slice(cap=5))
    assert out.status == LaneStatus.OK
    ids = {c.unit_id: c for c in out.candidates}
    assert set(ids) == {"u-old"}  # gen-0 row resolves at the gen-1 pin
    assert ids["u-old"].source_id == "src-old"
    assert ids["u-old"].revision == 1
    assert out.stats["orphaned_rowmap_keys"] == 2  # u-future + u-ghost


def test_lane_snapshot_below_pin(conn):
    """V7-30.02: ``unit_vectors_block`` is a complete rowmap snapshot per
    (encoder, scope, generation) — a bump that rewrote no blocks leaves
    the older snapshot live; the lane scans the newest generation
    at/below the pin."""
    enc = _hashing_encoder()
    _text_corpus(conn, enc)  # units + blocks all at generation 1
    ctx = _ctx(conn, encoder=enc)
    ctx.generation = 3  # pin bumped; no gen-2/3 vector blocks written
    out = lane_dense(
        ctx, _qv("alice moved to berlin in march 2024 with her dog"),
        _slice(cap=10),
    )
    assert out.status == LaneStatus.OK
    assert out.stats["vector_generation"] == 1
    assert out.examined == 31
    assert [c.unit_id for c in out.candidates[:2]] == ["u-held", "u-target"]


def test_lane_latest_snapshot_no_generation_mix(conn):
    """V7-30.02: when the same encoder has snapshots at two generations
    at/below the pin, only the newest generation's block set is scanned —
    rowmap keys from the stale generation never leak in."""
    enc = _hashing_encoder()
    _units(
        conn,
        [(k, "s", 1, _SCOPE, "turn", "alice", 1)
         for k in ("u-a", "u-b", "u-c")],
    )
    mx.write_block(
        conn, enc.encoder_id, _SCOPE, 1, 0,
        [("u-a", enc.encode(["alpha"])[0]),
         ("u-b", enc.encode(["beta"])[0])],
    )
    mx.write_block(
        conn, enc.encoder_id, _SCOPE, 2, 0,
        [("u-c", enc.encode(["gamma"])[0])],
    )
    ctx = _ctx(conn, encoder=enc)
    ctx.generation = 2
    out = lane_dense(ctx, _qv("gamma"), _slice(cap=10))
    assert out.status == LaneStatus.OK
    assert out.stats["vector_generation"] == 2
    assert out.examined == 1
    assert {c.unit_id for c in out.candidates} == {"u-c"}


def test_lane_deadline(conn, monkeypatch):
    enc = _hashing_encoder()
    _text_corpus(conn, enc)
    out = lane_dense(_ctx(conn, encoder=enc), _qv("x"), _slice(deadline_ms=0.0))
    assert out.status == LaneStatus.DEADLINE
    assert out.reason == "deadline"
    assert out.stats["encoded"] is False

    # Mid-scan cut: scripted monotonic clock expires during the block walk
    # (4 blocks of ~8 rows) -> PARTIAL with honest examined < total.
    conn2 = sqlite3.connect(":memory:")
    ensure_v7_additive(conn2)
    conn2.executescript(_ORACLE_DDL)
    _text_corpus(conn2, enc, block_size=8)
    seq = iter([0.0] * 6 + [1e9] * 64)
    monkeypatch.setattr(dense_mod, "_monotonic", lambda: next(seq))
    out2 = lane_dense(_ctx(conn2, encoder=enc), _qv("x"),
                      _slice(deadline_ms=1000.0, cap=10))
    assert out2.status == LaneStatus.PARTIAL
    assert out2.reason == "deadline"
    assert 0 < out2.examined < 31
    conn2.close()
