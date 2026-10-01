"""Contiguous per-(encoder_id, scope_id, generation) vector matrix blocks.

SPEC_V7 §07 (V7-07.04/05/06/14, D7-17): document vectors live in packed
blocks of the ``unit_vectors_block`` table — one contiguous matrix per
``(encoder_id, scope_id, generation, block_no)`` — so the dense lane scores
a whole block per fetch instead of decoding one vector BLOB per row (the
per-row scan is prohibited on the hot path above 2,000 eligible rows).
Per-row BLOBs survive only as the durable oracle (V7-07.04), used here for
the int8 → float32 rescore phase (V7-07.06).

Block layout (additive mirror of §30 ``unit_vectors_block``):

- ``data_blob`` — ``n_rows × dims`` packed values: little-endian float32
  for ``quant='f32'``, signed int8 for ``quant='int8'``.
- ``scale_blob`` — per-row metadata, little-endian float64:
  ``f32`` blocks store one value per row, the row's f64 L2 norm (so cosine
  is exact even when the writer did not pre-normalize); ``int8`` blocks
  store ``(quant_scale, f64_l2_norm)`` pairs, where ``quant_scale =
  max|v| / 127`` dequantizes the int8 row (``v ≈ scale · q``).
- ``rowmap_blob`` — ordered unit keys, one per matrix row:
  ``<I`` count, then per key ``<I`` utf8-length + utf8 bytes.

Scoring honors the §07 rules:

- Batched scan (V7-07.05): with numpy present, one matrix–vector product
  per block (eligible rows only); without numpy, an ``array('f')`` blocked
  loop. Both accumulate in float64 and share one heap + one final
  ``(-score, key)`` ordering, so identical inputs give identical top-k
  order — the test suite proves the two paths agree on a fixed corpus.
- Eligibility before rank: ``eligible_keys`` filters rowmap keys BEFORE a
  row is scored (and before its data is even decoded for scoring) — the
  lane applies the caller's predicate upstream and re-verifies at
  emission; a lane never widens the eligible set.
- int8 (V7-07.06): per-row scale quantization; phase-1 scans int8 dots and
  keeps a bounded top-4K heap; phase-2 rescores those rows with exact
  float32 vectors from the per-row oracle. Keys with no oracle row keep
  their approximate score and are labeled ``int8_approx`` in
  ``ScanResult.kinds`` — never silently upgraded.
- RSS (V7-07.14): blocks stream one at a time; resident structures are
  ``array('f')``/``array('b')``/numpy, never Python ``list[float]``
  buffers; the heaps are bounded at ``k`` (exact) and ``4·k`` (approx).
- Deadline: ``deadline_ms`` is a budget measured on ``clock`` (injectable
  for tests); expiry between blocks or mid-block stops the scan and marks
  the result ``partial`` — the returned hits are the best of the EXAMINED
  prefix only.

``bit`` quantization (V7-07.06, >200K rows) is declared but not
implemented in wave A — writing a bit block fails loudly with
``CAPABILITY_UNAVAILABLE``; a stored bit block is skipped and counted in
``stats['blocks_skipped_quant']`` rather than scored as garbage.

The per-row oracle table ``unit_vectors(unit_key, encoder_id, dims,
vector)`` is provisional — the durable per-row BLOB home lands with the
schema_v7 integration; ``write_oracle_row`` is the write seam so the
rescore path is real today.
"""

from __future__ import annotations

import heapq
import math
import sqlite3
import struct
import sys
import time
from array import array
from typing import Any, Callable, Iterator, Mapping, NamedTuple, Optional, Sequence

from ..core.types import ErrorCode, VerbatimError, require_id
from ..core.types_v7 import QuantMode
from .codec import Float32Codec

#: §30 table name — contiguous matrix blocks.
BLOCK_TABLE = "unit_vectors_block"
#: Provisional per-row f32 oracle table (V7-07.04 durable oracle). Marked
#: for the schema_v7 integration swap — the frozen DDL owns the final name.
ORACLE_TABLE = "unit_vectors"

#: Hard bound on rows per block — keeps one resident block ≤
#: ``MAX_BLOCK_ROWS × dims × bytes_per_dim`` (8K × 384 f32 ≈ 12 MB worst).
MAX_BLOCK_ROWS = 8192
#: V7-07.06 — a block space switches to int8 storage once its row count
#: crosses this bound; f32 below it (the float32 oracle still backs the
#: rescore phase for int8 blocks).
INT8_ROW_THRESHOLD = 20_000
#: V7-07.09 contextual embedding-header format tag, carried inside the
#: block space's ``encoder_id`` as ``<base encoder id>:hdr:v1`` (the
#: ``:`` joiner keeps the id inside ``require_id``'s charset).
#: ``hdr:v1`` = ``<YYYY-MM-DD | speaker | session label>: `` + the unit's
#: byte-pinned payload slice — see ``jobs/source_jobs._unit_embed_text``
#: for the exact field precedence. The header format is part of the
#: vector space's identity: a format change bumps the tag and forks the
#: space, so rows written under different preprocessing never mix in one
#: scan (V7-07.13). Producers write ``unit_vector_encoder_id(enc_id)``;
#: readers derive the same id from the pinned model id.
UNIT_HEADER_TAG = "hdr:v1"
#: Same bound as ``embeddings.vectors`` — per-row vector width limit.
MAX_DIMENSIONS = 8192
#: V7-07.06 — int8 phase-1 retains top ``RESCORE_FACTOR × k`` for the
#: float32 rescore.
RESCORE_FACTOR = 4
#: Deadline check cadence inside a block's pure-Python scoring loop.
_DEADLINE_STRIDE = 1024
#: Chunked IN() parameter bound (SQLite variable limit).
_LOOKUP_CHUNK = 400

_NP: Optional[Any] = None
_NP_TRIED = False


def _numpy() -> Optional[Any]:
    """Lazy optional numpy probe — never required, imported once.

    Absent OR broken-but-present (bad wheels raise on import) both resolve
    to the pure-Python path, which is the correctness reference.
    """
    global _NP, _NP_TRIED
    if _NP_TRIED:
        return _NP
    _NP_TRIED = True
    try:
        import numpy as np  # type: ignore
    except Exception:
        _NP = None
    else:
        _NP = np
    return _NP


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    return row is not None


def unit_vector_encoder_id(encoder_id: str) -> str:
    """Block-space id for unit vectors: ``<encoder_id>:hdr:<version>``.

    The ``encoder_id`` a caller pins names the *model*; the producer's
    ``unit_vectors_block`` space additionally binds the contextual-header
    format version (V7-07.09 — headers are part of the embedded text, so
    a format change is a different vector space). Idempotent: an id that
    already carries a ``:hdr:`` segment is returned unchanged, so a
    caller pinning a resolved space id stays on it.
    """
    if ":hdr:" in encoder_id:
        return encoder_id
    return f"{encoder_id}:{UNIT_HEADER_TAG}"


_ORACLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {ORACLE_TABLE} (
    unit_key TEXT NOT NULL,
    encoder_id TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (unit_key, encoder_id)
);
"""


def ensure_oracle_table(conn: sqlite3.Connection) -> None:
    """Provision the provisional per-row f32 oracle (idempotent, additive).

    The ``unit_vectors`` table is this module's own write seam — the V7
    block producer calls this inside its fenced commit so a store that
    never ran a per-row vector path still gets the durable oracle the
    int8 rescore phase reads (V7-07.04/07.06).
    """
    conn.execute(_ORACLE_DDL)


class _Desc(str):
    """str with inverted ordering for deterministic heap ties.

    ``heapq`` is a min-heap: entries sort ``(score, _Desc(key), key)`` so
    the lowest score pops first and equal scores keep the LARGEST key at
    the bottom — matching the output ordering ``(-score, key)``.
    """

    def __lt__(self, other: str) -> bool:
        return str.__gt__(self, other)

    def __le__(self, other: str) -> bool:
        return str.__ge__(self, other)


# ---------------------------------------------------------------------------
# rowmap / blob codecs
# ---------------------------------------------------------------------------


def _pack_rowmap(keys: Sequence[str]) -> bytes:
    out = bytearray(struct.pack("<I", len(keys)))
    for key in keys:
        kb = key.encode("utf-8")
        out += struct.pack("<I", len(kb))
        out += kb
    return bytes(out)


def _unpack_rowmap(blob: bytes, expected: int) -> tuple[str, ...]:
    keys: list[str] = []
    pos = 0
    n = struct.unpack_from("<I", blob, pos)[0]
    pos += 4
    for _ in range(n):
        (klen,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        keys.append(blob[pos : pos + klen].decode("utf-8"))
        pos += klen
    if n != expected:
        raise VerbatimError(
            ErrorCode.VECTOR_INVALID,
            f"rowmap holds {n} keys but block declares {expected} rows",
        )
    return tuple(keys)


def _decode_f32(blob: bytes) -> array:
    vec = array("f")
    vec.frombytes(blob)
    if sys.byteorder == "big":
        # Stored blobs are little-endian; array('f') is native-endian.
        vec.byteswap()
    return vec


def _decode_scales(blob: Optional[bytes], n_rows: int, width: int) -> tuple[float, ...]:
    if not blob:
        return (0.0,) * n_rows * width
    expected = n_rows * width
    if len(blob) != expected * 8:
        raise VerbatimError(
            ErrorCode.VECTOR_INVALID,
            f"scale_blob holds {len(blob)} bytes, expected {expected * 8}",
        )
    return struct.unpack(f"<{expected}d", bytes(blob))


# ---------------------------------------------------------------------------
# public block API
# ---------------------------------------------------------------------------


class MatrixBlock(NamedTuple):
    """One decoded ``unit_vectors_block`` row.

    ``data`` is flat: ``array('f')`` of ``n_rows × dims`` for f32,
    ``array('b')`` for int8. ``norms`` are per-row f32 L2 norms of the
    ORIGINAL vectors; ``scales`` are per-row int8 dequant scales (``None``
    for f32). ``keys[i]`` is the unit key of matrix row ``i``.
    """

    encoder_id: str
    scope_id: str
    generation: int
    block_no: int
    n_rows: int
    dims: int
    quant: str
    keys: tuple[str, ...]
    norms: tuple[float, ...]
    scales: Optional[tuple[float, ...]]
    data: array

    def row(self, i: int) -> list[float]:
        """Row ``i`` dequantized to floats (exact f32, or scale·int8)."""
        base = i * self.dims
        seg = self.data[base : base + self.dims]
        if self.quant == QuantMode.INT8.value:
            s = self.scales[i] if self.scales is not None else 0.0
            return [s * float(x) for x in seg]
        return [float(x) for x in seg]

    def vectors(self) -> Iterator[list[float]]:
        for i in range(self.n_rows):
            yield self.row(i)


def _as_float_vec(vec: Any, dims_hint: Optional[int] = None) -> tuple[float, ...]:
    """Normalize a vector input (f32le bytes | Sequence[float]) to floats."""
    if isinstance(vec, (bytes, bytearray, memoryview)):
        blob = bytes(vec)
        if not blob or len(blob) % 4:
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID,
                f"vector blob length {len(blob)} is not a float32 multiple",
            )
        out = Float32Codec.unpack(blob, len(blob) // 4)
    else:
        try:
            out = tuple(float(x) for x in vec)
        except (TypeError, ValueError) as exc:
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID, f"vector component not numeric: {exc}"
            ) from exc
    if dims_hint is not None and len(out) != dims_hint:
        raise VerbatimError(
            ErrorCode.VECTOR_INVALID,
            f"vector length {len(out)} != declared dims {dims_hint}",
        )
    if not out:
        raise VerbatimError(ErrorCode.VECTOR_INVALID, "empty vector")
    for x in out:
        if not math.isfinite(x):
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID, "vector contains NaN or infinite component"
            )
    return out


def _l2(vec: Sequence[float]) -> float:
    n = math.fsum(x * x for x in vec)
    return math.sqrt(n) if n > 0.0 else 0.0


def write_block(
    conn: sqlite3.Connection,
    encoder_id: str,
    scope_id: str,
    generation: int,
    block_no: int,
    rows: Sequence[tuple[str, Any]],
    *,
    quant: str = "f32",
) -> int:
    """Pack ``(unit_key, vec)`` rows into one ``unit_vectors_block`` row.

    ``vec`` accepts a float sequence or a packed f32le blob. ``quant`` is
    ``'f32'`` (exact) or ``'int8'`` (per-row scale); ``'bit'`` is declared
    in §30 but unimplemented in wave A and fails loudly. Returns the row
    count written; an empty ``rows`` writes nothing and returns 0.

    Blocks are immutable-by-convention for a fixed
    ``(encoder_id, scope_id, generation, block_no)`` — rewrite is allowed
    (idempotent rebuild writes the same bytes) but mixing generations into
    one space is impossible because the generation is part of the key
    (V7-07.13).
    """
    require_id(encoder_id, "encoder_id")
    require_id(scope_id, "scope_id")
    if not isinstance(generation, int) or generation < 0:
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid generation {generation!r}")
    if not isinstance(block_no, int) or block_no < 0:
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid block_no {block_no!r}")
    try:
        mode = QuantMode(quant)
    except ValueError:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"invalid quant {quant!r}"
        ) from None
    if mode is QuantMode.BIT:
        raise VerbatimError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "bit quantization not implemented in wave A (V7-07.06: permitted "
            ">200K rows only; lands with the rescore-index work)",
        )
    if not rows:
        return 0
    if len(rows) > MAX_BLOCK_ROWS:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"block row count {len(rows)} exceeds {MAX_BLOCK_ROWS}",
        )

    # Single pass, packed bytes only — no whole-block ``list[float]``
    # buffer is retained even on the write path (V7-07.14). Norms and
    # quantization parameters are computed over the f32-ROUNDED stored
    # values, not the f64 inputs — the durable per-row oracle and any
    # independent decoder see exactly these bytes, so the block's own
    # metadata must describe them (bit-comparable scoring paths).
    keys: list[str] = []
    seen: set[str] = set()
    data = bytearray()
    scales = bytearray()
    dims: Optional[int] = None
    rowpack = rowunpack = None
    for key, vec in rows:
        if not isinstance(key, str) or not key:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid unit key {key!r}")
        if key in seen:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"duplicate unit key in block: {key!r}"
            )
        seen.add(key)
        v = _as_float_vec(vec, dims)
        if dims is None:
            dims = len(v)
            if dims > MAX_DIMENSIONS:
                raise VerbatimError(
                    ErrorCode.VECTOR_INVALID, f"dims {dims} exceeds {MAX_DIMENSIONS}"
                )
            rowpack = struct.Struct(f"<{dims}f").pack
            rowunpack = struct.Struct(f"<{dims}f").unpack
        keys.append(key)
        vq = rowunpack(rowpack(*v))  # the stored f32 row
        if mode is QuantMode.F32:
            data += rowpack(*v)
            scales += struct.pack("<d", _l2(vq))
        else:  # int8 — per-row scale (V7-07.06)
            absmax = max(abs(x) for x in vq)
            s = absmax / 127.0 if absmax > 0.0 else 0.0
            if s > 0.0:
                qrow = array(
                    "b",
                    [
                        max(-127, min(127, int(round(x / s))))
                        for x in vq
                    ],
                )
            else:
                qrow = array("b", bytes(dims))
            data += qrow.tobytes()
            scales += struct.pack("<dd", s, _l2(vq))
    assert dims is not None

    conn.execute(
        f"""
        INSERT INTO {BLOCK_TABLE}
            (encoder_id, scope_id, generation, block_no,
             n_rows, dims, quant, scale_blob, data_blob, rowmap_blob)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(encoder_id, scope_id, generation, block_no)
        DO UPDATE SET
            n_rows = excluded.n_rows,
            dims = excluded.dims,
            quant = excluded.quant,
            scale_blob = excluded.scale_blob,
            data_blob = excluded.data_blob,
            rowmap_blob = excluded.rowmap_blob
        """,
        (
            encoder_id,
            scope_id,
            generation,
            block_no,
            len(keys),
            dims,
            mode.value,
            bytes(scales),
            bytes(data),
            _pack_rowmap(keys),
        ),
    )
    return len(keys)


def _decode_block_row(row: Any, *, with_data: bool = True) -> MatrixBlock:
    (encoder_id, scope_id, generation, block_no, n_rows, dims, quant,
     scale_blob, data_blob, rowmap_blob) = row
    keys = _unpack_rowmap(bytes(rowmap_blob), int(n_rows))
    if quant == QuantMode.INT8.value:
        flat = _decode_scales(scale_blob, int(n_rows), 2)
        scales = tuple(flat[2 * i] for i in range(int(n_rows)))
        norms = tuple(flat[2 * i + 1] for i in range(int(n_rows)))
        data = array("b", bytes(data_blob)) if with_data else array("b")
    elif quant == QuantMode.F32.value:
        scales = None
        norms = _decode_scales(scale_blob, int(n_rows), 1)
        data = _decode_f32(bytes(data_blob)) if with_data else array("f")
    else:
        raise VerbatimError(
            ErrorCode.VECTOR_INVALID, f"unsupported quant {quant!r} in stored block"
        )
    if with_data and len(data) != int(n_rows) * int(dims):
        raise VerbatimError(
            ErrorCode.VECTOR_INVALID,
            f"data_blob decodes to {len(data)} values, "
            f"expected {int(n_rows) * int(dims)}",
        )
    return MatrixBlock(
        encoder_id, scope_id, int(generation), int(block_no), int(n_rows),
        int(dims), quant, keys, norms, scales, data,
    )


def _block_rows(
    conn: sqlite3.Connection,
    encoder_id: str,
    scope_id: str,
    generation: int,
) -> Iterator:
    """Raw block rows ordered by ``block_no`` (deterministic scan order)."""
    yield from conn.execute(
        f"""
        SELECT encoder_id, scope_id, generation, block_no,
               n_rows, dims, quant, scale_blob, data_blob, rowmap_blob
        FROM {BLOCK_TABLE}
        WHERE encoder_id = ? AND scope_id = ? AND generation = ?
        ORDER BY block_no
        """,
        (encoder_id, scope_id, generation),
    )


def read_block(
    conn: sqlite3.Connection,
    encoder_id: str,
    scope_id: str,
    generation: int,
    block_no: int,
) -> Optional[MatrixBlock]:
    """Decode one block, or ``None`` when absent."""
    row = conn.execute(
        f"""
        SELECT encoder_id, scope_id, generation, block_no,
               n_rows, dims, quant, scale_blob, data_blob, rowmap_blob
        FROM {BLOCK_TABLE}
        WHERE encoder_id = ? AND scope_id = ? AND generation = ? AND block_no = ?
        """,
        (encoder_id, scope_id, generation, block_no),
    ).fetchone()
    if row is None:
        return None
    return _decode_block_row(row)


def iter_blocks(
    conn: sqlite3.Connection,
    encoder_id: str,
    scope_id: str,
    generation: int,
) -> Iterator[MatrixBlock]:
    """Decode blocks in ``block_no`` order — the streaming inspection API."""
    for row in _block_rows(conn, encoder_id, scope_id, generation):
        yield _decode_block_row(row)


def block_row_keys(
    conn: sqlite3.Connection,
    encoder_id: str,
    scope_id: str,
    generation: int,
) -> set[str]:
    """All rowmap unit keys across a space's blocks — metadata only.

    Decodes ``rowmap_blob`` without touching ``data_blob`` — the cheap
    "which unit ids does this snapshot actually carry" probe the dense
    lane's coverage accounting and the producer's idempotent append both
    need. A block whose rowmap won't decode is skipped (fail-closed
    toward *under*-counting coverage, which reports partial rather than
    claiming rows that may not exist).
    """
    out: set[str] = set()
    for n_rows, blob in conn.execute(
        f"SELECT n_rows, rowmap_blob FROM {BLOCK_TABLE}"
        " WHERE encoder_id = ? AND scope_id = ? AND generation = ?"
        " ORDER BY block_no",
        (encoder_id, scope_id, generation),
    ):
        try:
            out.update(_unpack_rowmap(bytes(blob), int(n_rows)))
        except Exception:
            continue
    return out


def write_oracle_row(
    conn: sqlite3.Connection,
    encoder_id: str,
    unit_key: str,
    vec: Any,
) -> None:
    """Persist one per-row f32 oracle vector (provisional ``unit_vectors``).

    The oracle is the durable per-row BLOB of V7-07.04 — used only for the
    int8 → f32 rescore, never on the scan hot path.
    """
    require_id(encoder_id, "encoder_id")
    if not isinstance(unit_key, str) or not unit_key:
        raise VerbatimError(ErrorCode.VALIDATION, f"invalid unit key {unit_key!r}")
    v = _as_float_vec(vec)
    conn.execute(
        f"""
        INSERT INTO {ORACLE_TABLE} (unit_key, encoder_id, dims, vector)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(unit_key, encoder_id) DO UPDATE SET
            dims = excluded.dims,
            vector = excluded.vector
        """,
        (unit_key, encoder_id, len(v), Float32Codec.pack(v)),
    )


def _table_oracle(
    conn: sqlite3.Connection, encoder_id: str, keys: Sequence[str]
) -> dict[str, bytes]:
    """Default rescore oracle: per-row f32 blobs from ``unit_vectors``."""
    if not keys or not _table_exists(conn, ORACLE_TABLE):
        return {}
    out: dict[str, bytes] = {}
    for i in range(0, len(keys), _LOOKUP_CHUNK):
        part = list(keys[i : i + _LOOKUP_CHUNK])
        marks = ",".join("?" for _ in part)
        for k, blob in conn.execute(
            f"SELECT unit_key, vector FROM {ORACLE_TABLE}"
            f" WHERE encoder_id = ? AND unit_key IN ({marks})",
            [encoder_id, *part],
        ):
            out[k] = bytes(blob)
    return out


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


class ScanResult(list):
    """``list[(unit_key, score)]`` plus honest scan metadata.

    Attributes: ``partial`` (deadline cut — hits cover the examined prefix
    only), ``examined`` (rowmap rows inspected), ``eligible`` (rows passing
    the eligibility filter), ``scored`` (rows cosine-scored), ``kinds``
    (per-emitted-key score provenance: ``f32`` | ``int8_rescored`` |
    ``int8_approx``), ``stats`` (granular counters for coverage reporting).
    """

    def __init__(self) -> None:
        super().__init__()
        self.partial: bool = False
        self.examined: int = 0
        self.eligible: int = 0
        self.scored: int = 0
        self.kinds: dict[str, str] = {}
        self.stats: dict[str, Any] = {}


def _heap_push(heap: list, bound: int, score: float, key: str) -> None:
    entry = (score, _Desc(key), key)
    if len(heap) < bound:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _eligible_keys(
    eligible: Any, keys: tuple[str, ...]
) -> Optional[set[str]]:
    """Resolve the eligibility filter for one block's rowmap keys.

    ``eligible`` may be ``None`` (no restriction), a container of unit
    keys, a per-key callable, or an object exposing
    ``filter_keys(list[str]) -> Iterable[str]`` for batch resolution (the
    dense lane's unit-row adapter). Returns ``None`` for "all pass".
    """
    if eligible is None:
        return None
    fk = getattr(eligible, "filter_keys", None)
    if fk is not None:
        return set(fk(list(keys)))
    if callable(eligible):
        return {k for k in keys if eligible(k)}
    return {k for k in keys if k in eligible}


def _cosine_oracle(
    query: tuple[float, ...], q_norm: float, blob: bytes, dims: int
) -> Optional[float]:
    try:
        vec = Float32Codec.unpack(blob, dims)
    except VerbatimError:
        return None
    norm = _l2(vec)
    if norm <= 0.0 or not math.isfinite(norm):
        return None
    return math.fsum(a * b for a, b in zip(query, vec)) / (q_norm * norm)


def scan(
    conn: sqlite3.Connection,
    encoder_id: str,
    scope_id: str,
    generation: int,
    query_vec: Any,
    *,
    eligible_keys: Any = None,
    k: int = 40,
    deadline_ms: Optional[float] = None,
    rescore: bool = True,
    oracle: Optional[Callable[[sqlite3.Connection, str, list[str]], Mapping[str, Any]]] = None,
    clock: Optional[Callable[[], float]] = None,
    use_numpy: Optional[bool] = None,
) -> ScanResult:
    """Exact/quantized cosine top-k over the contiguous block matrix.

    Streams ``unit_vectors_block`` rows for ``(encoder_id, scope_id,
    generation)`` in ``block_no`` order, filters rowmap keys through
    ``eligible_keys`` BEFORE scoring, and returns ``(key, score)`` sorted
    ``(-score, key)``. ``k < 1`` is a validation error; ``k`` bounds both
    the result and the resident heap.

    ``eligible_keys``: ``None`` → no restriction; container → membership on
    unit keys; callable → ``eligible_keys(key)`` per key; object with
    ``filter_keys(keys)`` → batch resolution. Applied before any scoring
    and before emission — the lane additionally re-verifies emitted keys
    against unit rows.

    ``deadline_ms`` budgets the whole scan on ``clock`` (monotonic
    seconds); expiry marks the result ``partial``. ``rescore=False``
    disables the int8→f32 oracle phase (results labeled ``int8_approx``).
    ``oracle`` overrides the default ``unit_vectors`` lookup with
    ``oracle(conn, encoder_id, keys) -> Mapping[key, f32le-bytes]``.
    ``use_numpy`` forces the acceleration decision (``None`` = auto-probe).
    """
    if k < 1:
        raise VerbatimError(ErrorCode.VALIDATION, "k must be >= 1")
    res = ScanResult()
    stats: dict[str, Any] = {
        "encoder_id": encoder_id,
        "scope_id": scope_id,
        "generation": generation,
        "k": k,
        "blocks_total": 0,
        "blocks_scanned": 0,
        "blocks_dims_mismatch": 0,
        "blocks_skipped_quant": 0,
        "excluded_zero_norm": 0,
        "duplicate_keys": 0,
        "rescore": "none_needed",
        "oracle_missing": 0,
        "partial": False,
        "stop_reason": None,
        "quant": [],
    }
    res.stats = stats
    if not _table_exists(conn, BLOCK_TABLE):
        stats["stop_reason"] = "no_block_table"
        return res

    query = _as_float_vec(query_vec)
    qdims = len(query)
    stats["dims"] = qdims
    q_norm = _l2(query)
    if q_norm <= 0.0 or not math.isfinite(q_norm):
        stats["stop_reason"] = "zero_norm_query"
        return res

    # Sized-but-empty eligible container short-circuits honestly.
    if (
        eligible_keys is not None
        and not callable(eligible_keys)
        and getattr(eligible_keys, "filter_keys", None) is None
        and hasattr(eligible_keys, "__len__")
        and len(eligible_keys) == 0
    ):
        stats["stop_reason"] = "eligible_empty"
        return res

    if isinstance(eligible_keys, (list, tuple)):
        eligible_keys = frozenset(eligible_keys)

    np = _numpy() if use_numpy is None else (_numpy() if use_numpy else None)
    stats["numpy"] = bool(np)
    if use_numpy and np is None:
        stats["numpy_note"] = "requested_but_absent"
    q64: Any = np.asarray(query, dtype=np.float64) if np is not None else query

    now = clock or time.monotonic
    t0 = now()
    cutoff = None if deadline_ms is None else t0 + deadline_ms / 1000.0

    def expired() -> bool:
        return cutoff is not None and now() >= cutoff

    exact_heap: list = []   # f32 rows — scores are final
    approx_heap: list = []  # int8 rows — scores are approximations
    approx_bound = max(RESCORE_FACTOR * k, k)
    seen: set[str] = set()
    quants: set[str] = set()

    for row in _block_rows(conn, encoder_id, scope_id, generation):
        if expired():
            res.partial = True
            break
        stats["blocks_total"] += 1
        (benc, bscope, bgen, bno, n_rows, bdims, quant,
         scale_blob, data_blob, rowmap_blob) = row
        n_rows = int(n_rows)
        bdims = int(bdims)
        keys = _unpack_rowmap(bytes(rowmap_blob), n_rows)
        res.examined += n_rows

        ok = _eligible_keys(eligible_keys, keys)
        # ``eligible`` counts distinct rowmap keys that pass the filter and
        # match the query's vector space (dims + supported quant) — the
        # dims/quant skips below subtract rows that could never score.
        ok_idx = [
            i for i, key in enumerate(keys)
            if (ok is None or key in ok) and key not in seen
        ]
        stats["duplicate_keys"] += sum(
            1 for key in keys
            if (ok is None or key in ok) and key in seen
        )
        res.eligible += len(ok_idx)

        if not ok_idx:
            continue  # eligibility-before-rank: never decoded for scoring
        for i in ok_idx:
            seen.add(keys[i])

        if bdims != qdims:
            stats["blocks_dims_mismatch"] += 1
            res.eligible -= len(ok_idx)  # different vector space — not eligible
            continue
        if quant not in (QuantMode.F32.value, QuantMode.INT8.value):
            stats["blocks_skipped_quant"] += 1
            res.eligible -= len(ok_idx)
            continue
        stats["blocks_scanned"] += 1
        quants.add(quant)

        if quant == QuantMode.INT8.value:
            flat = _decode_scales(scale_blob, n_rows, 2)
            scales = flat[0::2]
            norms = flat[1::2]
            data = array("b", bytes(data_blob))
            if np is not None:
                m = np.frombuffer(bytes(data_blob), dtype=np.int8).reshape(
                    n_rows, bdims
                )
                idx = np.asarray(ok_idx, dtype=np.int64)
                dots = m[idx].astype(np.float64) @ q64
                sc = np.asarray([scales[i] for i in ok_idx])
                nm = np.asarray([norms[i] for i in ok_idx])
                with np.errstate(divide="ignore", invalid="ignore"):
                    approx = dots * sc / (q_norm * nm)
                bad = ~np.isfinite(approx)
                bad |= nm <= 0.0
                stats["excluded_zero_norm"] += int(bad.sum())
                res.scored += int((~bad).sum())
                # Only a block's own top-``bound`` rows can survive into the
                # global bounded heap — argpartition keeps the Python-level
                # push count O(bound) per block, not O(rows). Ties AT the
                # cutoff are all included (``cand >= thresh``) so the heap's
                # deterministic ``(-score, key)`` ordering — not argpartition's
                # arbitrary tie pick — decides membership.
                cand = np.where(bad, -np.inf, approx)
                if len(ok_idx) > approx_bound:
                    top = np.argpartition(cand, -approx_bound)[-approx_bound:]
                    sel = np.nonzero(cand >= cand[top].min())[0]
                else:
                    sel = np.arange(len(ok_idx))
                for j in sel:
                    val = float(cand[j])
                    if not math.isfinite(val):
                        continue
                    _heap_push(approx_heap, approx_bound, val, keys[ok_idx[int(j)]])
            else:
                for j, i in enumerate(ok_idx):
                    if j % _DEADLINE_STRIDE == _DEADLINE_STRIDE - 1 and expired():
                        res.partial = True
                        break
                    norm = norms[i]
                    if norm <= 0.0 or not math.isfinite(norm):
                        stats["excluded_zero_norm"] += 1
                        continue
                    base = i * bdims
                    dot = 0.0
                    for d in range(bdims):
                        dot += query[d] * data[base + d]
                    val = scales[i] * dot / (q_norm * norm)
                    res.scored += 1
                    _heap_push(approx_heap, approx_bound, val, keys[i])
        else:
            norms = _decode_scales(scale_blob, n_rows, 1)
            data = _decode_f32(bytes(data_blob))
            if np is not None:
                m = np.frombuffer(bytes(data_blob), dtype="<f4").reshape(
                    n_rows, bdims
                )
                idx = np.asarray(ok_idx, dtype=np.int64)
                dots = m[idx].astype(np.float64) @ q64
                nm = np.asarray([norms[i] for i in ok_idx])
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals = dots / (q_norm * nm)
                bad = ~np.isfinite(vals)
                bad |= nm <= 0.0
                stats["excluded_zero_norm"] += int(bad.sum())
                res.scored += int((~bad).sum())
                cand = np.where(bad, -np.inf, vals)
                if len(ok_idx) > k:
                    top = np.argpartition(cand, -k)[-k:]
                    sel = np.nonzero(cand >= cand[top].min())[0]
                else:
                    sel = np.arange(len(ok_idx))
                for j in sel:
                    val = float(cand[j])
                    if not math.isfinite(val):
                        continue
                    _heap_push(exact_heap, k, val, keys[ok_idx[int(j)]])
            else:
                for j, i in enumerate(ok_idx):
                    if j % _DEADLINE_STRIDE == _DEADLINE_STRIDE - 1 and expired():
                        res.partial = True
                        break
                    norm = norms[i]
                    if norm <= 0.0 or not math.isfinite(norm):
                        stats["excluded_zero_norm"] += 1
                        continue
                    base = i * bdims
                    dot = 0.0
                    for d in range(bdims):
                        dot += query[d] * data[base + d]
                    val = dot / (q_norm * norm)
                    res.scored += 1
                    _heap_push(exact_heap, k, val, keys[i])
        if res.partial:
            break

    # Phase 2 — int8 top-4K f32 rescore (V7-07.06). Skipped honestly when
    # the deadline fired, rescore is disabled, or no oracle resolves.
    rescored: dict[str, float] = {}
    kinds: dict[str, str] = {}
    if approx_heap:
        if res.partial:
            stats["rescore"] = "deadline_skipped"
        elif not rescore:
            stats["rescore"] = "disabled"
        elif expired():
            res.partial = True
            stats["rescore"] = "deadline_skipped"
        else:
            fetch = oracle or _table_oracle
            oracle_map = dict(fetch(conn, encoder_id, [e[2] for e in approx_heap]))
            stats["rescore"] = "exact" if oracle_map else "unavailable"
            for _score, _dk, key in approx_heap:
                blob = oracle_map.get(key)
                val = None
                if blob is not None:
                    val = _cosine_oracle(query, q_norm, bytes(blob), qdims)
                if val is None:
                    stats["oracle_missing"] += 1
                    continue
                rescored[key] = val

    pool: list[tuple[str, float, str]] = []
    for score, _dk, key in exact_heap:
        kinds[key] = QuantMode.F32.value
        pool.append((key, score, kinds[key]))
    for score, _dk, key in approx_heap:
        if key in rescored:
            kinds[key] = "int8_rescored"
            pool.append((key, rescored[key], kinds[key]))
        else:
            kinds[key] = "int8_approx"
            pool.append((key, score, kinds[key]))

    pool.sort(key=lambda t: (-t[1], t[0]))
    for key, score, _kind in pool[:k]:
        res.append((key, score))
        res.kinds[key] = kinds[key]
    stats["partial"] = res.partial
    if res.partial:
        stats["stop_reason"] = "deadline"
    stats["quant"] = sorted(quants)
    return res


#: ``docs/v7_contracts.md`` names the block scan ``scan_block`` — alias so
#: workers coding against either spelling get the same function.
scan_block = scan


__all__ = [
    "BLOCK_TABLE",
    "ORACLE_TABLE",
    "MAX_BLOCK_ROWS",
    "MAX_DIMENSIONS",
    "RESCORE_FACTOR",
    "INT8_ROW_THRESHOLD",
    "UNIT_HEADER_TAG",
    "unit_vector_encoder_id",
    "ensure_oracle_table",
    "MatrixBlock",
    "ScanResult",
    "write_block",
    "read_block",
    "iter_blocks",
    "block_row_keys",
    "write_oracle_row",
    "scan",
    "scan_block",
]
