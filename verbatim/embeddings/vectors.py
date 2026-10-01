"""In-process exact cosine search over the authorized vector set.

SPEC §28 defers ANN indexes until measurements demand them: the first
implementation is exact cosine over an already-authorized candidate set,
performed entirely in-process — no vector database, no index service, no
network.

Boundaries:

- Both ``load_matrix`` and :func:`search` fetch vectors ONLY for
  caller-supplied span IDs that were authorized upstream. They never
  enumerate the embeddings table on their own — the caller's span-id list
  is the authorization bound.
- :func:`search` is the scalable path (V4-28.05): it streams bounded
  batches of eligible vectors and keeps a bounded top-k heap, so the
  eligible set may be arbitrarily large without materializing a
  whole-corpus matrix or rejecting it at a fixed input bound. Deadline
  expiry stops the scan and is reported through :class:`ScanCoverage`
  rather than silently truncating.
- ``load_matrix`` remains the materializing API for callers that need the
  full matrix in memory (bounded only by the caller's id list — the old
  4,096-id whole-input rejection is gone because rejection at a batch-size
  bound is exactly the F4-12 defect).
- Stale, generation-mismatched, and malformed rows are excluded from
  scoring AND counted in :class:`ScanCoverage` — a zero vector has no
  cosine direction and a stale vector answers a different input.

NumPy accelerates the dot-product math when installed (optional
``semantic`` extra, lazy import); the pure-Python ``array`` path is the
correctness reference and always available.
"""

from __future__ import annotations

import heapq
import math
import sqlite3
import sys
from array import array
from typing import Any, Callable, NamedTuple, Optional, Sequence

from ..core.types import ErrorCode, VerbatimError
from .codec import Float32Codec

# ``MAX_SPAN_IDS`` is retained for import compatibility only — it no
# longer bounds load_matrix input (that rejection WAS the F4-12 defect).
# ``MAX_DIMENSIONS`` still bounds per-row vector size.
MAX_SPAN_IDS = 4096
MAX_DIMENSIONS = 8192
# Rows fetched per scan round in ``search`` — bounds peak memory to
# SCAN_BATCH float32 vectors regardless of corpus size.
SCAN_BATCH = 512
# IN() parameter bound for chunked lookups (SQLite variable limit).
_LOOKUP_CHUNK = 400

_NP: Optional[Any] = None
_NP_TRIED = False


def _numpy() -> Optional[Any]:
    """Lazy optional numpy probe — never required, imported once."""
    global _NP, _NP_TRIED
    if _NP_TRIED:
        return _NP
    _NP_TRIED = True
    try:
        import numpy as np  # type: ignore
    except Exception:
        # Absent OR broken-but-present (bad binary wheels raise
        # RuntimeError/OSError on import): either way the pure-Python
        # path answers — it is the correctness reference anyway.
        _NP = None
    else:
        _NP = np
    return _NP


class EmbeddingMatrix(NamedTuple):
    """Authorized matrix: ``ids[i]`` pairs ``vectors[i]``.

    A NamedTuple so ``ids, vectors = load_matrix(...)`` unpacks naturally
    while the whole object passes to :func:`cosine` as one ``matrix``.
    """

    ids: list[str]
    vectors: list[array]


class ScanCoverage(NamedTuple):
    """Honest coverage counters for one :func:`search` scan (V4-29.05/06).

    ``eligible`` is the number of distinct authorized span ids the caller
    asked to be considered; ``examined`` how many had their embedding row
    actually fetched; ``scored`` how many vectors were cosine-scored;
    ``returned`` the hit count. ``partial`` is True iff a deadline (or an
    injected stop) cut the scan — the returned hits are then the best of
    the EXAMINED prefix only, and exact-search claims do not apply.

    Exclusions are counted by cause so stale-input and space-mismatch
    losses are reported rather than silently scored or silently dropped:

    - ``excluded_stale`` — ``source_hmac`` no longer matches the span's
      persisted ``excerpt_hmac`` (the embedded text changed), or the
      recorded ``embedding_inputs`` dependency digest disagrees.
    - ``excluded_generation`` — row ``dtype``/``dimensions``/preprocessing
      do not match the query vector's space, or ``input_known_seq``
      postdates the ``known_at_seq`` cutoff.
    - ``excluded_malformed`` — corrupt blob length, bad dimensions, or a
      zero/non-finite vector.
    - ``missing`` — examined span ids with no embedding row at all.
    """

    eligible: int = 0
    examined: int = 0
    scored: int = 0
    returned: int = 0
    partial: bool = False
    excluded_stale: int = 0
    excluded_generation: int = 0
    excluded_malformed: int = 0
    missing: int = 0


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    return row is not None


def _decode(blob: bytes, dims: int) -> array:
    vec = array("f")
    vec.frombytes(blob)
    if sys.byteorder == "big":
        # Stored blobs are little-endian; array('f') is native-endian.
        vec.byteswap()
    return vec


def load_matrix(
    conn: sqlite3.Connection,
    span_ids: Sequence[str],
    encoder_id: str,
) -> EmbeddingMatrix:
    """Fetch embedding vectors for an authorized span-id set.

    Rows whose blob length does not equal ``dimensions * 4`` are corrupt or
    identity-mismatched and are EXCLUDED — reported as missing coverage via
    :func:`coverage` rather than scored as garbage.

    Any input size is accepted: chunked ``IN()`` lookups keep parameter
    counts inside SQLite's limit. Callers needing bounded memory or
    deadline-aware partial scans should use :func:`search` instead —
    this function materializes every requested vector.
    """
    if not span_ids:
        return EmbeddingMatrix([], [])
    # Chunked IN() lookups keep parameter counts inside SQLite's limit while
    # staying fully parameterized (SPEC §41: no string-built SQL).
    rows: list[tuple[str, bytes, int]] = []
    ids_list = list(dict.fromkeys(span_ids))  # dedupe, preserve order
    for i in range(0, len(ids_list), _LOOKUP_CHUNK):
        part = ids_list[i : i + _LOOKUP_CHUNK]
        marks = ",".join("?" for _ in part)
        cur = conn.execute(
            f"SELECT span_id, vector, dimensions FROM embeddings "
            f"WHERE encoder_id = ? AND span_id IN ({marks})",
            [encoder_id, *part],
        )
        rows.extend(cur.fetchall())
    ids: list[str] = []
    vectors: list[array] = []
    for span_id, blob, dims in rows:
        if not isinstance(dims, int) or dims < 1 or dims > MAX_DIMENSIONS:
            continue
        if not isinstance(blob, (bytes, bytearray)) or len(blob) != dims * 4:
            continue
        ids.append(span_id)
        vectors.append(_decode(bytes(blob), dims))
    return EmbeddingMatrix(ids, vectors)


def _batch_rows(conn: sqlite3.Connection, part: Sequence[str],
                encoder_id: str, join_spans: bool) -> list:
    """One bounded batch of embedding rows for authorized span ids."""
    marks = ",".join("?" for _ in part)
    if join_spans:
        sql = (
            "SELECT e.span_id, e.vector, e.dimensions, e.dtype,"
            " e.preprocessing_version, e.source_hmac, s.excerpt_hmac"
            " FROM embeddings e"
            " LEFT JOIN spans s ON s.span_id = e.span_id"
            f" WHERE e.encoder_id = ? AND e.span_id IN ({marks})"
        )
    else:
        sql = (
            "SELECT e.span_id, e.vector, e.dimensions, e.dtype,"
            " e.preprocessing_version, e.source_hmac, NULL"
            " FROM embeddings e"
            f" WHERE e.encoder_id = ? AND e.span_id IN ({marks})"
        )
    return conn.execute(sql, [encoder_id, *part]).fetchall()


def _input_rows(conn: sqlite3.Connection, part: Sequence[str],
                encoder_id: str) -> dict:
    """``(span_id, preprocessing_version) -> (dependency_digest, seq)``."""
    marks = ",".join("?" for _ in part)
    rows = conn.execute(
        "SELECT span_id, preprocessing_version, dependency_digest,"
        f" input_known_seq FROM embedding_inputs"
        f" WHERE encoder_id = ? AND span_id IN ({marks})",
        [encoder_id, *part],
    ).fetchall()
    return {
        (r[0], r[1]): (r[2], int(r[3]) if r[3] is not None else 0)
        for r in rows
    }


class _Desc(str):
    """str subclass with inverted ordering for deterministic heap ties.

    ``heapq`` is a min-heap: entries sort as ``(score, _Desc(span_id))``
    so the *largest* score pops last and equal scores keep ascending
    span-id order — matching ``cosine``'s ``(-score, span_id)`` output.
    """

    def __lt__(self, other: str) -> bool:
        return str.__gt__(self, other)

    def __le__(self, other: str) -> bool:
        return str.__ge__(self, other)


def search(
    conn: sqlite3.Connection,
    span_ids: Sequence[str],
    encoder_id: str,
    query_blob: bytes,
    top_k: int,
    *,
    batch_size: int = SCAN_BATCH,
    deadline: Optional[Any] = None,
    preprocessing_version: Optional[str] = None,
    known_at_seq: Optional[int] = None,
    sink: Optional[Callable[[str, float], None]] = None,
) -> tuple[list[tuple[str, float]], ScanCoverage]:
    """Exact cosine top-k over arbitrarily many authorized vectors.

    Streams ``batch_size`` rows per round and keeps a ``top_k``-bounded
    heap — memory stays O(batch_size + top_k) for any eligible-set size.
    Every eligible row is examined unless ``deadline.expired()`` fires;
    a cut scan returns the best of the examined prefix with
    ``coverage.partial`` set and ``examined < eligible``.

    Space/staleness guards (V4-29.05): rows are excluded — and counted —
    when their dtype/dimensions/preprocessing do not match the query's
    space, when ``source_hmac`` no longer matches the span's stored
    ``excerpt_hmac`` (embedded text changed), or when the
    ``embedding_inputs`` ledger records a differing dependency digest or
    an ``input_known_seq`` beyond ``known_at_seq``. The ledger is optional
    (v1 stores lack the table); ``excerpt_hmac`` is checked whenever the
    spans join resolves.

    ``sink(span_id, score)`` — when given — observes EVERY scored vector,
    not just heap survivors, so a caller can fold span scores into
    claim-level aggregates without losing rows outside the top-k.
    Returns ``(hits, coverage)`` where ``hits`` are ``(span_id, score)``
    sorted by score desc, span_id asc.
    """
    if top_k < 1:
        raise VerbatimError(ErrorCode.VALIDATION, "top_k must be >= 1")
    ids = list(dict.fromkeys(span_ids))
    eligible = len(ids)
    empty_cov = ScanCoverage(eligible=eligible)
    if not query_blob or len(query_blob) % 4:
        return [], empty_cov
    dims = len(query_blob) // 4
    if dims < 1 or dims > MAX_DIMENSIONS:
        return [], ScanCoverage(eligible=eligible, excluded_malformed=eligible)
    query = Float32Codec.unpack(query_blob, dims)

    np = _numpy()
    if np is not None:
        q = np.asarray(query, dtype=np.float64)
        q_norm = float(np.linalg.norm(q))
    else:
        q_norm_sq = math.fsum(x * x for x in query)
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0.0 else 0.0
    if q_norm == 0.0 or not math.isfinite(q_norm):
        return [], empty_cov

    join_spans = _table_exists(conn, "spans")
    has_inputs = _table_exists(conn, "embedding_inputs")

    heap: list = []
    examined = scored = missing = 0
    stale = generation = malformed = 0
    partial = False

    def _score(vec: array) -> Optional[float]:
        if np is not None:
            v = np.asarray(vec, dtype=np.float64)
            vn = float(np.linalg.norm(v))
            if vn == 0.0 or not math.isfinite(vn):
                return None
            return float(np.dot(q, v) / (q_norm * vn))
        v_norm_sq = math.fsum(x * x for x in vec)
        if v_norm_sq <= 0.0 or not math.isfinite(v_norm_sq):
            return None
        dot = math.fsum(a * b for a, b in zip(query, vec))
        return dot / (q_norm * math.sqrt(v_norm_sq))

    for i in range(0, eligible, batch_size):
        if deadline is not None and deadline.expired():
            partial = True
            break
        part = ids[i : i + batch_size]
        rows = _batch_rows(conn, part, encoder_id, join_spans)
        inputs = _input_rows(conn, part, encoder_id) if has_inputs else {}
        seen = set()
        for (sid, blob, d, dtype, ppv, src_hmac,
             excerpt_hmac) in rows:
            seen.add(sid)
            # Vector-space identity guards — mismatched rows answer a
            # different embedding space and must never score (V4-29.05).
            if dtype != Float32Codec.dtype:
                generation += 1
                continue
            if not isinstance(d, int) or d < 1 or d > MAX_DIMENSIONS:
                malformed += 1
                continue
            if d != dims or (
                preprocessing_version is not None
                and ppv != preprocessing_version
            ):
                generation += 1
                continue
            if (not isinstance(blob, (bytes, bytearray))
                    or len(blob) != d * 4):
                malformed += 1
                continue
            # Staleness: the embedded text's digest must still equal the
            # span's persisted excerpt digest (both are store-keyed HMACs
            # of the same byte slice when fresh).
            if (excerpt_hmac is not None
                    and bytes(excerpt_hmac) != bytes(src_hmac)):
                stale += 1
                continue
            led = inputs.get((sid, ppv))
            if led is not None:
                dep_digest, input_seq = led
                if dep_digest is not None and (
                    bytes(dep_digest) != bytes(src_hmac)
                ):
                    stale += 1
                    continue
                if (known_at_seq is not None
                        and input_seq > known_at_seq):
                    generation += 1
                    continue
            vec = _decode(bytes(blob), d)
            value = _score(vec)
            if value is None:
                malformed += 1
                continue
            scored += 1
            if sink is not None:
                sink(sid, value)
            entry = (value, _Desc(sid), sid)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
        examined += len(part)
        missing += len(part) - len(seen)

    # Deterministic order: score desc, span_id asc (V4-28.10); the heap
    # entry's middle _Desc element is internal and never escapes.
    hits = [
        (sid, score)
        for score, _key, sid in sorted(heap, key=lambda e: (-e[0], e[2]))
    ]
    cov = ScanCoverage(
        eligible=eligible, examined=examined, scored=scored,
        returned=len(hits), partial=partial,
        excluded_stale=stale, excluded_generation=generation,
        excluded_malformed=malformed, missing=missing,
    )
    return hits, cov


def cosine(query_blob: bytes, matrix: EmbeddingMatrix, top_k: int) -> list[tuple[str, float]]:
    """Exact cosine similarity over the authorized matrix.

    ``query_blob`` is a packed float32le query vector (same encoder identity
    as the matrix — enforced by the caller's ``encoder_id`` selection).
    Returns ``[(span_id, score), ...]`` sorted by score desc, ties broken by
    span_id for deterministic output. Zero-norm queries and rows are
    excluded: cosine is undefined on a directionless vector.
    """
    if top_k < 1:
        raise VerbatimError(ErrorCode.VALIDATION, "top_k must be >= 1")
    ids, vectors = matrix
    if not vectors:
        return []
    # The query's own length is the reference dimension — a corrupt
    # dimension-mismatched row must never reframe the comparison space.
    dims = len(query_blob) // 4
    query = Float32Codec.unpack(query_blob, dims)

    np = _numpy()
    if np is not None:
        q = np.asarray(query, dtype=np.float64)
        qn = float(np.linalg.norm(q))
        if qn == 0.0 or not math.isfinite(qn):
            return []
        scores: list[tuple[str, float]] = []
        for sid, vec in zip(ids, vectors):
            if len(vec) != dims:
                continue  # dimension-mismatched row — corrupt/foreign generation
            # np.asarray on the array object reads values, not raw bytes —
            # no endianness assumption on any host.
            v = np.asarray(vec, dtype=np.float64)
            vn = float(np.linalg.norm(v))
            if vn == 0.0 or not math.isfinite(vn):
                continue
            scores.append((sid, float(np.dot(q, v) / (qn * vn))))
    else:
        q_norm_sq = math.fsum(x * x for x in query)
        if q_norm_sq <= 0.0 or not math.isfinite(q_norm_sq):
            return []
        qn = math.sqrt(q_norm_sq)
        scores = []
        for sid, vec in zip(ids, vectors):
            if len(vec) != dims:
                continue
            v_norm_sq = math.fsum(x * x for x in vec)
            if v_norm_sq <= 0.0 or not math.isfinite(v_norm_sq):
                continue
            dot = math.fsum(a * b for a, b in zip(query, vec))
            scores.append((sid, dot / (qn * math.sqrt(v_norm_sq))))
    scores.sort(key=lambda kv: (-kv[1], kv[0]))
    return scores[:top_k]


def coverage(conn: sqlite3.Connection, encoder_id: str) -> tuple[int, int]:
    """``(embedded_spans, total_spans)`` for capability reporting.

    SPEC §28: partial embedding coverage is reported explicitly — unembedded
    spans stay reachable through lexical/entity retrieval and the gap is
    visible in diagnostics rather than hidden.
    """
    embedded = conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE encoder_id = ?", (encoder_id,)
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    return int(embedded), int(total)


__all__ = [
    "EmbeddingMatrix",
    "MAX_SPAN_IDS",
    "MAX_DIMENSIONS",
    "SCAN_BATCH",
    "ScanCoverage",
    "load_matrix",
    "search",
    "cosine",
    "coverage",
]
