"""Little-endian float32 vector wire codec and acceptance validation.

Vectors persist as packed little-endian float32 BLOBs (SPEC §28). The codec
is stdlib-only: ``struct`` handles byte order explicitly, so the format is
identical on every platform — ``array('f')`` alone would silently write
native-endian bytes on a big-endian host. NumPy is used opportunistically
when installed (the optional ``semantic`` extra) but is never required and
is imported lazily so importing this module has no heavy cost.

Validation is the SPEC §28 acceptance gate: a stored vector must have the
declared length, finite components, and a nonzero finite L2 norm. Zero-norm,
wrong-length, NaN, and infinite vectors are rejected with a visible
embedding failure (``VECTOR_INVALID``) rather than being stored and later
poisoning cosine search.
"""

from __future__ import annotations

import math
import struct
from typing import Optional, Sequence

from ..core.types import ErrorCode, VerbatimError

_F32_BYTES = 4

# Tri-state lazy numpy probe: None = not attempted yet.
_NP: Optional[object] = None
_NP_TRIED = False


def _numpy() -> Optional[object]:
    """Return numpy if importable, else None. Probed once, lazily.

    Lazy so the embeddings package imports cleanly on minimal installs —
    ``numpy`` lives in the optional ``semantic`` extra, and an import-time
    hard dependency would break the default ``offline_rules`` install.
    """
    global _NP, _NP_TRIED
    if _NP_TRIED:
        return _NP  # type: ignore[return-value]
    _NP_TRIED = True
    try:
        import numpy as np  # type: ignore
    except ImportError:
        _NP = None
    else:
        _NP = np
    return _NP  # type: ignore[return-value]


class Float32Codec:
    """Pack/unpack/validate little-endian float32 vectors (SPEC §28)."""

    dtype = "float32le"

    @staticmethod
    def pack(vector: Sequence[float]) -> bytes:
        """Serialize floats to little-endian float32 bytes.

        ``struct`` with an explicit ``<`` prefix is byte-order-stable across
        platforms — unlike ``array('f').tobytes()``, which is native-endian.
        """
        n = len(vector)
        if n == 0:
            raise VerbatimError(ErrorCode.VECTOR_INVALID, "cannot pack an empty vector")
        np = _numpy()
        if np is not None:
            try:
                return np.asarray(vector, dtype="<f4").tobytes()  # type: ignore[union-attr]
            except (TypeError, ValueError, OverflowError) as exc:
                raise VerbatimError(
                    ErrorCode.VECTOR_INVALID,
                    f"vector contains non-numeric component: {exc}",
                ) from exc
        try:
            return struct.pack(f"<{n}f", *vector)
        except (struct.error, TypeError, OverflowError) as exc:
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID, f"vector contains non-numeric component: {exc}"
            ) from exc

    @staticmethod
    def unpack(blob: bytes, dimensions: int) -> tuple[float, ...]:
        """Deserialize little-endian float32 bytes into a float tuple.

        ``dimensions`` is the declared dimension; a blob whose length does
        not equal ``dimensions * 4`` is a stored-corruption or identity
        mismatch and is rejected rather than truncated.
        """
        if dimensions < 1:
            raise VerbatimError(ErrorCode.VECTOR_INVALID, f"invalid dimensions {dimensions}")
        expected = dimensions * _F32_BYTES
        if not isinstance(blob, (bytes, bytearray, memoryview)) or len(blob) != expected:
            actual = len(blob) if isinstance(blob, (bytes, bytearray, memoryview)) else "non-bytes"
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID,
                f"vector blob length {actual} != {expected} for {dimensions} dims",
            )
        np = _numpy()
        if np is not None:
            arr = np.frombuffer(bytes(blob), dtype="<f4")  # type: ignore[union-attr]
            return tuple(float(x) for x in arr)
        return struct.unpack(f"<{dimensions}f", bytes(blob))

    @staticmethod
    def validate(vector: Sequence[float], dimensions: int) -> tuple[float, ...]:
        """Acceptance gate for encoder output (SPEC §28).

        Returns the vector as a plain float tuple when valid. Rejects, with
        ``VECTOR_INVALID``:

        - wrong length — stored vectors must match the declared dimension
          so query/document vectors stay comparable;
        - non-finite components (NaN, ±inf) — they propagate through cosine
          math as NaN scores and corrupt ranking silently;
        - zero-norm or non-finite norm — a zero vector has no direction, so
          cosine similarity is undefined; a non-finite norm means squared
          components overflowed to inf even though each component was finite.
        """
        if len(vector) != dimensions:
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID,
                f"vector length {len(vector)} != declared dimensions {dimensions}",
            )
        try:
            out: tuple[float, ...] = tuple(float(x) for x in vector)
        except (TypeError, ValueError) as exc:
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID, f"vector component not numeric: {exc}"
            ) from exc
        for x in out:
            if not math.isfinite(x):
                raise VerbatimError(
                    ErrorCode.VECTOR_INVALID, "vector contains NaN or infinite component"
                )
        norm_sq = math.fsum(x * x for x in out)
        if not math.isfinite(norm_sq) or norm_sq <= 0.0:
            raise VerbatimError(
                ErrorCode.VECTOR_INVALID, "vector has zero or non-finite L2 norm"
            )
        return out

    @staticmethod
    def validate_blob(blob: bytes, dimensions: int) -> tuple[float, ...]:
        """Unpack then validate — the gate applied to encoder-returned bytes."""
        return Float32Codec.validate(Float32Codec.unpack(blob, dimensions), dimensions)
