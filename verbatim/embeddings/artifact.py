"""Pinned local artifact encoder — parse-only VBV1 loader (SPEC_V2 §38,
SPEC_V6 V6-03.06/03.08, docs/v6_contracts.md §6).

The honest V6 "neural" path is a *locally built, hash-pinned, parse-only*
artifact — no downloads, no pickle, no ``eval``, no ``torch.load``. The
artifact is a word-vector table produced by a recorded offline builder
(``verbatim.embeddings.artifact_build`` — e.g. the distilled-hashing dev
table in ``eval/v6/neural.py``), laid out as::

    <data_dir>/artifact_manifest.json          (shipped-package pin)
    <data_dir>/models/artifact_manifest.json   (build_artifact output)
    <data_dir>/models/<model>/<revision>/vectors.bin

``vectors.bin`` is the VBV1 format::

    magic b"VBV1" | u32 dim | u64 count |
    count records sorted by key (UTF-8 byte order):
        u16 key_len | key bytes (UTF-8) | dim x float32le

Trust contract (unchanged — now enforced, not aspirational):

1. PINNED MANIFEST. ``artifact_manifest.json`` declares, per
   ``models.<model>.<revision>``: file list with sha256+size, format id,
   dim, and the loader source-hash that was reviewed.
   ``EmbeddingConfig.artifact_revision`` must match an entry exactly;
   ``latest`` (and an empty revision) is rejected.
2. OFFLINE-ONLY LOAD. Files must already exist under the data dir. The
   loader performs no network I/O — the path is unreachable, not denied.
3. HASH VERIFICATION AT LOAD. Every manifest-declared file is re-hashed
   before the artifact verifies; any mismatch → ``available()`` False,
   never a substitute model, never an exception at probe time.
4. REVIEWED LOADER ONLY. This module's strict ``struct`` parser is the
   whole loader — a declared ``loader_source_sha256`` that does not match
   this file's bytes fails verification.
5. OUTPUT CONTRACT. Returned vectors are float32le blobs validated
   through :class:`Float32Codec` — same acceptance envelope as every
   backend: count, dims, finite values, nonzero norm.
6. CAPABILITY HONESTY. ``available()`` re-runs the full checklist on
   every call (a tampered file flips a previously-good probe back to
   False); a missing/corrupt artifact is a capability downgrade —
   ``encode`` raises ``ENCODER_UNAVAILABLE`` — never a silent fallback.

Encode semantics — embedding = mean of table vectors for known keys
(skip-OOV): a text is tokenized with the *same* normalizer the dev table
was keyed on (``hashing._tokens``); each known token contributes its row;
OOV tokens are skipped rather than synthesized, so their mass never
hallucinates similarity. A text with **zero** known tokens cannot take a
mean — it receives a deterministic pseudo-random unit vector derived
from the text bytes (blake2b counter-mode), so two *different* OOV texts
are near-orthogonal instead of falsely identical, while the *same* OOV
text still reproduces its own vector bit-for-bit (determinism the
storage contract requires). All outputs are L2-normalized.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import struct
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from ..config import EmbeddingConfig
from ..core.types import VerbatimError, ErrorCode
from .codec import Float32Codec
from .encoder import encoder_identity
from .hashing import _tokens as _normalize_tokens

_log = logging.getLogger(__name__)

MANIFEST_NAME = "artifact_manifest.json"
VECTORS_NAME = "vectors.bin"
MAGIC = b"VBV1"
FORMAT_ID = "VBV1"
_HEADER = struct.Struct("<4sIQ")
_KEY_LEN = struct.Struct("<H")
_SUPPORTED_FORMATS = frozenset({FORMAT_ID})
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_VECTORS_BYTES = 1 << 28          # 256 MiB — word tables stay far below
_MAX_DIM = 65536
_MAX_KEY_BYTES = 4096
_MAX_COUNT = 1 << 24                  # 16M rows sanity bound
_OOV_PERSON = b"vbv1-oov"
#: Manifest-declared filenames — plain basenames only (vectors.bin,
#: tokenizer.json, model.safetensors); no separators, drives, or dots.
_FNAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
#: Engine-side preprocessing identity: hashing-token normalizer v1.
_PREPROCESSING_VERSION = "vbv1-tok1"


class _ArtifactInvalid(Exception):
    """Internal: artifact bytes failed the strict parse/verify."""


def _l2_normalize(vec: list) -> list:
    norm = math.sqrt(math.fsum(x * x for x in vec))
    if norm <= 0.0 or not math.isfinite(norm):
        return [1.0] + [0.0] * (len(vec) - 1)
    return [x / norm for x in vec]


def _oov_vector(text: str, dim: int) -> list:
    """Deterministic unit vector for a text with no known tokens.

    blake2b counter-mode over the text bytes — same input → same vector
    on every platform and process (no ``hash()``, no PRNG seeding
    ambiguities); distinct inputs are near-orthogonal in expectation.
    """
    out: list = []
    counter = 0
    data = text.encode("utf-8")
    while len(out) < dim:
        block = hashlib.blake2b(
            data,
            digest_size=64,
            person=_OOV_PERSON,
            salt=counter.to_bytes(8, "little"),
        ).digest()
        out.extend((b - 127.5) / 127.5 for b in block)
        counter += 1
    return _l2_normalize(out[:dim])


def _read_file_bytes(path: Path) -> bytes:
    try:
        if path.stat().st_size > _MAX_VECTORS_BYTES:
            raise _ArtifactInvalid("vectors.bin exceeds sanity bound")
        return path.read_bytes()
    except OSError as exc:
        raise _ArtifactInvalid(f"vectors.bin unreadable: {exc}") from exc


def _load_vectors(path: Path, expected_dim: int) -> Dict[str, Tuple[float, ...]]:
    """Strictly parse a VBV1 payload into ``{key: vector}``.

    Every deviation raises :class:`_ArtifactInvalid` — bad magic, short
    file, declared-dim mismatch, truncated record, undecodable key,
    unsorted/duplicate keys, non-finite components, or trailing bytes.
    Callers translate the failure (``available()`` False /
    ``ENCODER_UNAVAILABLE``); the store never sees a partial table.
    """
    return _parse_vectors(_read_file_bytes(path), expected_dim)


def _parse_vectors(data: bytes, expected_dim: int) -> Dict[str, Tuple[float, ...]]:
    """The strict VBV1 parser — see :func:`_load_vectors`."""
    if len(data) < _HEADER.size:
        raise _ArtifactInvalid("vectors.bin shorter than VBV1 header")
    magic, dim, count = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise _ArtifactInvalid("bad VBV1 magic")
    if not (1 <= dim <= _MAX_DIM):
        raise _ArtifactInvalid(f"VBV1 dim {dim} out of bounds")
    if dim != expected_dim:
        raise _ArtifactInvalid(
            f"VBV1 dim {dim} != manifest dim {expected_dim}"
        )
    if count > _MAX_COUNT:
        raise _ArtifactInvalid(f"VBV1 count {count} out of bounds")

    row_floats = struct.Struct(f"<{dim}f")
    off = _HEADER.size
    table: Dict[str, Tuple[float, ...]] = {}
    prev_key: Optional[bytes] = None
    for _ in range(count):
        if off + _KEY_LEN.size > len(data):
            raise _ArtifactInvalid("truncated VBV1 record (key length)")
        (klen,) = _KEY_LEN.unpack_from(data, off)
        off += _KEY_LEN.size
        if not (1 <= klen <= _MAX_KEY_BYTES):
            raise _ArtifactInvalid(f"VBV1 key length {klen} out of bounds")
        if off + klen + row_floats.size > len(data):
            raise _ArtifactInvalid("truncated VBV1 record (key/row)")
        kb = data[off:off + klen]
        off += klen
        try:
            key = kb.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise _ArtifactInvalid("VBV1 key is not valid UTF-8") from exc
        if prev_key is not None and kb <= prev_key:
            raise _ArtifactInvalid(
                "VBV1 records out of key order or duplicated"
            )
        prev_key = kb
        row = row_floats.unpack_from(data, off)
        off += row_floats.size
        for x in row:
            if not math.isfinite(x):
                raise _ArtifactInvalid(
                    f"VBV1 row {key!r} has a non-finite component"
                )
        table[key] = row
    if off != len(data):
        raise _ArtifactInvalid("trailing bytes after last VBV1 record")
    return table


class ArtifactEncoder:
    """Fail-closed pinned-artifact encoder (VBV1 word tables).

    Construction never raises — the store must open and report a
    degraded capability instead of dying (SPEC_V2 §8). ``available()``
    re-runs the full verification each call: pinned revision → manifest
    loads → ``models.<model>.<revision>`` entry → file presence →
    sha256/size match → supported format → positive dim → loader-hash
    pin → strict VBV1 parse. Only then does ``encode`` serve.
    """

    def __init__(self, cfg: EmbeddingConfig, *, data_dir: Optional[Path] = None,
                 manifest: Optional[Mapping[str, Any]] = None):
        self._cfg = cfg
        self._data_dir = Path(data_dir) if data_dir is not None else None
        self._manifest = dict(manifest) if manifest else None
        self._model = str(getattr(cfg, "model", "") or "").strip()
        self._revision = str(
            getattr(cfg, "artifact_revision", "") or ""
        ).strip()
        self._table: Optional[Dict[str, Tuple[float, ...]]] = None
        self._dim: int = 0
        self._entry: Optional[Mapping[str, Any]] = None

    # ---- identity ------------------------------------------------------

    @property
    def name(self) -> str:
        return encoder_identity("artifact", self._model, self._revision)

    @property
    def encoder_id(self) -> str:
        return self.name

    @property
    def dimensions(self) -> int:
        if not self._artifact_verified():
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE,
                "artifact encoder dimensions unknown: no verified "
                "artifact loaded",
            )
        return self._dim

    @property
    def normalization(self) -> Optional[str]:
        # encode() emits L2-normalized vectors.
        return "none"

    def encoder_identity(self) -> str:
        return self.name

    # ---- verification ---------------------------------------------------

    def available(self) -> bool:
        """True only while a pinned, hash-verified artifact loads.

        Re-verifies on every call — post-probe tampering flips the next
        probe back to False (contract item 6). Never raises.
        """
        try:
            return self._artifact_verified()
        except Exception:  # noqa: BLE001 — a probe never propagates
            _log.debug("artifact verification failed", exc_info=True)
            return False

    def _artifact_verified(self) -> bool:
        """The verification checklist — every gate must pass.

        On success the parsed table is cached for ``encode``; on any
        failure the cache is cleared so a previously-good artifact that
        was tampered cannot serve stale vectors.
        """
        ok = False
        try:
            ok = self._verify()
        except Exception:  # noqa: BLE001 — verification stays fail-closed
            _log.debug("artifact verification error", exc_info=True)
            ok = False
        if not ok:
            self._table = None
            self._dim = 0
            self._entry = None
        return ok

    def _verify(self) -> bool:
        revision = self._revision.strip()
        if not revision or revision.lower() == "latest":
            return False
        manifest = self._load_manifest()
        if manifest is None:
            return False
        models = manifest.get("models")
        if not isinstance(models, dict):
            return False
        entry = (models.get(self._model) or {}).get(revision)
        if not isinstance(entry, dict):
            return False
        model_dir = self._model_dir()
        if model_dir is None or not model_dir.is_dir():
            return False
        files = entry.get("files")
        if not isinstance(files, dict) or not files:
            return False
        vectors_bytes: Optional[bytes] = None
        for fname, meta in files.items():
            if not self._file_safe(fname) or not isinstance(meta, dict):
                return False
            want_size = meta.get("size")
            want_sha = meta.get("sha256")
            if (
                not isinstance(want_size, int)
                or isinstance(want_size, bool)
                or want_size < 0
                or not isinstance(want_sha, str)
                or len(want_sha) != 64
            ):
                return False
            fpath = model_dir / fname
            try:
                if not fpath.is_file():
                    return False
                if fname == VECTORS_NAME:
                    # Hash AND parse the same bytes — never re-read the
                    # path (a swapped file between checks would serve
                    # unverified content).
                    vectors_bytes = _read_file_bytes(fpath)
                    if len(vectors_bytes) != want_size:
                        return False
                    if hashlib.sha256(vectors_bytes).hexdigest() != (
                        want_sha.lower()
                    ):
                        return False
                else:
                    if fpath.stat().st_size != want_size:
                        return False
                    if self._sha256_file(fpath) != want_sha.lower():
                        return False
            except (_ArtifactInvalid, OSError):
                return False
        if entry.get("format") not in _SUPPORTED_FORMATS:
            return False
        dim = entry.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            return False
        loader_pin = entry.get("loader_source_sha256")
        if loader_pin is not None and (
            not isinstance(loader_pin, str)
            or loader_pin.lower() != self._loader_source_sha256()
        ):
            return False
        if vectors_bytes is None:
            return False
        # Format gate: the pinned payload must actually parse — a
        # manifest that correctly hashes corrupt bytes still fails.
        self._table = _parse_vectors(vectors_bytes, dim)
        self._dim = dim
        self._entry = entry
        return True

    @staticmethod
    def _file_safe(name: Any) -> bool:
        # Manifest-declared filenames resolve under model_dir — allowlist
        # to plain basenames so a hostile entry can never escape it.
        return isinstance(name, str) and bool(_FNAME_RE.match(name))

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _loader_source_sha256() -> str:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def _manifest_paths(self) -> Tuple[Path, ...]:
        if self._data_dir is None:
            return ()
        return (
            self._data_dir / MANIFEST_NAME,
            self._data_dir / "models" / MANIFEST_NAME,
        )

    def _load_manifest(self) -> Optional[Mapping[str, Any]]:
        if self._manifest is not None:
            return self._manifest
        for path in self._manifest_paths():
            try:
                if not path.is_file() or (
                    path.stat().st_size > _MAX_MANIFEST_BYTES
                ):
                    continue
                data = json.loads(path.read_text("utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError):
                continue
        return None

    def _model_dir(self) -> Optional[Path]:
        if self._data_dir is None or not self._model or not self._revision:
            return None
        # Fixed layout: <data>/models/<model>/<revision>/ — no traversal.
        if any(part in ("", ".", "..") or "/" in part or "\\" in part
               for part in (self._model, self._revision)):
            return None
        return self._data_dir / "models" / self._model / self._revision

    # ---- encode ---------------------------------------------------------

    def _embed_text(self, text: str) -> list:
        """Mean of known-token rows, L2-normalized; deterministic OOV
        unit vector when no token is known (skip-OOV contract)."""
        assert self._table is not None
        seen: set = set()
        acc: Optional[list] = None
        n = 0
        for tok in _normalize_tokens(text):
            if tok in seen:
                continue
            seen.add(tok)
            row = self._table.get(tok)
            if row is None:
                continue
            n += 1
            if acc is None:
                acc = list(row)
            else:
                for i, x in enumerate(row):
                    acc[i] += x
        if acc is None:
            return _oov_vector(text, self._dim)
        return _l2_normalize([x / n for x in acc])

    def encode(self, texts):
        if not isinstance(texts, list) or not all(
            isinstance(t, str) for t in texts
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "encode expects a list of strings",
            )
        if not self._artifact_verified():
            raise VerbatimError(
                ErrorCode.ENCODER_UNAVAILABLE,
                "artifact encoder unavailable: no verified pinned "
                "artifact (manifest, sha256, format, and parse checks "
                "must all pass — SPEC_V2 §38/V6-03.08 fail-closed)",
            )
        assert self._table is not None and self._dim > 0
        blobs = [
            Float32Codec.pack(self._embed_text(t)) for t in texts
        ]
        # Output gate: validate the exact bytes that would persist.
        for blob in blobs:
            Float32Codec.validate_blob(blob, self._dim)
        return blobs

    def manifest(self) -> dict:
        verified = self._artifact_verified()
        entry = self._entry if verified else {}
        return {
            "backend": "artifact",
            "model": self._model,
            "artifact_revision": self._revision or "unpinned",
            "dimensions": self._dim if verified else None,
            "normalization": "none",
            "license_id": None,
            "preprocessing_version": _PREPROCESSING_VERSION,
            "status": "verified" if verified else "unavailable",
            "manifest_json": {
                "kind": "word-vector-table",
                "format": (entry or {}).get("format"),
                "count": (entry or {}).get("count"),
                "oov_policy": (
                    "skip-OOV mean of known-token vectors; texts with no "
                    "known token get a deterministic blake2b-derived unit "
                    "vector"
                ),
                "tokenizer": "verbatim.embeddings.hashing._tokens",
                "note": (
                    "locally built hash-pinned parse-only artifact; the "
                    "dev build is distilled hashing (eval/v6/neural.py) — "
                    "a real deterministic table, honestly labeled"
                ),
            },
        }

    def manifest_data(self) -> Mapping[str, Any]:
        return self.manifest()


def get_artifact_encoder(cfg: EmbeddingConfig, *,
                         data_dir: Optional[Path] = None) -> ArtifactEncoder:
    return ArtifactEncoder(cfg, data_dir=data_dir)


__all__ = [
    "ArtifactEncoder",
    "FORMAT_ID",
    "MAGIC",
    "MANIFEST_NAME",
    "VECTORS_NAME",
    "get_artifact_encoder",
]
