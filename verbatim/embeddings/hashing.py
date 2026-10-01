"""Deterministic local subword encoder (``embedding.backend = "hashing"``).

A pure-stdlib feature-hashing encoder: word unigrams plus bounded
character n-grams are hashed into a fixed-width float32 vector with
blake2b and L2-normalized. No model artifact, no network, no native
dependency — the algorithm revision is the pin (``encoder_id``
``hashing:<model>:<revision>`` changes whenever the featurizer changes,
so stored vectors can never silently mix generations, SPEC §28).

This is a *lexical-subword* encoder: it encodes real morphological
similarity (shared stems, prefixes, identifiers) and is honest about
it — the manifest declares ``kind: lexical-subword``. It is not a
neural semantic model; paraphrase-level semantics stay limited and the
evaluation reports what it actually measures (V3-54.11).

Determinism: hashing uses blake2b with a fixed personalization string —
never ``hash()``, which is salted per process and would make stored
vectors irreproducible.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Optional

from ..core.types import ErrorCode, VerbatimError
from .codec import Float32Codec
from .encoder import encoder_identity

#: Vector width. Matches the small-encoder convention (nomic-embed-class
#: models) so a later reviewed artifact backend can be compared on equal
#: dimensionality terms.
_DIMS = 384

#: Character n-gram sizes, word-bounded. n=3..5 captures stems and
#: affixes (``swagger-gen`` ↔ ``swagger``); n<3 is noise, n>5 is rare.
_NGRAM_MIN = 3
_NGRAM_MAX = 5

#: Feature weights. Whole-word identity outweighs subword overlap.
_WORD_W = 1.0
_NGRAM_W = 0.6

#: Domain separation for the hash — bumping this is a featurizer change
#: and therefore a new ``preprocessing_version``/artifact revision.
_HASH_PERSON = b"verbatim-h1"

#: Algorithm revision — the artifact pin for a code-defined encoder.
_ALGO_REVISION = "v1"

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'/.-]*")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _features(text: str) -> list[tuple[str, float]]:
    """(feature, weight) multiset: word unigrams + bounded char n-grams."""
    feats: list[tuple[str, float]] = []
    for tok in _tokens(text):
        feats.append((f"w:{tok}", _WORD_W))
        padded = f"<{tok}>"
        for n in range(_NGRAM_MIN, _NGRAM_MAX + 1):
            if len(padded) < n:
                continue
            for i in range(len(padded) - n + 1):
                feats.append((f"c:{padded[i:i + n]}", _NGRAM_W))
    return feats


def _bucket_sign(feat: str) -> tuple[int, float]:
    """Stable (bucket, ±1.0) for a feature string via blake2b-128."""
    d = hashlib.blake2b(
        feat.encode("utf-8"), digest_size=16, person=_HASH_PERSON
    ).digest()
    bucket = int.from_bytes(d[:8], "little") % _DIMS
    sign = 1.0 if int.from_bytes(d[8:], "little") & 1 else -1.0
    return bucket, sign


def _embed(text: str) -> list[float]:
    """Unnormalized hashed vector for one text."""
    vec = [0.0] * _DIMS
    counts: dict[str, float] = {}
    for feat, w in _features(text):
        counts[feat] = counts.get(feat, 0.0) + w
    for feat, w in counts.items():
        bucket, sign = _bucket_sign(feat)
        vec[bucket] += sign * math.log1p(w)
    return vec


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class HashingEncoder:
    """``Encoder`` protocol implementation for the ``hashing`` backend.

    Fully local and deterministic — ``available()`` is always True.
    Construction validates only that the backend got a coherent config;
    there is no runtime to probe and nothing to warm up.
    """

    #: The featurizer identity is fixed by this module's code — a config
    #: ``model`` string names neural artifacts, and letting it flow into
    #: the identity would mislabel stored vectors (``nomic-embed-text``
    #: vectors these are not).
    _MODEL = "subword-ngram"

    def __init__(self, cfg: Any) -> None:
        self._revision = (
            getattr(cfg, "artifact_revision", None) or _ALGO_REVISION
        )

    @property
    def encoder_id(self) -> str:
        return encoder_identity("hashing", self._MODEL, self._revision)

    @property
    def dimensions(self) -> int:
        return _DIMS

    @property
    def normalization(self) -> Optional[str]:
        # Vectors are emitted already L2-normalized.
        return "none"

    def available(self) -> bool:
        return True

    def encode(self, texts: list[str]) -> list[bytes]:
        if not isinstance(texts, list) or not all(
            isinstance(t, str) for t in texts
        ):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "encode expects a list of strings",
            )
        return [
            Float32Codec.pack(_l2_normalize(_embed(t))) for t in texts
        ]

    def manifest(self) -> dict[str, Any]:
        return {
            "artifact_revision": self._revision,
            "dimensions": _DIMS,
            "normalization": "none",
            "license_id": None,
            "preprocessing_version": _ALGO_REVISION,
            "manifest_json": {
                "kind": "lexical-subword",
                "featurizer": (
                    "word unigrams + bounded char n-grams "
                    f"{_NGRAM_MIN}..{_NGRAM_MAX}"
                ),
                "hash": "blake2b-128",
                "dimensions": _DIMS,
                "weights": {"word": _WORD_W, "ngram": _NGRAM_W},
                "note": (
                    "deterministic feature-hashing encoder — not a "
                    "neural semantic model; measured recall reflects "
                    "subword/lexical similarity only"
                ),
            },
        }


__all__ = ["HashingEncoder"]
