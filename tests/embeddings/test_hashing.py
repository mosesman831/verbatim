"""HashingEncoder — the deterministic local subword backend.

Covers the Encoder protocol contract plus the properties the semantic
lane depends on: determinism across calls, L2-normalized output,
encoder-identity stability, and a real (bounded) similarity signal —
shared vocabulary/subwords score above unrelated text.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.config import EmbeddingConfig
from verbatim.core.types import VerbatimError
from verbatim.embeddings import Float32Codec, get_encoder
from verbatim.embeddings.encoder import Encoder
from verbatim.embeddings.hashing import HashingEncoder


def _enc(**kw) -> HashingEncoder:
    return HashingEncoder(EmbeddingConfig(backend="hashing", **kw))


def _decode(blob: bytes, dims: int) -> tuple[float, ...]:
    return Float32Codec.unpack(blob, dims)


def _cos(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return num / (da * db) if da and db else 0.0


class TestProtocol:
    def test_conforms_to_encoder_protocol(self):
        assert isinstance(_enc(), Encoder)

    def test_always_available(self):
        assert _enc().available() is True

    def test_dimensions_positive(self):
        assert _enc().dimensions == 384

    def test_identity_stable_and_pinned(self):
        e = _enc()
        assert e.encoder_id == "hashing:subword-ngram:v1"
        assert e.manifest()["preprocessing_version"] == "v1"

    def test_manifest_honest_kind(self):
        m = _enc().manifest()
        assert m["dimensions"] == 384
        assert m["normalization"] == "none"
        assert m["manifest_json"]["kind"] == "lexical-subword"


class TestEncoding:
    def test_deterministic_across_calls(self):
        e = _enc()
        assert e.encode(["regenerate the api types"]) == e.encode(
            ["regenerate the api types"]
        )

    def test_output_is_l2_normalized(self):
        (blob,) = _enc().encode(["make codegen builds the types"])
        vec = _decode(blob, 384)
        assert math.isclose(
            math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-5
        )

    def test_empty_text_is_zero_vector_not_nan(self):
        (blob,) = _enc().encode(["!!!"])
        vec = _decode(blob, 384)
        assert all(x == 0.0 for x in vec)

    def test_batch_order_preserved(self):
        blobs = _enc().encode(["alpha beta", "gamma delta", "alpha beta"])
        assert len(blobs) == 3
        assert blobs[0] == blobs[2] != blobs[1]

    def test_input_validation(self):
        with pytest.raises(VerbatimError):
            _enc().encode("not-a-list")  # type: ignore[arg-type]
        with pytest.raises(VerbatimError):
            _enc().encode(["ok", 42])  # type: ignore[list-item]

    def test_subword_similarity_signal(self):
        """Shared stems/identifiers score above unrelated text — the real
        (bounded) lexical-subword signal this backend honestly provides."""
        e = _enc()
        blobs = e.encode([
            "regenerate the api types with make codegen",
            "run codegen to regenerate api types",      # paraphrase-ish
            "install the certificate on the proxy host",  # unrelated
        ])
        q, rel, unrel = (_decode(b, 384) for b in blobs)
        assert _cos(q, rel) > _cos(q, unrel) >= 0.0

    def test_identifier_overlap_signal(self):
        """Exact identifiers survive hashing — the dominant recall case."""
        e = _enc()
        blobs = e.encode([
            "which make target builds api types",
            "use make swagger-gen for api types",
            "database backups run nightly via cron",
        ])
        q, hit, miss = (_decode(b, 384) for b in blobs)
        assert _cos(q, hit) > _cos(q, miss)


class TestFactory:
    def test_get_encoder_resolves_hashing(self):
        cfg = EmbeddingConfig(backend="hashing")
        enc = get_encoder(type("C", (), {"embedding": cfg})())
        assert isinstance(enc, HashingEncoder)

    def test_neural_model_default_does_not_mislabel(self):
        """cfg.model defaults to a neural artifact name; the hashing
        featurizer identity is code-fixed so stored vectors can never
        be mislabeled as nomic-embed-text output."""
        enc = _enc()  # EmbeddingConfig.model defaults to nomic-embed-text
        assert enc.encoder_id == "hashing:subword-ngram:v1"
        enc2 = _enc(model="nomic-embed-text")
        assert enc2.encoder_id == "hashing:subword-ngram:v1"
