"""Optional semantic retrieval: encoders, vector codec, exact cosine.

Semantic retrieval is OPTIONAL and independent of any remote decision
service (SPEC §4, §28). Importing this package performs no network calls,
loads no models, and opens no sockets — encoder construction happens only
through :func:`get_encoder` at explicit call sites (SPEC §6).
"""

from __future__ import annotations

from .codec import Float32Codec
from .encoder import (
    Encoder,
    RemoteEncoder,
    SpanText,
    encode_spans,
    encoder_identity,
    encoder_requires_permit,
    get_encoder,
)
from .hashing import HashingEncoder
from .ollama import HttpResult, HttpTransport, OllamaEncoder
from .vectors import EmbeddingMatrix, cosine, coverage, load_matrix

__all__ = [
    "Encoder",
    "RemoteEncoder",
    "Float32Codec",
    "get_encoder",
    "encode_spans",
    "encoder_identity",
    "encoder_requires_permit",
    "SpanText",
    "HashingEncoder",
    "OllamaEncoder",
    "HttpResult",
    "HttpTransport",
    "EmbeddingMatrix",
    "load_matrix",
    "cosine",
    "coverage",
]
