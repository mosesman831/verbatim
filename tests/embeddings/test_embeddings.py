"""Embeddings package tests — codec, loopback guard, encoder, vectors.

No test performs real network I/O: every encoder test injects a fake
``HttpTransport``, and construction-time endpoint validation is pure
string parsing. SQLite runs in-memory against the real DDL.

COMPAT NOTE (F4-04 / SPEC_V4 §12): transport-bound encoders now refuse
``encode`` without a broker-minted ``DispatchPermit`` bound to the exact
request payload — the prior "caller wraps encode in EgressGate" contract
was the finding's defect. These unit tests attach ``_StubBroker`` (a
binding-checking stand-in) and mint permits via ``_permit()``; the real
broker's consent/budget/one-use/crash semantics are covered by
``tests/privacy/test_broker.py`` and ``tests/embeddings/test_permit_gate.py``.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.config import EmbeddingConfig, VerbatimConfig
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.embeddings import (
    EmbeddingMatrix,
    Float32Codec,
    OllamaEncoder,
    SpanText,
    cosine,
    coverage,
    encode_spans,
    get_encoder,
    load_matrix,
)
from verbatim.embeddings.ollama import HttpResult, PREPROCESSING_VERSION
from verbatim.storage.schema import DDL_V1


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

class FakeStore:
    """Store stand-in: only ``hmac`` is used by the embeddings write path."""

    def __init__(self, key: bytes = b"test-hmac-key") -> None:
        self._key = key

    def hmac(self, data: bytes) -> bytes:
        return hmac_mod.new(self._key, data, hashlib.sha256).digest()


class FakeEncoder:
    """Deterministic encoder returning codec-packed vectors from a table."""

    def __init__(self, dims: int = 4, vectors: dict[str, list[float]] | None = None):
        self._dims = dims
        self._vectors = vectors or {}
        self.encoded: list[list[str]] = []

    @property
    def encoder_id(self) -> str:
        return "fake:model-1:rev-9"

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def normalization(self) -> str:
        return "l2"

    def encode(self, texts: list[str]) -> list[bytes]:
        self.encoded.append(list(texts))
        out = []
        for t in texts:
            vec = self._vectors.get(t) or [1.0] + [0.0] * (self._dims - 1)
            out.append(Float32Codec.pack(vec))
        return out

    def available(self) -> bool:
        return True

    def manifest(self) -> dict:
        return {
            "artifact_revision": "rev-9",
            "dimensions": self._dims,
            "normalization": "l2",
            "license_id": None,
            "preprocessing_version": "fake-pre-1",
            "manifest_json": {"backend": "fake"},
        }


class FakeTransport:
    """HttpTransport stub: scripted responses keyed by (method, url path)."""

    def __init__(self, routes: dict | None = None, fail: Exception | None = None):
        self.routes = routes or {}
        self.fail = fail
        self.calls: list[dict] = []

    def request(self, method, url, *, body, headers, timeout_s):
        from urllib.parse import urlsplit

        path = urlsplit(url).path
        self.calls.append({"method": method, "url": url, "body": body})
        if self.fail is not None:
            raise self.fail
        return self.routes.get(path, HttpResult(status=404, body=b""))


def _cfg(**kw) -> EmbeddingConfig:
    return EmbeddingConfig(backend="ollama", **kw)


class _StubBroker:
    """Binding-checking permit validator for transport unit tests.

    Verifies the same contract ``TransportBroker.dispatch`` enforces —
    DispatchPermit type, recipient identity, and exact payload digest —
    without a store. Consent/budget/one-use/crash semantics are exercised
    against the real broker in tests/privacy/test_broker.py.
    """

    def __init__(self) -> None:
        self.dispatched: list[str] = []

    def dispatch(self, permit, *, recipient, payload_digest):
        from verbatim.core.types_v4 import DispatchPermit

        if not isinstance(permit, DispatchPermit):
            raise VerbatimError(ErrorCode.EGRESS_DENIED, "DispatchPermit required")
        if (
            permit.recipient != recipient
            or permit.payload_digest != payload_digest
        ):
            raise VerbatimError(ErrorCode.EGRESS_DENIED, "permit binding mismatch")
        self.dispatched.append(permit.permit_id)
        return permit


def _permit(enc, texts):
    """A test permit bound to ``enc``'s endpoint and this exact payload."""
    from verbatim.core.types_v4 import DispatchPermit

    return DispatchPermit(
        permit_id="p-test",
        recipient=enc.endpoint_id,
        purpose="embed_document",
        payload_digest=enc.payload_digest(texts),
        scope_ids=("scope-1",),
        consent_refs=(),
        reservation_id="r-test",
        max_spend=0.01,
        issued_us=1,
        expires_us=2**62,
    )


def _encode(enc, texts):
    """encode() under a minted permit — the post-F4-04 calling convention."""
    return enc.encode(texts, permit=_permit(enc, texts))


def _ollama(**kw) -> OllamaEncoder:
    http = kw.pop("http", None)
    broker = kw.pop("broker", None) or _StubBroker()
    return OllamaEncoder(_cfg(**kw), http=http, broker=broker)


def _vec_json(vectors):
    import json

    return HttpResult(status=200, body=json.dumps({"embeddings": vectors}).encode())


def _mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(DDL_V1)
    return conn


def _seed_span(conn: sqlite3.Connection, span_id: str, *, text: bytes = b"hello") -> None:
    """Insert the FK chain scope → source → revision → span."""
    # Real digests under the shim's key so verified reads (payload_hmac,
    # excerpt_hmac over the 0..len(text) slice) would pass for these rows.
    digest = FakeStore().hmac(text)
    conn.execute(
        "INSERT OR IGNORE INTO scopes(scope_id, profile_id, visibility)"
        " VALUES (?, 'profile-1', 'owner')",
        ("scope-1",),
    )
    conn.execute(
        "INSERT INTO sources(source_id, origin, source_kind, scope_id, created_us)"
        " VALUES (?, 'test', 'user_message', 'scope-1', 1)",
        (f"src-{span_id}",),
    )
    conn.execute(
        "INSERT INTO source_revisions(source_id, revision, payload, payload_hmac,"
        " event_us, captured_us, provenance) VALUES (?, 1, ?, ?, 1, 1, 'direct_user')",
        (f"src-{span_id}", text, digest),
    )
    conn.execute(
        "INSERT INTO spans(span_id, source_id, revision, start_byte, end_byte,"
        " excerpt_hmac, harvester_version) VALUES (?, ?, 1, 0, ?, ?, 'hv-1')",
        (span_id, f"src-{span_id}", len(text), digest),
    )


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------

class TestCodec:
    def test_roundtrip(self):
        vec = [0.5, -1.25, 3.0, 1e-9]
        blob = Float32Codec.pack(vec)
        assert len(blob) == 16
        back = Float32Codec.unpack(blob, 4)
        assert back == pytest.approx(vec, abs=1e-7)

    def test_pack_is_little_endian(self):
        # 1.0f little-endian is 00 00 80 3f — byte order is pinned, not native.
        blob = Float32Codec.pack([1.0])
        assert blob == b"\x00\x00\x80\x3f"

    def test_validate_ok(self):
        assert Float32Codec.validate([3.0, 4.0], 2) == (3.0, 4.0)

    def test_validate_wrong_dim(self):
        with pytest.raises(VerbatimError) as ei:
            Float32Codec.validate([1.0, 2.0], 3)
        assert ei.value.code == ErrorCode.VECTOR_INVALID

    @pytest.mark.parametrize("bad", [[float("nan"), 0.0], [float("inf"), 0.0], [0.0, float("-inf")]])
    def test_validate_non_finite(self, bad):
        with pytest.raises(VerbatimError) as ei:
            Float32Codec.validate(bad, 2)
        assert ei.value.code == ErrorCode.VECTOR_INVALID

    def test_validate_zero_norm(self):
        with pytest.raises(VerbatimError) as ei:
            Float32Codec.validate([0.0, 0.0, 0.0], 3)
        assert ei.value.code == ErrorCode.VECTOR_INVALID

    def test_unpack_wrong_blob_len(self):
        with pytest.raises(VerbatimError) as ei:
            Float32Codec.unpack(b"\x00" * 8, 3)
        assert ei.value.code == ErrorCode.VECTOR_INVALID

    def test_validate_blob_roundtrip(self):
        blob = Float32Codec.pack([0.1, 0.2])
        assert Float32Codec.validate_blob(blob, 2) == pytest.approx((0.1, 0.2), abs=1e-7)


# --------------------------------------------------------------------------
# endpoint validation (pure string parsing — no sockets)
# --------------------------------------------------------------------------

class TestEndpointValidation:
    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://127.0.0.1:11434",
            "http://localhost:11434",
            "http://[::1]:11434",
            "http://127.0.0.1",  # missing port normalizes to the allowlist
        ],
    )
    def test_accepts_loopback(self, endpoint):
        enc = _ollama(endpoint=endpoint, http=FakeTransport())
        assert enc.encoder_id.startswith("ollama:")

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://example.com:11434",
            "http://192.168.1.5:11434",
            "http://0.0.0.0:11434",
            "https://127.0.0.1:11434",  # scheme must be plain http
            "http://127.0.0.1:8080",  # non-allowlisted port
            "http://localhost:9999",
            "http://user:pw@127.0.0.1:11434",  # credentials
            "http://127.0.0.1:11434/api",  # path
            "http://127.0.0.1:11434?x=1",  # query
            "ftp://127.0.0.1:11434",
        ],
    )
    def test_rejects_non_loopback_or_weird(self, endpoint):
        with pytest.raises(VerbatimError) as ei:
            _ollama(endpoint=endpoint, http=FakeTransport())
        assert ei.value.code == ErrorCode.CONFIG_INVALID


# --------------------------------------------------------------------------
# OllamaEncoder with injected transport — no real network
# --------------------------------------------------------------------------

class TestOllamaEncoder:
    def test_encode_packs_vectors(self):
        transport = FakeTransport({"/api/embeddings": _vec_json([[1.0, 0.0], [0.0, 1.0]])})
        enc = _ollama(http=transport)
        blobs = _encode(enc, ["a", "b"])
        assert len(blobs) == 2
        assert Float32Codec.unpack(blobs[0], 2) == pytest.approx((1.0, 0.0))
        assert enc.dimensions == 2
        # Request went to the loopback base with model+input payload.
        call = transport.calls[0]
        assert call["url"].endswith("/api/embeddings")
        import json

        assert json.loads(call["body"]) == {"model": "nomic-embed-text", "input": ["a", "b"]}

    def test_dimensions_undiscovered_raises(self):
        enc = _ollama(http=FakeTransport())
        with pytest.raises(VerbatimError) as ei:
            _ = enc.dimensions
        assert ei.value.code == ErrorCode.ENCODER_UNAVAILABLE

    def test_declared_vs_actual_mismatch(self):
        # First call pins dims=2; second batch returns a 3-vector.
        transport = FakeTransport()
        enc = _ollama(http=transport)
        transport.routes["/api/embeddings"] = _vec_json([[1.0, 0.0]])
        _encode(enc, ["x"])
        transport.routes["/api/embeddings"] = _vec_json([[1.0, 0.0, 0.5]])
        with pytest.raises(VerbatimError) as ei:
            _encode(enc, ["y"])
        assert ei.value.code == ErrorCode.VECTOR_INVALID

    def test_mismatched_count_rejected(self):
        transport = FakeTransport({"/api/embeddings": _vec_json([[1.0]])})
        enc = _ollama(http=transport)
        with pytest.raises(VerbatimError) as ei:
            _encode(enc, ["a", "b"])
        assert ei.value.code == ErrorCode.VECTOR_INVALID

    def test_non_200_raises_unavailable(self):
        transport = FakeTransport({"/api/embeddings": HttpResult(status=500, body=b"")})
        enc = _ollama(http=transport)
        with pytest.raises(VerbatimError) as ei:
            _encode(enc, ["a"])
        assert ei.value.code == ErrorCode.ENCODER_UNAVAILABLE

    def test_available_caches_and_never_raises(self):
        transport = FakeTransport({"/api/tags": HttpResult(status=200, body=b"{}")})
        enc = _ollama(http=transport)
        assert enc.available() is True
        assert enc.available() is True
        # Cached: only one /api/tags call despite two probes.
        assert sum(1 for c in transport.calls if c["url"].endswith("/api/tags")) == 1

    def test_available_false_on_network_error(self):
        transport = FakeTransport(fail=OSError("connection refused"))
        enc = _ollama(http=transport)
        assert enc.available() is False  # never raises

    def test_encoder_id_and_manifest(self):
        enc = _ollama(
            http=FakeTransport({"/api/embeddings": _vec_json([[1.0, 2.0]])}),
            model="nomic-embed-text",
            artifact_revision="sha256:abc",
        )
        assert enc.encoder_id == "ollama:nomic-embed-text:sha256:abc"
        _encode(enc, ["t"])
        m = enc.manifest()
        assert m["dimensions"] == 2
        assert m["normalization"] == "l2"
        assert m["license_id"] is None
        assert m["preprocessing_version"] == PREPROCESSING_VERSION

    def test_encoder_id_unpinned(self):
        enc = _ollama(http=FakeTransport())
        assert enc.encoder_id == "ollama:nomic-embed-text:unpinned"


# --------------------------------------------------------------------------
# encode_spans — write path against real DDL
# --------------------------------------------------------------------------

class TestEncodeSpans:
    def test_stores_rows_and_manifest(self):
        conn = _mem_conn()
        _seed_span(conn, "span-a")
        _seed_span(conn, "span-b")
        store = FakeStore()
        enc = FakeEncoder(dims=3)
        n = encode_spans(
            store,
            conn,
            enc,
            [SpanText("span-a", "alpha text"), SpanText("span-b", "beta text")],
        )
        assert n == 2
        rows = conn.execute(
            "SELECT span_id, encoder_id, preprocessing_version, dimensions, dtype,"
            " vector, source_hmac FROM embeddings ORDER BY span_id"
        ).fetchall()
        assert len(rows) == 2
        sid, eid, prev, dims, dtype, vec, hmac_ = rows[0]
        assert sid == "span-a"
        assert eid == "fake:model-1:rev-9"
        assert prev == "fake-pre-1"
        assert dims == 3 and dtype == "float32le"
        assert hmac_ == store.hmac(b"alpha text")
        assert Float32Codec.unpack(vec, 3) == pytest.approx((1.0, 0.0, 0.0))
        m = conn.execute("SELECT * FROM encoder_manifests WHERE encoder_id=?",
                         ("fake:model-1:rev-9",)).fetchone()
        assert m is not None

    def test_upsert_refreshes_vector(self):
        conn = _mem_conn()
        _seed_span(conn, "span-a")
        enc = FakeEncoder(dims=2, vectors={"t": [1.0, 0.0]})
        encode_spans(FakeStore(), conn, enc, [SpanText("span-a", "t")])
        enc._vectors["t"] = [0.0, 1.0]
        encode_spans(FakeStore(), conn, enc, [SpanText("span-a", "t")])
        rows = conn.execute("SELECT vector FROM embeddings").fetchall()
        assert len(rows) == 1
        assert Float32Codec.unpack(rows[0][0], 2) == pytest.approx((0.0, 1.0))

    def test_invalid_vector_rejects_nothing_persisted(self):
        conn = _mem_conn()
        _seed_span(conn, "span-a")
        enc = FakeEncoder(dims=2, vectors={"bad": [float("nan"), 0.0]})
        with pytest.raises(VerbatimError) as ei:
            encode_spans(FakeStore(), conn, enc, [SpanText("span-a", "bad")])
        assert ei.value.code == ErrorCode.VECTOR_INVALID
        assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
        # Manifest registration happens after validation — no partial writes.
        assert conn.execute("SELECT COUNT(*) FROM encoder_manifests").fetchone()[0] == 0

    def test_empty_rows_no_encode_call(self):
        conn = _mem_conn()
        enc = FakeEncoder()
        assert encode_spans(FakeStore(), conn, enc, []) == 0
        assert enc.encoded == []


# --------------------------------------------------------------------------
# cosine search + coverage
# --------------------------------------------------------------------------

class TestVectors:
    def _conn_with(self, entries):
        """entries: [(span_id, [floats])]"""
        conn = _mem_conn()
        eid = "fake:model-1:rev-9"
        for sid, vec in entries:
            _seed_span(conn, sid)
            conn.execute(
                "INSERT INTO embeddings(span_id, encoder_id, preprocessing_version,"
                " dimensions, dtype, vector, source_hmac) VALUES (?, ?, 'p', ?,"
                " 'float32le', ?, ?)",
                (
                    sid,
                    eid,
                    len(vec),
                    Float32Codec.pack(vec),
                    FakeStore().hmac(b"hello"),  # span text is b"hello"
                ),
            )
        return conn, eid

    def test_load_and_cosine_ordering(self):
        conn, eid = self._conn_with(
            [("s1", [1.0, 0.0]), ("s2", [0.9, 0.1]), ("s3", [0.0, 1.0])]
        )
        ids, vecs = load_matrix(conn, ["s1", "s2", "s3"], eid)
        assert set(ids) == {"s1", "s2", "s3"}
        q = Float32Codec.pack([1.0, 0.0])
        scores = cosine(q, EmbeddingMatrix(ids, vecs), 3)
        assert [s for s, _ in scores] == ["s1", "s2", "s3"]
        assert scores[0][1] == pytest.approx(1.0)
        assert scores[2][1] == pytest.approx(0.0, abs=1e-6)

    def test_cosine_excludes_zero_norm_rows(self):
        conn, eid = self._conn_with([("s1", [1.0, 0.0]), ("sz", [0.0, 0.0])])
        ids, vecs = load_matrix(conn, ["s1", "sz"], eid)
        q = Float32Codec.pack([1.0, 0.0])
        scores = cosine(q, EmbeddingMatrix(ids, vecs), 5)
        assert [s for s, _ in scores] == ["s1"]

    def test_cosine_empty_matrix(self):
        assert cosine(Float32Codec.pack([1.0]), EmbeddingMatrix([], []), 5) == []

    def test_load_matrix_bounded_and_scoped(self):
        conn, eid = self._conn_with([("s1", [1.0]), ("s2", [0.5])])
        ids, _ = load_matrix(conn, ["s1"], eid)
        assert ids == ["s1"]  # only the authorized id loads
        # F4-12: inputs past the old 4096 bound are no longer rejected —
        # the scan streams in bounded batches instead of materializing a
        # capped matrix. Missing ids simply report as missing coverage.
        ids, vecs = load_matrix(conn, ["x"] * 5000, eid)
        assert ids == [] and vecs == []

    def test_coverage_math(self):
        conn, eid = self._conn_with([("s1", [1.0]), ("s2", [0.5])])
        _seed_span(conn, "s3")  # present but unembedded
        embedded, total = coverage(conn, eid)
        assert embedded == 2 and total == 3


# --------------------------------------------------------------------------
# get_encoder factory
# --------------------------------------------------------------------------

class TestFactory:
    def test_none_backend(self):
        cfg = VerbatimConfig(embedding=EmbeddingConfig(backend="none"))
        assert get_encoder(cfg) is None

    def test_artifact_unavailable(self):
        from verbatim.embeddings.artifact import ArtifactEncoder

        cfg = VerbatimConfig(embedding=EmbeddingConfig(backend="artifact"))
        enc = get_encoder(cfg)
        assert isinstance(enc, ArtifactEncoder)
        assert enc.available() is False
        with pytest.raises(VerbatimError) as ei:
            enc.encode(["x"])
        assert ei.value.code == ErrorCode.ENCODER_UNAVAILABLE

    def test_ollama_backend(self):
        cfg = VerbatimConfig(embedding=EmbeddingConfig(backend="ollama"))
        enc = get_encoder(cfg, http=FakeTransport())
        assert isinstance(enc, OllamaEncoder)
