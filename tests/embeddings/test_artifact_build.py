"""VBV1 artifact build + fail-closed loader tests (SPEC_V6
V6-03.06/03.08 — parse-only, hash-pinned, no pickle/eval).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from verbatim.config import EmbeddingConfig
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.embeddings.artifact import (
    MAGIC,
    ArtifactEncoder,
    _load_vectors,
    _ArtifactInvalid,
)
from verbatim.embeddings.artifact_build import (
    CONFIG_SNIPPET,
    build_artifact,
    encode_vectors_bin,
)
from verbatim.embeddings.codec import Float32Codec


def _cfg(model="m1", revision="r1") -> EmbeddingConfig:
    return EmbeddingConfig(
        backend="artifact", model=model, artifact_revision=revision
    )


def _table(dim=4):
    return {
        "alpha": [1.0] + [0.0] * (dim - 1),
        "beta": [0.0, 1.0] + [0.0] * (dim - 2),
        "gamma": [0.0, 0.0, 1.0] + [0.0] * (dim - 3),
    }


def _build(tmp_path, table=None, dim=4, model="m1", revision="r1"):
    out = Path(tmp_path) / "models"
    manifest = build_artifact(
        str(out),
        model=model,
        revision=revision,
        table=table if table is not None else _table(dim),
        dim=dim,
    )
    return out, manifest


def _enc(tmp_path, model="m1", revision="r1", manifest=None):
    return ArtifactEncoder(
        _cfg(model, revision), data_dir=Path(tmp_path), manifest=manifest
    )


class TestBuild:
    def test_layout_and_manifest(self, tmp_path):
        out, m = _build(tmp_path)
        assert (out / "m1" / "r1" / "vectors.bin").is_file()
        assert (out / "artifact_manifest.json").is_file()
        entry = m["models"]["m1"]["r1"]
        assert entry["format"] == "VBV1"
        assert entry["dim"] == 4
        fmeta = entry["files"]["vectors.bin"]
        payload = (out / "m1" / "r1" / "vectors.bin").read_bytes()
        import hashlib
        assert fmeta["sha256"] == hashlib.sha256(payload).hexdigest()
        assert fmeta["size"] == len(payload)
        assert entry["loader_source_sha256"]
        # on-disk manifest equals returned manifest
        disk = json.loads((out / "artifact_manifest.json").read_text())
        assert disk == m

    def test_header_format(self, tmp_path):
        payload = encode_vectors_bin(_table(4), 4)
        magic, dim, count = struct.unpack_from("<4sIQ", payload, 0)
        assert magic == b"VBV1" and dim == 4 and count == 3

    def test_deterministic_bytes(self, tmp_path):
        t = _table(4)
        b1 = encode_vectors_bin(t, 4)
        b2 = encode_vectors_bin(dict(reversed(list(t.items()))), 4)
        assert b1 == b2  # key order of the input mapping is irrelevant

    def test_dim_inferred(self, tmp_path):
        out, m = _build(tmp_path, table=_table(4), dim=None)
        assert m["models"]["m1"]["r1"]["dim"] == 4

    def test_ragged_table_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            build_artifact(
                str(tmp_path / "models"), model="m", revision="r",
                table={"a": [1.0], "b": [1.0, 2.0]}, dim=1,
            )

    def test_dim_mismatch_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            build_artifact(
                str(tmp_path / "models"), model="m", revision="r",
                table=_table(4), dim=8,
            )

    def test_unsafe_names_rejected(self, tmp_path):
        for bad in ("../x", "a/b", "..", "", "latest", "a\\b"):
            with pytest.raises(ValueError):
                build_artifact(
                    str(tmp_path / "models"), model=bad, revision="r",
                    table=_table(2), dim=2,
                )
            with pytest.raises(ValueError):
                build_artifact(
                    str(tmp_path / "models"), model="m", revision=bad,
                    table=_table(2), dim=2,
                )

    def test_nonfinite_rows_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            build_artifact(
                str(tmp_path / "models"), model="m", revision="r",
                table={"a": [float("nan"), 0.0]}, dim=2,
            )

    def test_config_snippet_documents_overrides(self):
        assert "artifact" in CONFIG_SNIPPET
        assert "artifact_revision" in CONFIG_SNIPPET
        assert "data_dir" in CONFIG_SNIPPET


class TestLoadVectors:
    def test_bad_magic(self, tmp_path):
        p = tmp_path / "v.bin"
        p.write_bytes(b"XXXX" + encode_vectors_bin(_table(4), 4)[4:])
        with pytest.raises(_ArtifactInvalid):
            _load_vectors(p, 4)

    def test_short_file(self, tmp_path):
        p = tmp_path / "v.bin"
        p.write_bytes(b"VBV")
        with pytest.raises(_ArtifactInvalid):
            _load_vectors(p, 4)

    def test_dim_mismatch(self, tmp_path):
        p = tmp_path / "v.bin"
        p.write_bytes(encode_vectors_bin(_table(4), 4))
        with pytest.raises(_ArtifactInvalid):
            _load_vectors(p, 8)

    def test_truncated(self, tmp_path):
        p = tmp_path / "v.bin"
        p.write_bytes(encode_vectors_bin(_table(4), 4)[:-3])
        with pytest.raises(_ArtifactInvalid):
            _load_vectors(p, 4)

    def test_trailing_bytes(self, tmp_path):
        p = tmp_path / "v.bin"
        p.write_bytes(encode_vectors_bin(_table(4), 4) + b"junk")
        with pytest.raises(_ArtifactInvalid):
            _load_vectors(p, 4)

    def test_unsorted_keys(self, tmp_path):
        # Hand-craft an unsorted payload: sortedness is part of VBV1.
        t = _table(4)
        rows = sorted(t.items(), key=lambda kv: kv[0], reverse=True)
        parts = [struct.pack("<4sIQ", MAGIC, 4, len(rows))]
        for k, v in rows:
            kb = k.encode()
            parts.append(struct.pack("<H", len(kb)) + kb)
            parts.append(struct.pack("<4f", *v))
        p = tmp_path / "v.bin"
        p.write_bytes(b"".join(parts))
        with pytest.raises(_ArtifactInvalid):
            _load_vectors(p, 4)

    def test_nonfinite_row(self, tmp_path):
        payload = bytearray(encode_vectors_bin({"a": [1.0]}, 1))
        struct.pack_into("<f", payload, len(payload) - 4, float("nan"))
        p = tmp_path / "v.bin"
        p.write_bytes(bytes(payload))
        with pytest.raises(_ArtifactInvalid):
            _load_vectors(p, 1)


class TestVerifiedEncoder:
    def test_roundtrip_available(self, tmp_path):
        _build(tmp_path)
        enc = _enc(tmp_path)
        assert enc.available() is True
        assert enc.encoder_id == "artifact:m1:r1"
        assert enc.dimensions == 4
        assert enc.name == "artifact:m1:r1"

    def test_encode_determinism_and_codec(self, tmp_path):
        _build(tmp_path)
        enc = _enc(tmp_path)
        b1 = enc.encode(["alpha beta", "unknown-zzz", "alpha"])
        b2 = enc.encode(["alpha beta", "unknown-zzz", "alpha"])
        assert b1 == b2
        for blob in b1:
            vec = Float32Codec.validate_blob(blob, 4)
            assert len(vec) == 4
            assert pytest.approx(1.0) == sum(x * x for x in vec) ** 0.5
        # "alpha" alone == its table row (unit vector e0)
        assert Float32Codec.unpack(b1[2], 4) == pytest.approx(
            (1.0, 0.0, 0.0, 0.0)
        )

    def test_oov_texts_near_orthogonal(self, tmp_path):
        _build(tmp_path)
        enc = _enc(tmp_path)
        a, b, a2 = enc.encode(["zzq-one", "zzq-two", "zzq-one"])
        assert a == a2  # same OOV text → same vector
        va = Float32Codec.unpack(a, 4)
        vb = Float32Codec.unpack(b, 4)
        cos = sum(x * y for x, y in zip(va, vb))
        assert abs(cos) < 0.9  # distinct OOV texts are not falsely equal

    def test_mean_semantics(self, tmp_path):
        _build(tmp_path)
        enc = _enc(tmp_path)
        blob = enc.encode(["alpha beta"])[0]
        v = Float32Codec.unpack(blob, 4)
        s = 2 ** -0.5
        assert v == pytest.approx((s, s, 0.0, 0.0), abs=1e-6)

    def test_encode_validation(self, tmp_path):
        _build(tmp_path)
        enc = _enc(tmp_path)
        with pytest.raises(VerbatimError) as ei:
            enc.encode("not-a-list")
        assert ei.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError):
            enc.encode([1, 2])

    def test_manifest_reports_verified(self, tmp_path):
        _build(tmp_path)
        enc = _enc(tmp_path)
        m = enc.manifest()
        assert m["status"] == "verified"
        assert m["dimensions"] == 4
        assert m["preprocessing_version"]


class TestFailClosed:
    def test_no_manifest_unavailable(self, tmp_path):
        enc = _enc(tmp_path)
        assert enc.available() is False
        with pytest.raises(VerbatimError) as ei:
            enc.encode(["x"])
        assert ei.value.code == ErrorCode.ENCODER_UNAVAILABLE
        with pytest.raises(VerbatimError) as ei:
            _ = enc.dimensions
        assert ei.value.code == ErrorCode.ENCODER_UNAVAILABLE

    def test_wrong_revision_unavailable(self, tmp_path):
        _build(tmp_path)
        enc = _enc(tmp_path, revision="r9")
        assert enc.available() is False
        with pytest.raises(VerbatimError):
            enc.encode(["x"])

    def test_latest_rejected(self, tmp_path):
        _build(tmp_path, revision="r1")
        enc = _enc(tmp_path, revision="latest")
        assert enc.available() is False

    def test_empty_revision_unavailable(self, tmp_path):
        _build(tmp_path)
        enc = ArtifactEncoder(
            EmbeddingConfig(backend="artifact", model="m1",
                            artifact_revision=None),
            data_dir=Path(tmp_path),
        )
        assert enc.available() is False
        assert enc.encoder_id == "artifact:m1:unpinned"

    def test_tampered_file_unavailable_not_crash(self, tmp_path):
        _build(tmp_path)
        enc = _enc(tmp_path)
        assert enc.available() is True
        vpath = Path(tmp_path) / "models" / "m1" / "r1" / "vectors.bin"
        blob = bytearray(vpath.read_bytes())
        blob[20] ^= 0xFF
        vpath.write_bytes(bytes(blob))
        assert enc.available() is False  # sha256 mismatch
        with pytest.raises(VerbatimError) as ei:
            enc.encode(["x"])
        assert ei.value.code == ErrorCode.ENCODER_UNAVAILABLE

    def test_deleted_file_unavailable(self, tmp_path):
        _build(tmp_path)
        (Path(tmp_path) / "models" / "m1" / "r1" / "vectors.bin").unlink()
        assert _enc(tmp_path).available() is False

    def test_tampered_manifest_hash(self, tmp_path):
        out, _ = _build(tmp_path)
        mpath = out / "artifact_manifest.json"
        m = json.loads(mpath.read_text())
        meta = m["models"]["m1"]["r1"]["files"]["vectors.bin"]
        meta["sha256"] = "0" * 64
        mpath.write_text(json.dumps(m))
        assert _enc(tmp_path).available() is False

    def test_manifest_size_mismatch(self, tmp_path):
        out, _ = _build(tmp_path)
        mpath = out / "artifact_manifest.json"
        m = json.loads(mpath.read_text())
        m["models"]["m1"]["r1"]["files"]["vectors.bin"]["size"] += 1
        mpath.write_text(json.dumps(m))
        assert _enc(tmp_path).available() is False

    def test_corrupt_but_hash_pinned_unavailable(self, tmp_path):
        """A manifest that correctly pins a corrupt file's real hash
        still fails — the strict parse gate runs after sha256."""
        out = Path(tmp_path) / "models"
        rev = out / "m1" / "r1"
        rev.mkdir(parents=True)
        bad = b"XXXX" + b"\x00" * 12
        (rev / "vectors.bin").write_bytes(bad)
        import hashlib
        m = {
            "models": {"m1": {"r1": {
                "files": {"vectors.bin": {
                    "sha256": hashlib.sha256(bad).hexdigest(),
                    "size": len(bad)}},
                "format": "VBV1", "dim": 4}}},
        }
        (out / "artifact_manifest.json").write_text(json.dumps(m))
        assert _enc(tmp_path).available() is False

    def test_wrong_loader_hash_unavailable(self, tmp_path):
        out, _ = _build(tmp_path)
        mpath = out / "artifact_manifest.json"
        m = json.loads(mpath.read_text())
        m["models"]["m1"]["r1"]["loader_source_sha256"] = "f" * 64
        mpath.write_text(json.dumps(m))
        assert _enc(tmp_path).available() is False

    def test_unknown_format_unavailable(self, tmp_path):
        out, _ = _build(tmp_path)
        mpath = out / "artifact_manifest.json"
        m = json.loads(mpath.read_text())
        m["models"]["m1"]["r1"]["format"] = "PICKLE"
        mpath.write_text(json.dumps(m))
        assert _enc(tmp_path).available() is False

    def test_manifest_at_data_dir_root_also_accepted(self, tmp_path):
        """The shipped-package layout pins the manifest at data_dir."""
        out, m = _build(tmp_path)
        (Path(tmp_path) / "artifact_manifest.json").write_text(
            json.dumps(m)
        )
        (out / "artifact_manifest.json").unlink()
        assert _enc(tmp_path).available() is True

    def test_data_dir_manifest_takes_precedence(self, tmp_path):
        """``<data_dir>/artifact_manifest.json`` wins over the
        build-output manifest under ``models/`` when both exist."""
        out, m = _build(tmp_path)
        bogus = json.loads(json.dumps(m))
        bogus["models"]["m1"]["r1"]["dim"] = 9  # would fail verification
        (Path(tmp_path) / "artifact_manifest.json").write_text(
            json.dumps(bogus)
        )
        # root manifest wins → dim 9 != file dim 4 → unavailable
        assert _enc(tmp_path).available() is False
        (Path(tmp_path) / "artifact_manifest.json").unlink()
        assert _enc(tmp_path).available() is True

    def test_injected_manifest(self, tmp_path):
        """The manifest kwarg path (tests/shipped pins) verifies the same
        checklist against data_dir files."""
        out, m = _build(tmp_path)
        (out / "artifact_manifest.json").unlink()  # no file manifest
        enc = _enc(tmp_path, manifest=m)
        assert enc.available() is True
        # ...and the injected manifest fails closed on revision too
        enc2 = _enc(tmp_path, revision="rX", manifest=m)
        assert enc2.available() is False

    def test_available_never_raises(self, tmp_path):
        # garbage manifest file
        (Path(tmp_path) / "models").mkdir()
        (Path(tmp_path) / "models" / "artifact_manifest.json").write_text(
            "{not json"
        )
        assert _enc(tmp_path).available() is False
