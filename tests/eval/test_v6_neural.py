"""V6 neural-artifact eval tests — build path, calibration fit, and the
paired quality gate (SPEC_V6 V6-03.07/03.09/03.10)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from eval.v6.neural import (
    ENCODER_ID,
    MODEL,
    REVISION,
    build_dev_artifact,
    dev_encoder,
    dev_vocabulary,
    distilled_table,
    fit_calibration,
    paired_quality,
)
from eval.v5.corpus import seed_corpus
from verbatim.querying.calibration import (
    calibration_for,
    is_calibrated,
    similarity_floor,
)


class TestBuildDevArtifact:
    def test_build_produces_loadable_artifact(self, tmp_path):
        rep = build_dev_artifact(str(tmp_path), memories=16, seed=42)
        assert rep["encoder_id"] == ENCODER_ID
        assert rep["vocab_size"] > 0
        manifest_path = (
            Path(rep["models_root"]) / "artifact_manifest.json"
        )
        assert manifest_path.is_file()
        vec = (
            Path(rep["models_root"]) / MODEL / REVISION / "vectors.bin"
        )
        assert vec.is_file()

        enc = dev_encoder(str(tmp_path))
        assert enc.available() is True
        assert enc.encoder_id == ENCODER_ID
        assert enc.dimensions == rep["dim"]

    def test_vocab_covers_corpus_terms(self):
        corpus = seed_corpus(memories=16, seed=42)
        vocab = set(dev_vocabulary(corpus))
        # gold query terms are covered — a query never hits OOV on
        # vocabulary grounds alone
        assert "wifi" in vocab
        assert "eng-4821" in vocab

    def test_table_rows_are_teacher_float32(self):
        from verbatim.embeddings.codec import Float32Codec
        from verbatim.embeddings.hashing import HashingEncoder
        from verbatim.config import EmbeddingConfig

        table = distilled_table(["wifi"], dim=384)
        teacher = HashingEncoder(EmbeddingConfig(backend="hashing"))
        want = Float32Codec.unpack(teacher.encode(["wifi"])[0], 384)
        assert table["wifi"] == list(want)

    def test_rebuild_deterministic(self, tmp_path):
        r1 = build_dev_artifact(str(tmp_path / "a"), memories=16, seed=42)
        r2 = build_dev_artifact(str(tmp_path / "b"), memories=16, seed=42)
        v1 = (
            Path(r1["models_root"]) / MODEL / REVISION / "vectors.bin"
        ).read_bytes()
        v2 = (
            Path(r2["models_root"]) / MODEL / REVISION / "vectors.bin"
        ).read_bytes()
        assert v1 == v2


class TestCalibrationFit:
    def test_entry_registered(self):
        rec = calibration_for(ENCODER_ID)
        assert rec is not None
        assert rec.floor is not None and rec.floor > 0.0
        assert is_calibrated(ENCODER_ID)
        assert similarity_floor(ENCODER_ID, "factual") == rec.floor
        # unlisted class fails closed
        assert similarity_floor(ENCODER_ID, "identifier") is None
        # unknown encoder fails closed
        assert similarity_floor("artifact:hash-distilled:v9") is None

    def test_validate_separates(self, tmp_path):
        rep = fit_calibration(str(tmp_path), memories=16, seed=42)
        assert rep["encoder_id"] == ENCODER_ID
        assert rep["floor"] is not None
        assert rep["separates"] is True
        # every reject-slice score sits strictly under the fitted floor
        assert rep["reject_envelope_max"] < rep["floor"]

    def test_validate_unverified_encoder_no_crash(self, tmp_path):
        """An unverified artifact reports honestly — validate() returns a
        report with no scores instead of propagating ENCODER_UNAVAILABLE."""
        from verbatim.querying.calibration import validate

        enc = dev_encoder(str(tmp_path))  # nothing built
        assert enc.available() is False
        rep = validate(enc)
        assert rep["encoder_id"] == ENCODER_ID
        assert rep["separates"] is False
        assert rep["slices"]["support"]["n"] == 0


class TestPairedQuality:
    def test_end_to_end_report(self, tmp_path):
        rep = paired_quality(str(tmp_path), memories=16, seed=42, k=8)
        assert rep["measured"] is True
        assert rep["artifact"]["encoder_id"] == ENCODER_ID
        assert rep["artifact"]["available"] is True
        for arm in ("verbatim_memory-hashing", "verbatim_memory-artifact"):
            assert arm in rep["arms"]
            a = rep["arms"][arm]
            assert a["tasks"] == rep["corpus"]["tasks"]
            assert a["recall_at_k"] is not None
            assert a["errors"] == 0
        v = rep["verdict"]
        assert isinstance(v["neural_recommended"], bool)
        assert v["reason"]
        # Gate arithmetic is consistent with the published numbers.
        h, a = (
            rep["arms"]["verbatim_memory-hashing"],
            rep["arms"]["verbatim_memory-artifact"],
        )
        expected = (
            a["recall_at_k"] > h["recall_at_k"]
            and a["precision_at_k"] >= h["precision_at_k"] - 1e-9
        )
        assert v["neural_recommended"] == expected
        # The artifact arm ran the REAL write path: source_vectors rows
        # exist under the artifact encoder_id.
        import sqlite3
        db = os.path.join(tmp_path, "arm-artifact", "mem.db")
        conn = sqlite3.connect(db)
        try:
            rows = conn.execute(
                "SELECT DISTINCT encoder FROM source_vectors"
            ).fetchall()
        finally:
            conn.close()
        assert [r[0] for r in rows] == [ENCODER_ID]

    def test_unavailable_artifact_honest_verdict(self, tmp_path,
                                               monkeypatch):
        """When the pinned artifact fails verification the arm reports
        unavailable and the gate cannot recommend — never a silent
        hashing fallback."""
        import eval.v6.neural as neural
        from verbatim.embeddings.artifact import ArtifactEncoder
        from verbatim.config import EmbeddingConfig

        dead = ArtifactEncoder(
            EmbeddingConfig(backend="artifact", model=MODEL,
                            artifact_revision=REVISION),
            data_dir=Path(tmp_path) / "empty",
        )
        assert dead.available() is False
        monkeypatch.setattr(neural, "dev_encoder", lambda *a, **k: dead)

        rep = neural.paired_quality(str(tmp_path), memories=16, seed=42)
        assert rep["artifact"]["available"] is False
        art = rep["arms"]["verbatim_memory-artifact"]
        assert art["unavailable"] is True
        assert rep["verdict"]["neural_recommended"] is False
        assert "unavailable" in rep["verdict"]["reason"]
        # the hashing arm still measured — the report stays honest
        assert rep["arms"]["verbatim_memory-hashing"]["tasks"] > 0
