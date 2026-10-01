"""Portable bundles — SPEC_V4 §47 (V4-47.02/03/04, C68).

A bundle is a deterministic tar of one published generation. Import
verifies the manifest + per-file digests, enforces path/size/member
bounds, and labels every projected row against the local store —
foreign rows stay foreign; suppressed rows can never resurrect.
"""

from __future__ import annotations

import io
import json
import os
import tarfile

import pytest

from verbatim.core.types import ErrorCode, Scope, VerbatimError
from verbatim.ingest import Ingester
from verbatim.config import VerbatimConfig
from verbatim.projections import (
    ProjectionAuthority,
    build,
    export_bundle,
    import_bundle,
    verify_bundle,
)
from verbatim.purge import suppress

from .conftest import grant, scope_row, seed_claim


def _auth(pid="human:alice", purpose="recall"):
    return ProjectionAuthority(principal_id=pid, purpose=purpose)


def _built(store, root):
    with store.tx() as conn:
        scope_row(conn, "sA")
        grant(conn, "sA")
        seed_claim(conn, "cl1", "sA", "src1", "sp1", "the cat sat",
                   obj={"text": "cat"})
        seed_claim(conn, "cl2", "sA", "src2", "sp2", "second fact",
                   obj={"text": "second"})
    return build(store, str(root), authority=_auth())


def _craft_tar(members, dest):
    """Write a hostile/edge-case tar: members = [(name, bytes, type)]."""
    with tarfile.open(dest, "w:") as tf:
        for name, data, mtype in members:
            if mtype == tarfile.SYMTYPE:
                info = tarfile.TarInfo(name=name)
                info.type = tarfile.SYMTYPE
                info.linkname = data.decode()
                tf.addfile(info)
            else:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.type = mtype
                tf.addfile(info, io.BytesIO(data))


class TestExport:
    def test_bundle_bytes_deterministic(self, store, tmp_path):
        _built(store, tmp_path / "proj")
        a = export_bundle(str(tmp_path / "proj"), str(tmp_path / "a.tar"))
        b = export_bundle(str(tmp_path / "proj"), str(tmp_path / "b.tar"))
        assert a["bundle_sha256"] == b["bundle_sha256"]
        assert open(a["path"], "rb").read() == open(b["path"], "rb").read()

    def test_manifest_fields(self, store, tmp_path):
        _built(store, tmp_path / "proj")
        out = export_bundle(str(tmp_path / "proj"), str(tmp_path / "a.tar"))
        v = verify_bundle(out["path"])
        man = v["manifest"]
        assert man["store_id"] == store.db_id()
        assert man["projection_generation"] == store.projection_generation()
        assert man["builder"]["version"] == "1.0.0"
        assert {f["path"] for f in v["files"]} == set(man["files"])

    def test_corrupted_generation_fails_export(self, store, tmp_path):
        """V4-43.06: projection corruption is INTEGRITY, not STORE_CORRUPT."""
        res = _built(store, tmp_path / "proj")
        f = next(
            f["path"] for f in res["files"] if f["path"].startswith("scope-")
        )
        gen_dir = os.path.join(str(tmp_path / "proj"), res["published"])
        path = os.path.join(gen_dir, f)
        data = open(path, "rb").read()
        open(path, "wb").write(data[:-1] + b"X")
        with pytest.raises(VerbatimError) as ei:
            export_bundle(str(tmp_path / "proj"), str(tmp_path / "a.tar"))
        assert ei.value.code is ErrorCode.INTEGRITY


class TestVerify:
    def test_tampered_member_rejected(self, store, tmp_path):
        _built(store, tmp_path / "proj")
        out = export_bundle(str(tmp_path / "proj"), str(tmp_path / "a.tar"))
        v = verify_bundle(out["path"])
        # Repack with one member's content mutated — digest must catch it.
        name = sorted(
            n for n in v["contents"] if n != "manifest.json"
        )[0]
        members = []
        for n in v["contents"]:
            data = v["contents"][n]
            if n == name:
                data = data[:-1] + bytes([data[-1] ^ 0xFF])
            members.append((n, data, tarfile.REGTYPE))
        _craft_tar(members, str(tmp_path / "evil.tar"))
        with pytest.raises(VerbatimError) as ei:
            verify_bundle(str(tmp_path / "evil.tar"))
        assert ei.value.code is ErrorCode.INTEGRITY

    def test_manifest_digest_mismatch(self, store, tmp_path):
        _built(store, tmp_path / "proj")
        out = export_bundle(str(tmp_path / "proj"), str(tmp_path / "a.tar"))
        # Repack with a manifest that lies about a digest.
        v = verify_bundle(out["path"])
        man = dict(v["manifest"])
        files = dict(man["files"])
        name = sorted(files)[0]
        files[name] = dict(files[name], sha256="0" * 64)
        man["files"] = files
        members = [
            (n, v["contents"][n], tarfile.REGTYPE) for n in v["contents"]
            if n != "manifest.json"
        ]
        members.append(
            ("manifest.json",
             (json.dumps(man, sort_keys=True) + "\n").encode(),
             tarfile.REGTYPE)
        )
        _craft_tar(members, str(tmp_path / "evil.tar"))
        with pytest.raises(VerbatimError) as ei:
            verify_bundle(str(tmp_path / "evil.tar"))
        assert ei.value.code is ErrorCode.INTEGRITY

    def test_traversal_member_rejected(self, store, tmp_path):
        _craft_tar(
            [("../escape.md", b"evil", tarfile.REGTYPE)],
            str(tmp_path / "evil.tar"),
        )
        with pytest.raises(VerbatimError) as ei:
            verify_bundle(str(tmp_path / "evil.tar"))
        assert ei.value.code in (
            ErrorCode.VALIDATION,
            ErrorCode.INTEGRITY,
        )

    def test_symlink_member_rejected(self, store, tmp_path):
        _craft_tar(
            [("link.md", b"/etc/passwd", tarfile.SYMTYPE)],
            str(tmp_path / "evil.tar"),
        )
        with pytest.raises(VerbatimError) as ei:
            verify_bundle(str(tmp_path / "evil.tar"))
        assert ei.value.code is ErrorCode.INTEGRITY

    def test_missing_manifest_rejected(self, tmp_path):
        _craft_tar(
            [("index.md", b"# hi", tarfile.REGTYPE)],
            str(tmp_path / "nomani.tar"),
        )
        with pytest.raises(VerbatimError) as ei:
            verify_bundle(str(tmp_path / "nomani.tar"))
        assert ei.value.code is ErrorCode.INTEGRITY

    def test_extra_unlisted_member_rejected(self, store, tmp_path):
        _built(store, tmp_path / "proj")
        out = export_bundle(str(tmp_path / "proj"), str(tmp_path / "a.tar"))
        v = verify_bundle(out["path"])
        members = [
            (n, v["contents"][n], tarfile.REGTYPE) for n in v["contents"]
        ]
        members.append(("stowaway.md", b"not in manifest", tarfile.REGTYPE))
        _craft_tar(members, str(tmp_path / "evil.tar"))
        with pytest.raises(VerbatimError) as ei:
            verify_bundle(str(tmp_path / "evil.tar"))
        assert ei.value.code is ErrorCode.INTEGRITY


class TestImport:
    def test_foreign_rows_labeled_unverifiable(self, store, store_b, tmp_path):
        """V4-47.04: foreign ids carry no local authority."""
        _built(store, tmp_path / "proj")
        out = export_bundle(str(tmp_path / "proj"), str(tmp_path / "a.tar"))
        rep = import_bundle(store_b, out["path"])
        assert rep["foreign_store_id"] == store.db_id()
        assert rep["counts"]["foreign-unverifiable"] == 2
        assert rep["counts"]["verified"] == 0
        # Nothing was written to the second store.
        with store_b.read() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM claims"
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM sources"
            ).fetchone()[0] == 0

    def test_same_store_rows_verify(self, store, tmp_path):
        """Round-trip on the origin store verifies digests honestly."""
        _built(store, tmp_path / "proj")
        out = export_bundle(str(tmp_path / "proj"), str(tmp_path / "a.tar"))
        rep = import_bundle(store, out["path"])
        assert rep["counts"]["verified"] == 2
        assert rep["quote_blocks_verified"] == 2

    def test_suppressed_rows_never_resurrect(self, store, tmp_path):
        """C68/V4-47.03: a locally suppressed claim stays suppressed."""
        _built(store, tmp_path / "proj")
        out = export_bundle(str(tmp_path / "proj"), str(tmp_path / "a.tar"))
        suppress(store, "sA", [("claim", "cl1", None)], "tester")
        rep = import_bundle(store, out["path"])
        by_id = {c["claim_id"]: c["status"] for c in rep["claims"]}
        assert by_id["cl1"] == "suppressed"
        assert by_id["cl2"] == "verified"
        assert rep["blocked_files"]  # the whole file is refused for ingest

    def test_ingest_lands_documents_not_authority(self, store_b, store, tmp_path):
        """Ingested files become ordinary import sources — foreign
        provenance recorded, no claims/grants minted."""
        _built(store, tmp_path / "proj")
        out = export_bundle(str(tmp_path / "proj"), str(tmp_path / "a.tar"))
        ing = Ingester(store_b, VerbatimConfig())
        rep = import_bundle(
            store_b,
            out["path"],
            ingest=True,
            target_scope=Scope(profile_id="imported"),
            ingester=ing,
        )
        assert rep["ingested"]
        with store_b.read() as conn:
            rows = conn.execute(
                "SELECT s.origin, sr.provenance, sr.metadata_json"
                " FROM sources s JOIN source_revisions sr"
                " ON sr.source_id = s.source_id"
            ).fetchall()
        assert rows and all(r[0] == "projection-bundle" for r in rows)
        assert all(r[1] == "legacy_import" for r in rows)
        meta = json.loads(rows[0][2])
        assert meta["foreign"] is True
        assert meta["foreign_store_id"] == store.db_id()
        # No claims minted from foreign rows.
        with store_b.read() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM claims"
            ).fetchone()[0] == 0
