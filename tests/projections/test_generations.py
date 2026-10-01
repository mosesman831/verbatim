"""Projection generations — SPEC_V4 §43 (V4-43.01–09).

Covers: build-state records + manifest coverage, generation-scoped
atomic visibility (readers see the prior generation while staging),
scope-partition rebuilds that don't strand siblings, checkpointed
resume, stale-artifact discard, suppression observed at publish time,
and generation reclamation.
"""

from __future__ import annotations

import os

import pytest

from verbatim.projections import (
    ProjectionAuthority,
    build,
    prune_staging,
    read_current,
    reclaim,
    status,
    suppress,
)
from verbatim.projections import paths as pjpaths
from verbatim.projections.markdown import parse_header
from verbatim.purge import suppress as purge_suppress
from verbatim.storage.repos_v2 import ProjectionRepo

from .conftest import grant, scope_row, seed_claim


def _auth(pid="human:alice", purpose="recall"):
    return ProjectionAuthority(principal_id=pid, purpose=purpose)


def _two_scopes(store):
    with store.tx() as conn:
        scope_row(conn, "sA")
        scope_row(conn, "sB")
        grant(conn, "sA")
        grant(conn, "sB")
        seed_claim(conn, "clA", "sA", "srcA", "spA", "alpha content",
                   obj={"n": "alpha"})
        seed_claim(conn, "clB", "sB", "srcB", "spB", "beta content",
                   obj={"n": "beta"})


def _file(res, suffix="scope-"):
    return next(f["path"] for f in res["files"] if f["path"].startswith(suffix))


class TestBuildRecords:
    def test_build_row_manifest_coverage(self, store, tmp_path):
        """V4-43.01: watermark + producer manifest + coverage + state."""
        _two_scopes(store)
        res = build(store, str(tmp_path / "p"), authority=_auth())
        assert res["recorded"] is True
        row = ProjectionRepo(store).build_get(res["build_id"])
        assert row is not None
        assert row["kind"] == "markdown-files"
        assert row["status"] == "published"
        assert row["snapshot_seq"] == res["snapshot_seq"]
        assert row["caught_up_seq"] == res["snapshot_seq"]
        import json

        manifest = json.loads(row["manifest_json"])
        assert manifest["builder"] == "markdown-files/1.0.0"
        assert manifest["store_id"] == store.db_id()
        assert manifest["projection_generation"] == res["generation"]
        assert set(manifest["scopes"]) == {"sA", "sB"}
        assert manifest["claims"] == 2
        # CAS publication pointer per scope.
        act = ProjectionRepo(store).active_get("sA", "markdown-files")
        assert act is not None and act["build_id"] == res["build_id"]

    def test_manifest_carries_required_fields(self, store, tmp_path):
        """V4-47: store id, generation, builder version, digests."""
        _two_scopes(store)
        seq_before = res_seq(store)
        res = build(store, str(tmp_path / "p"), authority=_auth())
        st = status(str(tmp_path / "p"))
        man = st["manifest"]
        assert man["store_id"] == store.db_id()
        assert man["projection_generation"] == store.projection_generation()
        assert man["builder"]["name"] == "markdown-files"
        assert man["files"]["index.md"]["sha256"]
        assert man["counts"]["claims"] == 2
        # V4-43.07: coverage states the indexed snapshot explicitly.
        assert man["snapshot_seq"] == seq_before == res["snapshot_seq"]


def res_seq(store):
    with store.read() as conn:
        return int(
            conn.execute(
                "SELECT COALESCE(MAX(event_seq),0) FROM events"
            ).fetchone()[0]
        )


class TestAtomicVisibility:
    def test_readers_see_prior_generation_while_staging(
        self, store, tmp_path
    ):
        """V4-43.02: staged bytes are never reader-visible mid-build."""
        _two_scopes(store)
        root = str(tmp_path / "p")
        r1 = build(store, root, authority=_auth())
        gen1_file = _file(r1)
        gen1_bytes = read_current(root, gen1_file)

        # Mutate the store so generation 2 renders different bytes.
        with store.tx() as conn:
            seed_claim(conn, "clA2", "sA", "srcA2", "spA2",
                       "alpha second fact", obj={"n": "a2"})

        seen_mid_build = {}

        def probe(_relpath):
            # Whatever the reader sees mid-stage must be generation 1.
            seen_mid_build["file"] = read_current(root, gen1_file)
            seen_mid_build["current"] = status(root)["current"]

        r2 = build(store, root, authority=_auth(), on_file=probe)
        assert seen_mid_build["file"] == gen1_bytes
        assert seen_mid_build["current"] == "gen-000001"
        assert r2["published"] == "gen-000002"
        assert status(root)["current"] == "gen-000002"
        new_text = read_current(root, gen1_file).decode()
        assert "alpha second fact" in new_text
        assert gen1_bytes.decode() != new_text

    def test_scoped_rebuild_carries_siblings(self, store, tmp_path):
        """V4-43.04: a scope-partition rebuild keeps sibling files."""
        _two_scopes(store)
        root = str(tmp_path / "p")
        r1 = build(store, root, authority=_auth())
        sB_file = next(
            f["path"] for f in r1["files"] if "-sB-" in f["path"]
        )
        sB_bytes = read_current(root, sB_file)
        gen_before = store.projection_generation()

        with store.tx() as conn:
            seed_claim(conn, "clA2", "sA", "srcA2", "spA2",
                       "alpha second fact", obj={"n": "a2"})
        r2 = build(store, root, authority=_auth(), scope_ids=["sA"])

        # Sibling carried verbatim; global generation untouched by the
        # file-projection build (it is not the store's FTS generation).
        assert read_current(root, sB_file) == sB_bytes
        assert r2["carried_scopes"] == ["sB"]
        assert r2["built_scopes"] == ["sA"]
        assert store.projection_generation() == gen_before
        sA_file = next(f["path"] for f in r2["files"] if "-sA-" in f["path"])
        assert "alpha second fact" in read_current(root, sA_file).decode()

    def test_revoked_sibling_not_carried(self, store, tmp_path):
        """Carry-forward re-authorizes: a revoked right drops the file."""
        _two_scopes(store)
        root = str(tmp_path / "p")
        r1 = build(store, root, authority=_auth())
        sB_file = next(
            f["path"] for f in r1["files"] if "-sB-" in f["path"]
        )
        # Revoke alice's read on sB, then scoped-rebuild sA.
        from verbatim.governance import grants_for, revoke_grant

        with store.tx() as conn:
            for g in grants_for(
                conn, "sB", "human:alice"
            ):
                revoke_grant(conn, g["grant_id"])
        r2 = build(store, root, authority=_auth(), scope_ids=["sA"])
        assert read_current(root, sB_file) is None
        assert "sB" not in r2["carried_scopes"]


class TestResume:
    def test_resume_after_mid_build_crash(self, store, tmp_path):
        """V4-43.05: a checkpointed partial build resumes, not restarts."""
        _two_scopes(store)
        root = str(tmp_path / "p")

        calls = []

        def crash_hook(relpath):
            calls.append(relpath)
            if len(calls) == 2:
                raise RuntimeError("simulated crash mid-stage")

        # A clean build of the SAME snapshot — the resumed output must
        # be byte-identical to what a crash-free run would render.
        rq = build(store, str(tmp_path / "q"), authority=_auth())
        with pytest.raises(RuntimeError):
            build(store, root, authority=_auth(), on_file=crash_hook)
        st = status(root)
        assert st["current"] is None
        assert len(st["staging"]) == 1

        r2 = build(store, root, authority=_auth())
        assert r2["resumed"] is True
        assert r2["published"] == "gen-000001"
        assert status(root)["staging"] == []
        for f in r2["files"]:
            if f["path"].startswith("scope-"):
                assert read_current(root, f["path"]) == read_current(
                    str(tmp_path / "q"), f["path"]
                )

    def test_stale_checkpoint_discarded(self, store, tmp_path):
        """V4-43.05: changed inputs invalidate partial artifacts."""
        _two_scopes(store)
        root = str(tmp_path / "p")

        def crash_hook(_rp):
            raise RuntimeError("crash")

        with pytest.raises(RuntimeError):
            build(store, root, authority=_auth(), on_file=crash_hook)
        # Store moved forward — the checkpoint's watermark is stale.
        with store.tx() as conn:
            seed_claim(conn, "clA2", "sA", "srcA2", "spA2", "new", obj=None)
        r = build(store, root, authority=_auth())
        assert r["resumed"] is False
        assert len(r["discarded"]) == 1
        assert r["published"] == "gen-000001"
        assert '"claim_id":"clA2"' in read_current(root, _file(r)).decode()

    def test_incompatible_db_checkpoint_discarded(self, store, tmp_path):
        """A checkpoint from a different store never resumes here."""
        _two_scopes(store)
        root = str(tmp_path / "p")

        def crash_hook(_rp):
            raise RuntimeError("crash")

        with pytest.raises(RuntimeError):
            build(store, root, authority=_auth(), on_file=crash_hook)
        st_dir = pjpaths.list_staging(str(tmp_path / "p"))[0]
        import json, os

        cp_path = os.path.join(st_dir, "_checkpoint.json")
        cp = json.loads(open(cp_path).read())
        cp["db_id"] = "deadbeef" * 4
        open(cp_path, "w").write(json.dumps(cp))
        r = build(store, root, authority=_auth())
        assert r["resumed"] is False
        assert r["discarded"]

    def test_prune_staging_marks_failed(self, store, tmp_path):
        _two_scopes(store)
        root = str(tmp_path / "p")

        def crash(_rp):
            raise RuntimeError("crash")

        with pytest.raises(RuntimeError):
            build(store, root, authority=_auth(), on_file=crash)
        res = prune_staging(store, root)
        assert res["discarded"]
        assert status(root)["staging"] == []


class TestReclaim:
    def test_old_generations_reclaimed_current_kept(self, store, tmp_path):
        """V4-43.08: old gens are reclaimed; the live pointer never is."""
        _two_scopes(store)
        root = str(tmp_path / "p")
        build(store, root, authority=_auth())
        with store.tx() as conn:
            seed_claim(conn, "clA2", "sA", "srcA2", "spA2", "new fact")
        build(store, root, authority=_auth())
        st = status(root)
        assert sorted(st["generations"]) == ["gen-000001", "gen-000002"]
        out = reclaim(store, root, keep=0)
        assert out["removed"] == ["gen-000001"]
        assert status(root)["current"] == "gen-000002"
        assert read_current(root, "index.md") is not None


class TestImmediateSuppress:
    """V4-43.09: deletion suppresses across active + staging generations
    without waiting for the next scheduled rebuild."""

    def test_excise_published_generation(self, store, tmp_path):
        import hashlib

        _two_scopes(store)
        root = str(tmp_path / "p")
        r1 = build(store, root, authority=_auth())
        sA_file = next(
            f["path"] for f in r1["files"] if "-sA-" in f["path"]
        )
        sB_file = next(
            f["path"] for f in r1["files"] if "-sB-" in f["path"]
        )
        assert '"claim_id":"clA"' in read_current(root, sA_file).decode()

        # Canonical tombstone first, then the projection pass.
        purge_suppress(store, "sA", [("claim", "clA")], "human:alice")
        out = suppress(store, root, [("claim", "clA")])
        assert out["suppressed_claims"] == ["clA"]
        assert out["generations"] == ["gen-000001"]
        assert out["canon_unsuppressed"] == []

        text = read_current(root, sA_file).decode()
        assert '"claim_id":"clA"' not in text
        assert "alpha content" not in text
        assert "suppressed by deletion policy" in text
        hdr = parse_header(text)
        assert hdr["claims"] == 0
        assert hdr["withheld"] == 1
        # Sibling scope untouched.
        assert '"claim_id":"clB"' in read_current(root, sB_file).decode()

        # Every digest in the rewritten manifest still verifies.
        man = status(root)["manifest"]
        for path, fmeta in man["files"].items():
            raw = read_current(root, path)
            assert raw is not None
            assert hashlib.sha256(raw).hexdigest() == fmeta["sha256"]
            assert len(raw) == fmeta["bytes"]
        assert man["events"][-1]["type"] == "suppression"
        assert man["events"][-1]["claims"] == ["clA"]
        assert man["counts"]["claims"] == 1  # only clB remains

        # The next scheduled rebuild keeps the claim out — the canonical
        # tombstone drops it from the head-revision set at plan time.
        r2 = build(store, root, authority=_auth())
        assert '"claim_id":"clA"' not in read_current(
            root, sA_file
        ).decode()
        assert "alpha content" not in read_current(
            root, sA_file
        ).decode()

    def test_span_target_excises_citing_claim(self, store, tmp_path):
        """Suppressing an evidence span excises the claim citing it."""
        _two_scopes(store)
        root = str(tmp_path / "p")
        r1 = build(store, root, authority=_auth())
        sA_file = next(
            f["path"] for f in r1["files"] if "-sA-" in f["path"]
        )
        out = suppress(store, root, [("span", "spA")])
        assert out["suppressed_claims"] == ["clA"]
        text = read_current(root, sA_file).decode()
        assert '"claim_id":"clA"' not in text
        # Not canonically suppressed (span target, no purge run) — the
        # result says so honestly rather than implying store consensus.
        assert out["canon_unsuppressed"] == []

    def test_staging_bytes_physically_removed(self, store, tmp_path):
        """V4-43.09: attributable staged bytes are deleted, not left
        for a crashed build's resume to publish."""
        _two_scopes(store)
        root = str(tmp_path / "p")

        calls = []

        def crash_after_first(rp):
            calls.append(rp)
            if len(calls) == 1:
                raise RuntimeError("crash after first staged file")

        with pytest.raises(RuntimeError):
            build(store, root, authority=_auth(), on_file=crash_after_first)
        st_dir = pjpaths.list_staging(root)[0]
        staged_names = os.listdir(st_dir)
        assert any(n.startswith("scope-sA-") for n in staged_names)

        out = suppress(store, root, [("claim", "clA")])
        assert any("scope-sA-" in p for p in out["staging_files"])
        assert not any(
            n.startswith("scope-sA-") for n in os.listdir(st_dir)
        )
        # Unrelated staged bytes (sB not yet written / other scope) are
        # untouched; the checkpoint no longer claims the sA file.
        import json

        cp = json.loads(
            open(os.path.join(st_dir, "_checkpoint.json")).read()
        )
        assert not any(
            n.startswith("scope-sA-") for n in cp.get("files", {})
        )
        # clA has no canonical tombstone — the report says so.
        assert out["canon_unsuppressed"] == ["clA"]

    def test_unrelated_targets_are_noop(self, store, tmp_path):
        _two_scopes(store)
        root = str(tmp_path / "p")
        build(store, root, authority=_auth())
        out = suppress(
            store, root, [("episode", "ep9"), ("claim", "nope")]
        )
        assert out["suppressed_claims"] == []
        assert out["generations"] == []
        assert out["not_applicable"] == ["episode:ep9"]
        assert out["canon_unsuppressed"] == ["nope"]
