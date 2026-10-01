"""Markdown file projection semantics — SPEC_V4 §47 (V4-47.01/02/06).

Covers: deterministic bytes, stable object references, revision + view
kind + provenance + edit-semantics markers, quote-vs-read-only authority
modes, held/suppressed content absence with honest withheld counts,
denied-scope indistinguishability, and projection-root path safety.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.projections import (
    ProjectionAuthority,
    build,
    check_inside,
    read_current,
    safe_relpath,
    scope_filename,
)
from verbatim.projections import markdown as md
from verbatim.security.quarantine import open_quarantine
from verbatim.purge import suppress

from .conftest import grant, scope_row, seed_claim


def _auth(pid="human:alice", purpose="recall"):
    return ProjectionAuthority(principal_id=pid, purpose=purpose)


def _seeded(store, verbs=("read", "quote")):
    with store.tx() as conn:
        scope_row(conn, "sA")
        grant(conn, "sA", verbs=verbs)
        seed_claim(
            conn, "cl1", "sA", "src1", "sp1", "the cat sat on the mat",
            obj={"text": "cat on mat"}, predicate="note",
        )
        seed_claim(
            conn, "cl2", "sA", "src2", "sp2", "second fact here",
            obj={"text": "second fact"},
        )


def _scope_file(res):
    return next(f["path"] for f in res["files"] if f["path"].startswith("scope-"))


class TestRendering:
    def test_deterministic_bytes(self, store, tmp_path):
        _seeded(store)
        auth = _auth()
        r1 = build(store, str(tmp_path / "p1"), authority=auth)
        r2 = build(store, str(tmp_path / "p2"), authority=auth)
        f = _scope_file(r1)
        assert read_current(str(tmp_path / "p1"), f) == read_current(
            str(tmp_path / "p2"), f
        )
        # Rebuild in place — same inputs, same bytes.
        r3 = build(store, str(tmp_path / "p1"), authority=auth)
        assert read_current(str(tmp_path / "p1"), f) == read_current(
            str(tmp_path / "p2"), f
        )
        assert r1["inputs_sha256"] == r3["inputs_sha256"]

    def test_stable_refs_revision_provenance_edit_semantics(
        self, store, tmp_path
    ):
        _seeded(store)
        res = build(store, str(tmp_path / "p"), authority=_auth())
        text = read_current(str(tmp_path / "p"), _scope_file(res)).decode()
        hdr = md.parse_header(text)
        assert hdr["kind"] == "scope"
        assert hdr["store"] == store.db_id()
        assert hdr["projection_generation"] == store.projection_generation()
        assert hdr["quote"] is True
        # V4-47.01: stable ref + revision + provenance + edit semantics.
        assert '"claim_id":"cl1"' in text
        assert '"revision":1' in text
        assert '"edit":"proposal_only"' in text
        assert '"source_id":"src1"' in text
        assert '"provenance":"direct_user"' in text
        assert "payload_sha256" in text  # revision digest in provenance
        assert "the cat sat on the mat" in text  # quote-mode excerpt

    def test_quote_grant_releases_exact_bytes(self, store, tmp_path):
        _seeded(store)
        res = build(store, str(tmp_path / "p"), authority=_auth())
        text = read_current(str(tmp_path / "p"), _scope_file(res)).decode()
        blocks = list(md.iter_quote_blocks(text))
        assert len(blocks) == 2
        for sha, nbytes, quote in blocks:
            assert hashlib.sha256(quote.encode()).hexdigest() == sha
            assert len(quote.encode()) == nbytes

    def test_read_only_authority_is_metadata_only(self, store, tmp_path):
        _seeded(store, verbs=("read",))
        res = build(store, str(tmp_path / "p"), authority=_auth())
        srow = res["scopes"][0]
        assert srow["quote"] is False
        text = read_current(str(tmp_path / "p"), srow["file"]).decode()
        assert "the cat sat on the mat" not in text
        assert "verbatim:quote" not in text
        assert "excerpt withheld" in text
        # References + digests remain — the file is still navigable.
        assert '"claim_id":"cl1"' in text
        assert '"locator_sha256"' in text

    def test_no_authority_omits_scope(self, store, tmp_path):
        with store.tx() as conn:
            scope_row(conn, "sA")
            scope_row(conn, "sB")
            grant(conn, "sA")                       # alice: read+quote sA
            grant(conn, "sB", pid="human:bob")      # bob only on sB
            seed_claim(conn, "cl1", "sA", "src1", "sp1", "alpha")
            seed_claim(conn, "cl2", "sB", "src2", "sp2", "beta secret")
        res = build(store, str(tmp_path / "p"), authority=_auth())
        assert res["built_scopes"] == ["sA"]
        assert res["denied_scopes"] == 1
        assert "beta secret" not in read_current(
            str(tmp_path / "p"), "index.md"
        ).decode()

    def test_inactive_heads_omitted_and_counted(self, store, tmp_path):
        with store.tx() as conn:
            scope_row(conn, "sA")
            grant(conn, "sA")
            seed_claim(conn, "cl1", "sA", "src1", "sp1", "live claim")
            seed_claim(
                conn, "cl2", "sA", "src2", "sp2", "pending claim",
                state="pending",
            )
            seed_claim(
                conn, "cl3", "sA", "src3", "sp3", "rejected claim",
                state="rejected",
            )
        res = build(store, str(tmp_path / "p"), authority=_auth())
        srow = res["scopes"][0]
        text = read_current(str(tmp_path / "p"), srow["file"]).decode()
        assert '"claim_id":"cl1"' in text
        assert '"claim_id":"cl2"' not in text
        assert '"claim_id":"cl3"' not in text
        assert res["inactive"] == 2


class TestWithholding:
    """V4-43.09 / §36: held and tombstoned content never ships."""

    def test_quarantined_source_withholds_claim(self, store, tmp_path):
        _seeded(store)
        with store.tx() as conn:
            open_quarantine(
                conn, ("source", "src2", 1), ["manual:test"],
                [{"rule_id": "test"}], scope_id="sA",
            )
        res = build(store, str(tmp_path / "p"), authority=_auth())
        srow = res["scopes"][0]
        text = read_current(str(tmp_path / "p"), srow["file"]).decode()
        assert "the cat sat on the mat" in text
        assert "second fact here" not in text
        assert '"claim_id":"cl2"' not in text
        assert res["withheld"] >= 1
        assert "withheld" in text  # honest marker, no leak

    def test_quarantined_span_withholds_dependent_claim(
        self, store, tmp_path
    ):
        _seeded(store)
        with store.tx() as conn:
            open_quarantine(
                conn, ("span", "sp1", 1), ["manual:test"],
                [{"rule_id": "test"}], scope_id="sA",
            )
        res = build(store, str(tmp_path / "p"), authority=_auth())
        text = read_current(str(tmp_path / "p"), _scope_file(res)).decode()
        assert "the cat sat on the mat" not in text
        assert '"claim_id":"cl1"' not in text
        assert "second fact here" in text

    def test_suppressed_claim_absent(self, store, tmp_path):
        """A suppressed claim's head is tombstoned — the projection must
        not carry its content, its id, or even a 'withheld' marker that
        would leak that something was erased (V4-43.09, §36)."""
        _seeded(store)
        suppress(store, "sA", [("claim", "cl1", None)], "tester")
        res = build(store, str(tmp_path / "p"), authority=_auth())
        text = read_current(str(tmp_path / "p"), _scope_file(res)).decode()
        assert "the cat sat on the mat" not in text
        assert '"claim_id":"cl1"' not in text
        assert '"span_id":"sp1"' not in text
        assert "second fact here" in text

    def test_suppressed_source_revision_absent(self, store, tmp_path):
        _seeded(store)
        suppress(store, "sA", [("source_revision", "src2", 1)], "tester")
        res = build(store, str(tmp_path / "p"), authority=_auth())
        text = read_current(str(tmp_path / "p"), _scope_file(res)).decode()
        assert "second fact here" not in text
        assert '"claim_id":"cl2"' not in text


class TestPathSafety:
    """V4-47.06: roots reject escapes, hooks, hidden names."""

    def test_safe_relpath_rejects_traversal(self):
        for bad in ("../x.md", "a/b.md", ".hidden.md", "..", "x", "x.py"):
            with pytest.raises(VerbatimError) as ei:
                safe_relpath(bad)
            assert ei.value.code is ErrorCode.VALIDATION

    def test_root_symlink_rejected(self, store, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        with pytest.raises(VerbatimError) as ei:
            build(store, str(link), authority=_auth())
        assert ei.value.code is ErrorCode.VALIDATION

    def test_check_inside_escapes(self, tmp_path):
        root = str(tmp_path / "proj")
        os.makedirs(root)
        with pytest.raises(VerbatimError):
            check_inside(root, str(tmp_path / "outside.md"))
        with pytest.raises(VerbatimError):
            check_inside(root, os.path.join(root, ".git", "hooks", "x"))

    def test_scope_filename_safe_and_distinct(self):
        a = scope_filename("sA")
        b = scope_filename("../evil/../../x")
        assert a == safe_relpath(a) and a.startswith("scope-")
        assert b == safe_relpath(b) and ".." not in b
        assert a != b
