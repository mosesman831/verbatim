"""Edit semantics — SPEC_V4 V4-47.05 (C67).

File edits become review proposals pinned to the revision the file
showed; they never overwrite original evidence. Conflict detection
comes from the existing ``reviews.expected_versions`` discipline.
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.projections import (
    ProjectionAuthority,
    audit_file,
    build,
    propose_edit,
    read_current,
)

from .conftest import bump_head, grant, scope_row, seed_claim


def _auth(pid="human:alice", purpose="recall"):
    return ProjectionAuthority(principal_id=pid, purpose=purpose)


def _seeded(store, verbs=("read", "quote", "review")):
    with store.tx() as conn:
        scope_row(conn, "sA")
        grant(conn, "sA", verbs=verbs)
        seed_claim(conn, "cl1", "sA", "src1", "sp1", "the cat sat",
                   obj={"text": "cat"})


class TestProposeEdit:
    def test_proposal_created_claim_untouched(self, store):
        _seeded(store)
        out = propose_edit(
            store, authority=_auth(), scope_id="sA", claim_id="cl1",
            base_revision=1, new_object={"text": "cat on rug"},
            note="corrected from projected file",
        )
        assert out["conflict"] is False
        assert out["applied"] is False
        with store.read() as conn:
            row = conn.execute(
                "SELECT state, proposed_effect_json, expected_versions_json"
                " FROM reviews WHERE review_id = ?",
                (out["review_id"],),
            ).fetchone()
            assert row is not None and row[0] == "open"
            import json
            assert json.loads(row[2]) == {"cl1": 1}
            # The original evidence is untouched.
            assert conn.execute(
                "SELECT revision FROM claim_revisions WHERE claim_id='cl1'"
            ).fetchall() == [(1,)]

    def test_stale_base_is_conflict_not_merge(self, store):
        _seeded(store)
        with store.tx() as conn:
            bump_head(conn, "cl1", 2, obj={"text": "moved"})
        with pytest.raises(VerbatimError) as ei:
            propose_edit(
                store, authority=_auth(), scope_id="sA", claim_id="cl1",
                base_revision=1, new_object={"text": "x"},
            )
        assert ei.value.code is ErrorCode.STALE_PROPOSAL

    def test_foreign_claim_denied(self, store):
        _seeded(store)
        with pytest.raises(VerbatimError) as ei:
            propose_edit(
                store, authority=_auth(), scope_id="sA",
                claim_id="nonexistent", base_revision=1,
                new_object={"text": "x"},
            )
        assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED

    def test_no_review_verb_denied(self, store):
        _seeded(store, verbs=("read", "quote"))
        with pytest.raises(VerbatimError) as ei:
            propose_edit(
                store, authority=_auth(), scope_id="sA", claim_id="cl1",
                base_revision=1, new_object={"text": "x"},
            )
        assert ei.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


class TestAuditFile:
    def test_audit_current_stale_foreign(self, store, tmp_path):
        _seeded(store)
        res = build(store, str(tmp_path / "p"), authority=_auth())
        f = next(
            x["path"] for x in res["files"] if x["path"].startswith("scope-")
        )
        text = read_current(str(tmp_path / "p"), f).decode()

        out = audit_file(store, authority=_auth(), scope_id="sA", text=text)
        assert out["counts"] == {"current": 1}

        with store.tx() as conn:
            bump_head(conn, "cl1", 2, obj={"text": "moved"})
        out = audit_file(store, authority=_auth(), scope_id="sA", text=text)
        assert out["counts"] == {"stale": 1}
        assert out["entries"][0]["head_revision"] == 2
