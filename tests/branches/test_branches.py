"""I5 reversible memory branches (SPEC_V4_5 §07; V45-07.*; D09/D10).

Every test drives the real ``BranchService`` against real
``Store.create`` fixtures — claims stand on genuine
source/span/claim_evidence chains so ``LifecycleMachine.apply``
preconditions are honestly satisfiable. Apply runs the same
transition + review path the coordinator uses; purge paths run the
real ``plan_purge``/``execute_purge`` closure.
"""

from __future__ import annotations

import pytest

from verbatim.branches import (
    BRANCH_KIND,
    BranchService,
    STATE_ABANDONED,
    STATE_APPLIED,
    STATE_HELD,
    STATE_LIVE,
)
from verbatim.core.lifecycle import LifecycleMachine, read_claim_head
from verbatim.core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    safe_json_loads,
)
from verbatim.purge import execute_purge, plan_purge
from verbatim.storage import repos_v4
from verbatim.storage.repos import ReviewsRepo
from verbatim.synthesis import Synthesizer
from verbatim.synthesis.types import VIEW_OBJECT_KIND

from .conftest import (
    T0,
    add_claim,
    bootstrap,
    caller,
    transition,
)


SID = "scope:a"


def _seed(store, claims=("c1", "c2"), sid=SID):
    """Bootstrap governance + real evidence-chained claims."""
    with store.tx() as conn:
        bootstrap(conn, sid)
        for i, cid in enumerate(claims):
            add_claim(
                conn, cid, sid, f"text for {cid}",
                recorded_from=1 + i,
            )


def _objects_row(conn, branch_id):
    return repos_v4.get(
        conn, "objects", {"kind": BRANCH_KIND, "object_id": branch_id}
    )


def _head(conn, claim_id):
    return read_claim_head(conn, claim_id)


def _purge(store, *targets):
    """Real production erasure — preview + physical purge + closure."""
    preview = plan_purge(store, SID, list(targets), actor="alice")
    return execute_purge(store, preview["purge_id"])


# ---------------------------------------------------------------------------
# D09 — creation, isolation, authorization (V45-07.01/07.02)
# ---------------------------------------------------------------------------

def test_create_isolates_overlay_from_live_state(store):
    """A branch is a durable, digest-bound overlay — creating it writes
    zero claim mutations, and its base snapshot pins epoch, parents, and
    revisions through the same edge tables every dependent uses."""
    _seed(store)
    svc = BranchService(store)
    out = svc.create(
        caller(), SID, name="fix-c1",
        ops=[{"effect": "archive", "claim_id": "c1"}],
    )
    bid = out["branch_id"]
    with store.read() as conn:
        # Live claims untouched — the overlay never leaks into live
        # recall state (D09).
        head = _head(conn, "c1")
        assert (head.revision, head.state.value) == (1, "active")
        assert conn.execute(
            "SELECT COUNT(*) FROM claim_revisions WHERE claim_id='c1'",
        ).fetchone()[0] == 1
        # Durable registry record — objects + digest-bound revision.
        obj = _objects_row(conn, bid)
        assert obj["disposition"] == "active"
        rev = conn.execute(
            "SELECT digest, metadata_json FROM object_revisions"
            " WHERE kind='branch' AND object_id=? AND revision=1",
            (bid,),
        ).fetchone()
        assert rev[0].startswith("hmac-sha256:")
        doc = safe_json_loads(rev[1])
        assert doc["state"] == "live"
        assert doc["base"]["epoch"] == 0
        assert ["claim", "c1", 1] in doc["base"]["objects"]
        # Ops store references + proposed values — never byte copies:
        # no span payload text appears anywhere in the doc.
        assert "text for c1" not in rev[1]
        # The branch joins both ancestry graphs: kernel invalidation
        # walks dependency_edges, v3 closure walks derivations.
        dep = conn.execute(
            "SELECT parent_kind, parent_id, parent_revision"
            " FROM dependency_edges WHERE child_kind='branch'"
            " AND child_id=?",
            (bid,),
        ).fetchall()
        assert dep == [("claim", "c1", 1)]
        der = conn.execute(
            "SELECT parent_kind, parent_id, parent_revision"
            " FROM derivations WHERE child_kind='branch'"
            " AND child_id=?",
            (bid,),
        ).fetchall()
        assert der == [("claim", "c1", 1)]


def test_branch_metadata_reads_require_authorization(store):
    """Branch docs carry proposed scope content — reads authorize like
    any other scope read; an ungranted principal is denied
    indistinguishably."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID, ops=[{"effect": "archive", "claim_id": "c1"}],
    )["branch_id"]
    with pytest.raises(VerbatimError) as exc:
        svc.get(caller("agent:prod"), bid)
    assert exc.value.code in (
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
        ErrorCode.NOT_FOUND_OR_FORBIDDEN,
    )
    assert svc.list(caller("agent:prod"), SID) == []
    # And the authorized read resolves the doc.
    got = svc.get(caller(), bid)
    assert got["branch_id"] == bid and got["state"] == "live"


def test_create_rejects_missing_or_foreign_parents(store):
    """An op naming a claim that does not resolve in the branch's scope
    fails at proposal time — indistinguishable denial, no record kept."""
    _seed(store)
    svc = BranchService(store)
    with pytest.raises(VerbatimError) as exc:
        svc.create(
            caller(), SID,
            ops=[{"effect": "archive", "claim_id": "ghost"}],
        )
    assert exc.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    # Foreign-scope claim: real claim, wrong scope — same denial.
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('scope:b', 'prof', 'owner')",
        )
        add_claim(conn, "cX", "scope:b", "foreign text")
    with pytest.raises(VerbatimError) as exc2:
        svc.create(
            caller(), SID,
            ops=[{"effect": "archive", "claim_id": "cX"}],
        )
    assert exc2.value.code is ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
    with store.read() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM objects WHERE kind='branch'",
        ).fetchone()[0] == 0


def test_create_rejects_bad_effects_and_erasure(store):
    """The op vocabulary is the claim-lifecycle vocabulary minus
    ``erase`` — erasure belongs to the purge path, never to a what-if."""
    _seed(store)
    svc = BranchService(store)
    with pytest.raises(VerbatimError) as exc:
        svc.create(
            caller(), SID,
            ops=[{"effect": "erase", "claim_id": "c1"}],
        )
    assert exc.value.code is ErrorCode.VALIDATION
    with pytest.raises(VerbatimError) as exc2:
        svc.create(
            caller(), SID,
            ops=[{"effect": "nuke", "claim_id": "c1"}],
        )
    assert exc2.value.code is ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# submit / apply — the reviewed, fenced, atomic path (V45-07.03/07.04)
# ---------------------------------------------------------------------------

def test_submit_records_forecast_and_opens_one_review(store):
    """Submit computes the invalidation forecast (V45-07.04) and opens
    ONE review carrying expected_versions for every pinned parent."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID,
        ops=[{"effect": "archive", "claim_id": "c1"},
             {"effect": "archive", "claim_id": "c2"}],
    )["branch_id"]
    sub = svc.submit(caller(), bid)
    assert sub["review_id"]
    # The forecast names the branch itself (a dependent of both claims).
    assert ["branch", bid, 1] in [list(r) for r in sub["forecast"]]
    with store.read() as conn:
        row = conn.execute(
            "SELECT state, expected_versions_json FROM reviews"
            " WHERE review_id = ?",
            (sub["review_id"],),
        ).fetchone()
        assert row[0] == "open"
        expected = safe_json_loads(row[1])
        assert expected == {"c1": 1, "c2": 1}
        # Only one review row exists for the branch.
        assert conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE scope_id = ?",
            (SID,),
        ).fetchone()[0] == 1
    # Double submit refuses — the review gate is not re-openable.
    with pytest.raises(VerbatimError) as exc:
        svc.submit(caller(), bid)
    assert exc.value.code is ErrorCode.INVALID_TRANSITION


def test_apply_requires_open_review_and_runs_lifecycle_atomically(store):
    """D09→apply: no write reaches live state until the operator gate —
    then every op runs through ``LifecycleMachine.apply`` inside one
    transaction, the review resolves, and a receipt makes replay
    idempotent."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID,
        ops=[{"effect": "archive", "claim_id": "c1"},
             {"effect": "archive", "claim_id": "c2"}],
    )["branch_id"]
    # The operator gate is not optional (V45-07.03).
    with pytest.raises(VerbatimError) as exc:
        svc.apply(caller(), bid)
    assert exc.value.code is ErrorCode.INVALID_TRANSITION
    with store.read() as conn:
        assert _head(conn, "c1").state.value == "active"

    svc.submit(caller(), bid)
    out = svc.apply(caller(), bid)
    assert out["state"] == "applied"
    assert {a["claim_id"] for a in out["applied"]} == {"c1", "c2"}
    with store.read() as conn:
        # Both effects committed — same atomic commit.
        assert _head(conn, "c1").state.value == "archived"
        assert _head(conn, "c2").state.value == "archived"
        rid = svc.get(caller(), bid)["review_id"]
        assert conn.execute(
            "SELECT state FROM reviews WHERE review_id = ?", (rid,),
        ).fetchone()[0] == "approved"
        # Idempotency receipt is durable.
        rec = repos_v4.get(
            conn, "operation_receipts",
            {"operation_id": f"branch-apply:{bid}"},
        )
        assert rec is not None and rec["effects_applied"] == 2
    # Replay returns the stored receipt — no double-apply.
    again = svc.apply(caller(), bid)
    assert again["replayed"] is True


def test_apply_fails_when_review_resolved_elsewhere(store):
    """An open review is the fence — resolving it through the review
    repo before apply makes the branch's apply STALE_PROPOSAL."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID, ops=[{"effect": "archive", "claim_id": "c1"}],
    )["branch_id"]
    sub = svc.submit(caller(), bid)
    with store.tx() as conn:
        ReviewsRepo(store).resolve(
            conn, sub["review_id"], "rejected", 999,
        )
    with pytest.raises(VerbatimError) as exc:
        svc.apply(caller(), bid)
    assert exc.value.code is ErrorCode.STALE_PROPOSAL
    with store.read() as conn:
        assert _head(conn, "c1").state.value == "active"


def test_atomic_apply_rolls_back_per_op_failure(store):
    """A multi-op branch where a later op fails ``LifecycleMachine``
    requirements commits NOTHING — the earlier op's effect rolls back
    with the transaction (V45-07.03 atomicity)."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID,
        ops=[
            {"effect": "archive", "claim_id": "c1"},
            # dispute requires a recorded conflict edge — none exists,
            # so machine.apply raises mid-transaction.
            {"effect": "dispute", "claim_id": "c2"},
        ],
    )["branch_id"]
    svc.submit(caller(), bid)
    with pytest.raises(VerbatimError):
        svc.apply(caller(), bid)
    with store.read() as conn:
        # Nothing committed: c1 was never archived, the review is still
        # open, and the branch is not marked applied.
        assert _head(conn, "c1").state.value == "active"
        assert _head(conn, "c1").revision == 1
        rid = svc.get(caller(), bid)["review_id"]
        assert conn.execute(
            "SELECT state FROM reviews WHERE review_id = ?", (rid,),
        ).fetchone()[0] == "open"
        assert _objects_row(conn, bid)["disposition"] == "active"
        assert repos_v4.get(
            conn, "operation_receipts",
            {"operation_id": f"branch-apply:{bid}"},
        ) is None


# ---------------------------------------------------------------------------
# fencing — moved, purged, and forecast-shifted parents (V45-07.03/07.04, D10)
# ---------------------------------------------------------------------------

def test_moved_parent_blocks_submit_and_apply(store):
    """A parent that advanced past its pinned revision is STALE_PROPOSAL
    — at submit when it moved early, at apply when it moved under the
    open review."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID, ops=[{"effect": "archive", "claim_id": "c1"}],
    )["branch_id"]
    # The world moved under the proposal before submit.
    transition(store, "c1", "archive")
    with pytest.raises(VerbatimError) as exc:
        svc.submit(caller(), bid)
    assert exc.value.code is ErrorCode.STALE_PROPOSAL
    # Rebase re-pins, then submit succeeds — and a SECOND move between
    # submit and apply is fenced identically.
    svc.rebase(caller(), bid)
    svc.submit(caller(), bid)
    transition(store, "c1", "restore", expected=2)
    with pytest.raises(VerbatimError) as exc2:
        svc.apply(caller(), bid)
    assert exc2.value.code is ErrorCode.STALE_PROPOSAL


def test_purged_parent_fails_without_restoring_bytes(store):
    """D10: purging the pinned parent tombstones the branch itself
    through deletion closure — apply refuses with a typed error and no
    proposed text can resurrect the purged claim."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID,
        ops=[{"effect": "archive", "claim_id": "c1",
              "new_object": {"kind": "literal", "text": "resurrect"}}],
    )["branch_id"]
    svc.submit(caller(), bid)
    _purge(store, ("claim", "c1"))
    with pytest.raises(VerbatimError) as exc:
        svc.apply(caller(), bid)
    # Tombstoned (or held) — either way an honest typed refusal, never a
    # partial apply.
    assert exc.value.code in (
        ErrorCode.INVALID_TRANSITION,
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
        ErrorCode.NOT_FOUND_OR_FORBIDDEN,
    )
    with store.read() as conn:
        # Nothing was restored: the only revisions are the original and
        # the purge's own erased tombstone — no revision carries the
        # branch's proposed payload.
        revs = conn.execute(
            "SELECT revision, state, object_json FROM claim_revisions"
            " WHERE claim_id='c1' ORDER BY revision",
        ).fetchall()
        assert [r[1] for r in revs] == ["active", "erased"]
        assert all("resurrect" not in (r[2] or "") for r in revs)
        # No apply receipt exists — the op never ran.
        assert repos_v4.get(
            conn, "operation_receipts",
            {"operation_id": f"branch-apply:{bid}"},
        ) is None
        # The branch itself is erased (sole parent purged → all-parents
        # closure ⇒ tombstone, V45-12.03).
        assert _objects_row(conn, bid)["disposition"] == "erased"


def test_partial_purge_suppresses_branch_then_rebase_deadends_op(store):
    """Mixed ancestry: purging one of two parents suppresses the branch
    to held (V45-12.03) — submit and apply both refuse while held, and
    rebase marks the dead op honestly rather than dropping it."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID,
        ops=[{"effect": "archive", "claim_id": "c1"},
             {"effect": "archive", "claim_id": "c2"}],
    )["branch_id"]
    _purge(store, ("claim", "c1"))
    with store.read() as conn:
        assert _objects_row(conn, bid)["disposition"] == "held"
    with pytest.raises(VerbatimError) as exc:
        svc.submit(caller(), bid)
    assert exc.value.code is ErrorCode.INVALID_TRANSITION
    with pytest.raises(VerbatimError) as exc2:
        svc.apply(caller(), bid)
    assert exc2.value.code is ErrorCode.INVALID_TRANSITION
    out = svc.rebase(caller(), bid)
    assert out["dead"] == 1
    doc = svc.get(caller(), bid)
    assert doc["state"] == "live"
    assert len(doc["ops"]) == 1 and doc["ops"][0]["claim_id"] == "c2"
    assert doc["dead_ops"][0]["claim_id"] == "c1"
    assert doc["dead_ops"][0]["status"] == "dead"


def test_forecast_mismatch_refuses_apply(store):
    """V45-07.04: a dependent that appears between submit and apply is
    an unpredicted blast radius — apply fails CONTEXT_INCOMPLETE and
    nothing commits."""
    _seed(store)
    svc = BranchService(store)
    synth = Synthesizer(store, kernel=None)
    with store.tx() as conn:
        synth.register_producer(conn)
    bid = svc.create(
        caller(), SID, ops=[{"effect": "archive", "claim_id": "c1"}],
    )["branch_id"]
    svc.submit(caller(), bid)
    # The dependency graph grows after the forecast: a real composed
    # view binds c1 — exactly the hidden dependent D04 forbids
    # publishing past.
    synth.compose(
        SID,
        caller=caller(),
        purpose="recall",
        view_kind="typed_summary",
        inputs=[{"kind": "claim", "id": "c1", "revision": 1}],
        now_us=T0,
    )
    with pytest.raises(VerbatimError) as exc:
        svc.apply(caller(), bid)
    assert exc.value.code is ErrorCode.CONTEXT_INCOMPLETE
    with store.read() as conn:
        assert _head(conn, "c1").state.value == "active"


# ---------------------------------------------------------------------------
# lifecycle — diff, rebase, abandon, invalidation
# ---------------------------------------------------------------------------

def test_diff_reports_per_op_health(store):
    """The operator's review surface: ok → moved → suppressed across a
    parent's lifecycle, never a fabricated green."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID,
        ops=[{"effect": "archive", "claim_id": "c1"},
             {"effect": "archive", "claim_id": "c2"}],
    )["branch_id"]
    d = svc.diff(caller(), bid)
    assert {o["health"] for o in d["ops"]} == {"ok"}
    transition(store, "c1", "archive")
    _purge(store, ("claim", "c2"))
    d2 = svc.diff(caller(), bid)
    health = {o["claim_id"]: o["health"] for o in d2["ops"]}
    assert health["c1"] == "moved"
    assert health["c2"] in ("gone", "suppressed")


def test_rebase_clears_forecast_and_pending_review(store):
    """Rebase re-pins surviving parents to current heads, voids the open
    review honestly, and forces a fresh submit — a rebased branch never
    applies on stale fences."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID, ops=[{"effect": "archive", "claim_id": "c1"}],
    )["branch_id"]
    transition(store, "c1", "archive")
    svc.rebase(caller(), bid)
    sub = svc.submit(caller(), bid)
    # Move again and rebase — the open review is voided, forecast clear.
    transition(store, "c1", "restore", expected=2)
    out = svc.rebase(caller(), bid)
    assert out["moved"] == 1
    doc = svc.get(caller(), bid)
    assert doc["forecast"] is None and doc["review_id"] is None
    with store.read() as conn:
        assert conn.execute(
            "SELECT state FROM reviews WHERE review_id = ?",
            (sub["review_id"],),
        ).fetchone()[0] == "rejected"
    # And the branch can complete its lifecycle on the new pins.
    svc.submit(caller(), bid)
    applied = svc.apply(caller(), bid)
    assert applied["state"] == "applied"
    with store.read() as conn:
        assert _head(conn, "c1").state.value == "archived"


def test_abandon_preserves_audit_record(store):
    """Abandoning is retirement, not deletion — the doc, its base
    snapshot, and the abandon attribution all persist for audit."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID, name="wont-fix",
        ops=[{"effect": "archive", "claim_id": "c1"}],
    )["branch_id"]
    out = svc.abandon(caller(), bid, reason="superseded by hand")
    assert out["state"] == "abandoned"
    doc = svc.get(caller(), bid)
    assert doc["state"] == STATE_ABANDONED
    assert doc["abandon_reason"] == "superseded by hand"
    assert doc["base"]["objects"] == [["claim", "c1", 1]]
    with pytest.raises(VerbatimError):
        svc.apply(caller(), bid)
    with store.read() as conn:
        assert _head(conn, "c1").state.value == "active"
        assert _objects_row(conn, bid)["disposition"] == "archived"


def test_kernel_invalidation_holds_branch_in_same_transaction(store):
    """V45-07.02: a live invalidation event on a pinned parent flips the
    branch to held inside the invalidating transaction — apply refuses
    until an operator rebases."""
    from verbatim.kernel import Kernel

    _seed(store)
    svc = BranchService(store)
    kernel = Kernel(store)
    bid = svc.create(
        caller(), SID, ops=[{"effect": "archive", "claim_id": "c1"}],
    )["branch_id"]
    with store.tx() as conn:
        report = kernel.invalidate(
            conn,
            {"kind": "correction", "scope_ids": [SID],
             "object_refs": [["claim", "c1", 1]]},
        )
        marked = svc.note_invalidation(conn, report)
        assert marked == 1
        # Same-transaction visibility: the hold is already written on
        # the caller's conn — no second commit is needed.
        _obj, doc = svc._doc(conn, bid)
        assert doc["state"] == STATE_HELD
    # The held mark is durable after commit; submit and apply refuse.
    doc = svc.get(caller(), bid)
    assert doc["state"] == STATE_HELD
    with pytest.raises(VerbatimError) as exc:
        svc.apply(caller(), bid)
    assert exc.value.code is ErrorCode.INVALID_TRANSITION
    with pytest.raises(VerbatimError) as exc2:
        svc.submit(caller(), bid)
    assert exc2.value.code is ErrorCode.INVALID_TRANSITION
    out = svc.rebase(caller(), bid)
    assert out["state"] == "live"


# ---------------------------------------------------------------------------
# V45-12.03 — deletion closure tombstones the branch durably
# ---------------------------------------------------------------------------

def test_deletion_closure_tombstones_branch_and_scrubs_text(store):
    """Purging every pinned parent erases the branch through the real
    closure engine: ``objects.disposition`` flips to erased and every
    revision doc becomes a digest-bound tombstone carrying no proposed
    text."""
    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID,
        ops=[{"effect": "archive", "claim_id": "c1",
              "new_object": {"kind": "literal",
                             "text": "proposed replacement text"}}],
    )["branch_id"]
    _purge(store, ("claim", "c1"))
    with store.read() as conn:
        obj = _objects_row(conn, bid)
        assert obj["disposition"] == "erased"
        rows = conn.execute(
            "SELECT revision, digest, metadata_json FROM object_revisions"
            " WHERE kind='branch' AND object_id = ?",
            (bid,),
        ).fetchall()
        assert rows, "branch revisions must persist as tombstones"
        for _rev, digest, meta in rows:
            doc = safe_json_loads(meta)
            assert doc["erased"] is True
            assert doc["state"] == "tombstoned"
            assert digest.startswith("hmac-sha256:")
            # The proposed payload is gone from every revision.
            assert "proposed replacement text" not in meta
            assert "ops" not in doc or not doc["ops"]


def test_scope_purge_reaches_branch_through_provenance(store):
    """The v3-style closure path also reaches the branch — the
    derivations edge recorded at create makes ``affected_by_purge`` see
    it exactly like ClosureEngine sees the dependency_edges edge."""
    from verbatim import derivations

    _seed(store)
    svc = BranchService(store)
    bid = svc.create(
        caller(), SID, ops=[{"effect": "archive", "claim_id": "c1"}],
    )["branch_id"]
    with store.read() as conn:
        affected = derivations.affected_by_purge(
            conn, [("claim", "c1", None)]
        )
        assert ("branch", bid, 1) in affected["derived"]
