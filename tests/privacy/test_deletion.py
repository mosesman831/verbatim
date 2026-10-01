"""Deletion-closure tests (SPEC_V3 §17.03, §34.02, §36.02–36.09).

Chain purge, mixed-ancestry suppression, diamond dedup, ledger rows,
orphan detection, single-transaction atomicity, outside-boundary
reporting, and snapshot rollback planning — all on a real v3 ``Store``.
"""

from __future__ import annotations

import pytest

from verbatim import derivations as deriv
from verbatim.core.lifecycle import read_claim_head
from verbatim.core.types import (
    ErrorCode,
    Lifecycle,
    VerbatimError,
    new_id,
)
from verbatim.privacy import deletion
from verbatim.privacy.deletion import (
    ACTION_DELETE,
    ACTION_REVALIDATE,
    ACTION_SUPPRESS,
    execute_closure,
    plan_closure,
    verify_closure,
)
from verbatim.privacy.snapshots import rollback_plan, take_snapshot
from verbatim.purge import plan_purge
from verbatim.storage.repos_v2 import ErasureRepo
from verbatim.storage.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v3.db"))
    yield s
    s.close()


@pytest.fixture
def scope_id(store):
    sid = "scope:deletion"
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES (?, 'prof', 'owner')",
            (sid,),
        )
        conn.execute(
            "INSERT INTO scopes (scope_id, profile_id, visibility)"
            " VALUES ('scope:other', 'prof', 'owner')"
        )
    return sid


def qrows(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def qrow(conn, sql, params=()):
    rs = qrows(conn, sql, params)
    return rs[0] if rs else None


# ------------------------------------------------------------------ builders


def make_source(conn, store, sid, src_id, payload=b"sensitive body text"):
    conn.execute(
        "INSERT INTO sources (source_id, origin, source_kind, scope_id,"
        " created_us) VALUES (?, 'test', 'user_message', ?, 1)",
        (src_id, sid),
    )
    conn.execute(
        "INSERT INTO source_revisions"
        "(source_id, revision, payload, payload_hmac, event_us, captured_us,"
        " provenance) VALUES (?, 1, ?, ?, 1, 1, 'direct_user')",
        (src_id, payload, store.hmac(payload)),
    )


def make_span(conn, store, src_id, span_id, rev=1, end=10):
    # excerpt_hmac covers exactly the stored slice this span cites
    payload = bytes(
        conn.execute(
            "SELECT payload FROM source_revisions"
            " WHERE source_id = ? AND revision = ?",
            (src_id, rev),
        ).fetchone()[0]
    )
    conn.execute(
        "INSERT INTO spans (span_id, source_id, revision, start_byte,"
        " end_byte, excerpt_hmac, harvester_version)"
        " VALUES (?, ?, ?, 0, ?, ?, 'test-harvest')",
        (span_id, src_id, rev, end, store.hmac(payload[0:end])),
    )


def make_claim(conn, sid, claim_id, span_id=None, rev=1):
    conn.execute(
        "INSERT INTO claims (claim_id, scope_id, predicate, created_event)"
        " VALUES (?, ?, 'met', 1)",
        (claim_id, sid),
    )
    conn.execute(
        "INSERT INTO claim_revisions (claim_id, revision, state, object_json,"
        " recorded_from) VALUES (?, ?, 'active', '{\"text\":\"x\"}', 1)",
        (claim_id, rev),
    )
    if span_id is not None:
        conn.execute(
            "INSERT INTO claim_evidence (claim_id, revision, span_id)"
            " VALUES (?, ?, ?)",
            (claim_id, rev, span_id),
        )


def make_procedure(conn, sid, pid, state="active"):
    conn.execute(
        "INSERT INTO procedures (procedure_id, scope_id, task_label, state)"
        " VALUES (?, ?, 'do-the-thing', ?)",
        (pid, sid, state),
    )


def make_observation(conn, sid, oid):
    conn.execute(
        "INSERT INTO observations (observation_id, scope_id, text)"
        " VALUES (?, ?, 'observed pattern')",
        (oid, sid),
    )


def make_episode(conn, sid, eid):
    conn.execute(
        "INSERT INTO episodes (episode_id, scope_id) VALUES (?, ?)",
        (eid, sid),
    )


def edge(conn, child, parent, sid, seq):
    deriv.record_edge(conn, child, parent, "harvester", "producer-1", sid, seq)


def actions_of(plan):
    return {d.ref: d.action for d in plan.derived}


# ------------------------------------------------------------- chain purge


def test_chain_span_claim_procedure_delete(store, scope_id):
    """span → claim_rev → procedure: purging the span deletes the chain."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        make_claim(conn, scope_id, "c1", "sp1")
        make_procedure(conn, scope_id, "p1")
        edge(conn, ("claim", "c1", 1), ("span", "sp1", 1), scope_id, 1)
        edge(conn, ("procedure", "p1", 1), ("claim", "c1", 1), scope_id, 2)

        plan = plan_closure(conn, [("span", "sp1", 1)])
        assert plan.scope_id == scope_id
        assert plan.roots == (("span", "sp1", 1),)
        acts = actions_of(plan)
        assert acts[("claim", "c1", 1)] == ACTION_DELETE
        assert acts[("procedure", "p1", 1)] == ACTION_DELETE

        receipt = execute_closure(conn, store, plan, "purge-chain-1")
        assert receipt["state"] == "completed"
        assert receipt["verification"]["closed"] is True
        assert receipt["ledger_rows"] == 3  # span + claim + procedure

        # physical effects: revision bytes emptied, procedure row gone,
        # claim driven to erased with scrubbed content
        rev = qrow(
            conn,
            "SELECT payload FROM source_revisions WHERE source_id = 'src1'",
        )
        assert bytes(rev["payload"]) == b""
        assert qrow(
            conn, "SELECT 1 AS x FROM procedures WHERE procedure_id = 'p1'"
        ) is None
        head = read_claim_head(conn, "c1")
        assert head is not None and head.state == Lifecycle.ERASED
        assert qrow(
            conn,
            "SELECT object_json FROM claim_revisions WHERE claim_id = 'c1'"
            " AND object_json IS NOT NULL",
        ) is None
        # claim_evidence rows citing the purged span are stripped
        assert qrow(
            conn, "SELECT 1 AS x FROM claim_evidence WHERE span_id = 'sp1'"
        ) is None
        # the derivation graph itself is closed (V3-36.02)
        assert qrow(conn, "SELECT 1 AS x FROM derivations") is None
        # opaque tombstones — one per purged object (V3-36.03)
        er = ErasureRepo(store)
        assert er.is_erased(conn, scope_id, "span", "sp1")
        assert er.is_erased(conn, scope_id, "claim", "c1")
        assert er.is_erased(conn, scope_id, "procedure", "p1")
        # the purge registry row completed under our purge_id
        assert qrow(
            conn, "SELECT state FROM purges WHERE purge_id = 'purge-chain-1'"
        )["state"] == "completed"


def test_verify_after_execute_is_closed(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        make_claim(conn, scope_id, "c1", "sp1")
        edge(conn, ("claim", "c1", 1), ("span", "sp1", 1), scope_id, 1)
        plan = plan_closure(conn, [("span", "sp1", 1)])
        execute_closure(conn, store, plan, "purge-v")
        res = verify_closure(conn, plan)
        assert res == {"closed": True, "orphans": []}


# ----------------------------------------------------------- mixed ancestry


def test_mixed_ancestry_suppresses_then_revalidates(store, scope_id):
    """procedure(spanA+spanB) and observation(procedure): purge only A."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "srcA")
        make_source(conn, store, scope_id, "srcB")
        make_span(conn, store, "srcA", "spA")
        make_span(conn, store, "srcB", "spB")
        make_procedure(conn, scope_id, "p1")
        make_observation(conn, scope_id, "o1")
        edge(conn, ("procedure", "p1", 1), ("span", "spA", 1), scope_id, 1)
        edge(conn, ("procedure", "p1", 1), ("span", "spB", 1), scope_id, 2)
        edge(conn, ("observation", "o1", 1), ("procedure", "p1", 1), scope_id, 3)

        plan = plan_closure(conn, [("span", "spA", 1)])
        acts = actions_of(plan)
        assert acts[("procedure", "p1", 1)] == ACTION_SUPPRESS
        assert acts[("observation", "o1", 1)] == ACTION_REVALIDATE

        receipt = execute_closure(conn, store, plan, "purge-mixed")
        assert receipt["verification"]["closed"] is True
        assert receipt["ledger_rows"] == 1  # only the span is erased
        assert ErasureRepo(store).is_erased(conn, scope_id, "span", "spA")
        assert not ErasureRepo(store).is_erased(
            conn, scope_id, "procedure", "p1"
        )

        # suppressed, not deleted: row stays, recorded_until closes it
        proc = qrow(
            conn,
            "SELECT recorded_until FROM procedures WHERE procedure_id = 'p1'",
        )
        assert proc["recorded_until"] is not None
        # revalidation flag lands on the observation (§24 freshness)
        fresh = qrow(
            conn,
            "SELECT class FROM freshness"
            " WHERE object_kind = 'observation' AND object_id = 'o1'",
        )
        assert fresh["class"] == "revalidate_after"
        # edge to the surviving parent stays; edge to the purged span goes
        assert deriv.parents_of(conn, ("procedure", "p1", 1)) == [
            ("span", "spB", 1)
        ]
        assert deriv.parents_of(conn, ("observation", "o1", 1)) == [
            ("procedure", "p1", 1)
        ]


# --------------------------------------------------------------- the diamond


def test_diamond_both_children_in_closure(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        make_claim(conn, scope_id, "c1", "sp1")
        make_claim(conn, scope_id, "c2", "sp1")
        make_procedure(conn, scope_id, "g")
        edge(conn, ("claim", "c1", 1), ("span", "sp1", 1), scope_id, 1)
        edge(conn, ("claim", "c2", 1), ("span", "sp1", 1), scope_id, 2)
        edge(conn, ("procedure", "g", 1), ("claim", "c1", 1), scope_id, 3)
        edge(conn, ("procedure", "g", 1), ("claim", "c2", 1), scope_id, 4)

        plan = plan_closure(conn, [("span", "sp1", 1)])
        acts = actions_of(plan)
        assert acts == {
            ("claim", "c1", 1): ACTION_DELETE,
            ("claim", "c2", 1): ACTION_DELETE,
            ("procedure", "g", 1): ACTION_DELETE,
        }
        execute_closure(conn, store, plan, "purge-diamond")
        assert verify_closure(conn, plan)["closed"] is True
        assert qrow(conn, "SELECT 1 AS x FROM derivations") is None


# ------------------------------------------------------- orphan detection


def test_verify_catches_deliberately_broken_deletion(store, scope_id):
    """Manually scrub only the parent — children/edges become orphans."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        make_claim(conn, scope_id, "c1", "sp1")
        make_procedure(conn, scope_id, "p1")
        edge(conn, ("claim", "c1", 1), ("span", "sp1", 1), scope_id, 1)
        edge(conn, ("procedure", "p1", 1), ("claim", "c1", 1), scope_id, 2)
        plan = plan_closure(conn, [("span", "sp1", 1)])

        # broken deletion: bytes gone, but derived rows + edges untouched
        conn.execute(
            "UPDATE source_revisions SET payload = X'' WHERE source_id = 'src1'"
        )
        res = verify_closure(conn, plan)
        assert res["closed"] is False
        reasons = {o["reason"] for o in res["orphans"]}
        assert "row_remains" in reasons          # claim/procedure still live
        assert "edge_to_purged_parent" in reasons  # claim → span edge remains
        assert "reference_remains" in reasons    # claim_evidence still cites it
        orphan_refs = {o["ref"] for o in res["orphans"]}
        assert ("claim", "c1", 1) in orphan_refs
        assert ("procedure", "p1", 1) in orphan_refs


def test_verify_flags_missing_suppression_marker(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "srcA")
        make_source(conn, store, scope_id, "srcB")
        make_span(conn, store, "srcA", "spA")
        make_span(conn, store, "srcB", "spB")
        make_procedure(conn, scope_id, "p1")
        edge(conn, ("procedure", "p1", 1), ("span", "spA", 1), scope_id, 1)
        edge(conn, ("procedure", "p1", 1), ("span", "spB", 1), scope_id, 2)
        plan = plan_closure(conn, [("span", "spA", 1)])
        # broken: bytes gone, but the suppressed procedure marker never set
        conn.execute(
            "UPDATE source_revisions SET payload = X'' WHERE source_id = 'srcA'"
        )
        conn.execute(
            "DELETE FROM derivations WHERE parent_id = 'spA'"
        )
        res = verify_closure(conn, plan)
        assert res["closed"] is False
        assert any(
            o["reason"] == "suppression_missing" for o in res["orphans"]
        )


# ------------------------------------------------------------- atomicity


def test_mid_execution_failure_leaves_no_partial_deletion(
    store, scope_id, monkeypatch
):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        make_claim(conn, scope_id, "c1", "sp1")
        make_procedure(conn, scope_id, "p1")
        edge(conn, ("claim", "c1", 1), ("span", "sp1", 1), scope_id, 1)
        edge(conn, ("procedure", "p1", 1), ("claim", "c1", 1), scope_id, 2)
        plan = plan_closure(conn, [("span", "sp1", 1)])

    def _boom(*a, **k):
        raise RuntimeError("forced mid-execution failure")

    monkeypatch.setattr(deletion, "_strip_references", _boom)
    with pytest.raises(RuntimeError):
        with store.tx() as conn:
            execute_closure(conn, store, plan, "purge-atomic")
    # the whole transaction rolled back — nothing partially erased
    with store.read() as conn:
        rev = qrow(
            conn,
            "SELECT payload FROM source_revisions WHERE source_id = 'src1'",
        )
        assert bytes(rev["payload"]) == b"sensitive body text"
        assert qrow(
            conn, "SELECT 1 AS x FROM procedures WHERE procedure_id = 'p1'"
        ) is not None
        head = read_claim_head(conn, "c1")
        assert head is not None and head.state == Lifecycle.ACTIVE
        assert len(qrows(conn, "SELECT * FROM derivations")) == 2
        assert qrow(conn, "SELECT 1 AS x FROM purges") is None
        assert qrow(conn, "SELECT 1 AS x FROM erasure_ledger") is None


def test_execute_with_v2_purge_preview(store, scope_id):
    """Composition: a v2 ``plan_purge`` preview row drives execute_closure."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        make_claim(conn, scope_id, "c1", "sp1")
        edge(conn, ("claim", "c1", 1), ("span", "sp1", 1), scope_id, 1)
    preview = plan_purge(store, scope_id, [("span", "sp1")], actor="tester")
    with store.tx() as conn:
        plan = plan_closure(conn, [("span", "sp1", 1)])
        receipt = execute_closure(conn, store, plan, preview["purge_id"])
        assert receipt["state"] == "completed"
        assert qrow(
            conn,
            "SELECT state FROM purges WHERE purge_id = ?",
            (preview["purge_id"],),
        )["state"] == "completed"


# ------------------------------------------------------------ validation


def test_plan_rejects_missing_and_cross_scope(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        with pytest.raises(VerbatimError) as ei:
            plan_closure(conn, [("span", "nope", 1)])
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED
        with pytest.raises(VerbatimError) as ei:
            plan_closure(conn, [("mystery", "x", 1)])
        assert ei.value.code == ErrorCode.VALIDATION

    # foreign-scope object as a second root → single-scope rule
    with store.tx() as conn:
        payload_f = b"\xaa"
        conn.execute(
            "INSERT INTO sources (source_id, origin, source_kind, scope_id,"
            " created_us) VALUES ('srcF', 'test', 'user_message',"
            " 'scope:other', 1)"
        )
        conn.execute(
            "INSERT INTO source_revisions"
            "(source_id, revision, payload, payload_hmac, event_us,"
            " captured_us, provenance)"
            " VALUES ('srcF', 1, ?, ?, 1, 1, 'direct_user')",
            (payload_f, store.hmac(payload_f)),
        )
        conn.execute(
            "INSERT INTO spans (span_id, source_id, revision, start_byte,"
            " end_byte, excerpt_hmac, harvester_version)"
            " VALUES ('spF', 'srcF', 1, 0, 1, ?, 'h')",
            (store.hmac(payload_f[0:1]),),
        )
        with pytest.raises(VerbatimError) as ei:
            plan_closure(conn, [("span", "sp1", 1), ("span", "spF", 1)])
        assert ei.value.code == ErrorCode.VALIDATION


def test_plan_empty_targets_rejected(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            plan_closure(conn, [])
        assert ei.value.code == ErrorCode.VALIDATION


def test_execute_rejects_stale_plan(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        plan = plan_closure(conn, [("span", "sp1", 1)])
    # plan computed; the object disappears before execution
    with store.tx() as conn:
        conn.execute("DELETE FROM claim_evidence WHERE span_id = 'sp1'")
        conn.execute("DELETE FROM spans WHERE span_id = 'sp1'")
        with pytest.raises(VerbatimError) as ei:
            execute_closure(conn, store, plan, "purge-stale")
        assert ei.value.code == ErrorCode.STALE_PROPOSAL


# --------------------------------------------------------- outside boundary


def test_propagated_copies_reported_outside_boundary(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        conn.execute(
            "INSERT INTO propagations"
            "(propagation_id, scope_id, object_kind, object_id, revision,"
            " recipient_id, verbs_json, purpose, epoch, created_us)"
            " VALUES ('prop1', ?, 'span', 'sp1', 1, 'agent-remote',"
            " '[\"read\"]', 'share', 0, 1)",
            (scope_id,),
        )
        plan = plan_closure(conn, [("span", "sp1", 1)])
        kinds = {o["reason"] for o in plan.outside_boundary}
        assert "propagated_copy" in kinds
        receipt = execute_closure(conn, store, plan, "purge-prop")
        # revocation is marked locally; remote erasure is disclosed, not claimed
        prop = qrow(
            conn,
            "SELECT revoked_seq FROM propagations WHERE propagation_id = 'prop1'",
        )
        assert prop["revoked_seq"] is not None
        assert any(
            o["reason"] == "propagated_copy"
            for o in receipt["outside_boundary"]
        )


def test_foreign_scope_descendant_is_outside_boundary(store, scope_id):
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        make_claim(conn, "scope:other", "cF", "sp1")
        edge(conn, ("claim", "cF", 1), ("span", "sp1", 1), "scope:other", 1)
        plan = plan_closure(conn, [("span", "sp1", 1)])
        reasons = {o["reason"] for o in plan.outside_boundary}
        assert "foreign_scope" in reasons
        receipt = execute_closure(conn, store, plan, "purge-foreign")
        assert receipt["verification"]["closed"] is True
        # the foreign claim survives — local deletion cannot reach it — but
        # its edge into erased material is gone
        assert qrow(
            conn, "SELECT 1 AS x FROM claims WHERE claim_id = 'cF'"
        ) is not None
        assert qrow(conn, "SELECT 1 AS x FROM derivations") is None


def test_unresolvable_descendant_kind_outside_boundary(store, scope_id):
    """Descendants outside the purge's scope are disclosed, not silently
    pulled in — every registered derivation kind is resolvable now, so
    the foreign-scope boundary is the reachable case."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        make_observation(conn, "scope:other", "o1")
        edge(conn, ("observation", "o1", 1), ("span", "sp1", 1),
             "scope:other", 1)
        plan = plan_closure(conn, [("span", "sp1", 1)])
        reasons = {o["reason"] for o in plan.outside_boundary}
        assert "foreign_scope" in reasons
        receipt = execute_closure(conn, store, plan, "purge-unk")
        assert receipt["verification"]["closed"] is True
        # the foreign-scope object survives untouched
        assert qrow(
            conn, "SELECT 1 AS x FROM observations WHERE observation_id='o1'"
        ) is not None


# ---------------------------------------------------------------- episodes


def test_episode_and_transition_closure(store, scope_id):
    """trajectory evidence → episode → transition → procedure chain."""
    with store.tx() as conn:
        make_source(conn, store, scope_id, "src1")
        make_span(conn, store, "src1", "sp1")
        make_episode(conn, scope_id, "e1")
        conn.execute(
            "INSERT INTO transitions (transition_id, episode_id, scope_id, ord)"
            " VALUES ('t1', 'e1', ?, 1)",
            (scope_id,),
        )
        make_procedure(conn, scope_id, "p1")
        edge(conn, ("episode", "e1", 1), ("span", "sp1", 1), scope_id, 1)
        edge(conn, ("transition", "t1", 1), ("episode", "e1", 1), scope_id, 2)
        edge(conn, ("procedure", "p1", 1), ("transition", "t1", 1), scope_id, 3)

        plan = plan_closure(conn, [("span", "sp1", 1)])
        acts = actions_of(plan)
        assert acts[("episode", "e1", 1)] == ACTION_DELETE
        assert acts[("transition", "t1", 1)] == ACTION_DELETE
        assert acts[("procedure", "p1", 1)] == ACTION_DELETE
        receipt = execute_closure(conn, store, plan, "purge-exp")
        assert receipt["verification"]["closed"] is True
        assert qrow(conn, "SELECT 1 AS x FROM episodes") is None
        assert qrow(conn, "SELECT 1 AS x FROM transitions") is None
        assert qrow(conn, "SELECT 1 AS x FROM procedures") is None


# ---------------------------------------------------------------- snapshots


def test_snapshot_and_rollback_plan(store, scope_id):
    with store.tx() as conn:
        edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, 1)
        edge(conn, ("claim", "c2", 1), ("span", "s1", 1), scope_id, 2)
        snap = take_snapshot(conn, scope_id)
        edge(conn, ("procedure", "p1", 1), ("claim", "c1", 1), scope_id, 3)
        edge(conn, ("procedure", "p2", 1), ("claim", "c2", 1), scope_id, 4)

        plan = rollback_plan(conn, snap)
        got = {(c["kind"], c["object_id"], c["revision"]) for c in plan["candidates"]}
        # only objects created after the watermark
        assert got == {("procedure", "p1", 1), ("procedure", "p2", 1)}
        assert plan["seq"] == 2
        assert plan["validation"] == "unvalidated"


def test_rollback_plan_empty_when_nothing_new(store, scope_id):
    with store.tx() as conn:
        edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, 1)
        snap = take_snapshot(conn, scope_id)
        plan = rollback_plan(conn, snap)
        assert plan["candidates"] == []


def test_rollback_plan_unknown_snapshot(store, scope_id):
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as ei:
            rollback_plan(conn, "no-such-snapshot")
        assert ei.value.code == ErrorCode.NOT_FOUND_OR_UNAUTHORIZED


def test_snapshot_digest_is_deterministic(store, scope_id):
    with store.tx() as conn:
        edge(conn, ("claim", "c1", 1), ("span", "s1", 1), scope_id, 1)
        s1 = take_snapshot(conn, scope_id)
        s2 = take_snapshot(conn, scope_id)
        rows = qrows(
            conn,
            "SELECT snapshot_id, seq, hex(digest) AS d FROM learning_snapshots"
            " ORDER BY created_us",
        )
        assert len(rows) == 2
        # same graph → same digest, distinct snapshot ids
        assert rows[0]["d"] == rows[1]["d"]
        assert rows[0]["snapshot_id"] != rows[1]["snapshot_id"]
        del s1, s2
