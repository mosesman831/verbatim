"""Lifecycle transition tests (SPEC §15): table completeness, expected
versions, atomicity, erase gating, reverse_supersede semantics."""

from __future__ import annotations

import pytest

from verbatim.core.lifecycle import (
    LifecycleMachine,
    can_transition,
    read_claim_head,
    read_intervals,
)
from verbatim.core.types import (
    ErrorCode,
    Lifecycle,
    TimeInterval,
    Precision,
    TransitionCommand,
    VerbatimError,
)

from tests.core.conftest import seed_claim, q, scope_id_of

DAY = 86_400_000_000
T0 = 1_760_000_000_000_000


def _machine(store):
    return LifecycleMachine(store)


def _apply(store, machine, cmd):
    with store.tx() as conn:
        return machine.apply(cmd, conn)


def _cmd(claim_id, rev, effect, **kw):
    return TransitionCommand(
        claim_id=claim_id,
        expected_revision=rev,
        effect=effect,
        actor_id=kw.pop("actor_id", "operator"),
        reason=kw.pop("reason", "test"),
        **kw,
    )


def _state(store, claim_id):
    with store.tx() as conn:
        return read_claim_head(conn, claim_id).state


# ---------------------------------------------------------------------------
# can_transition — pure table check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frm,effect",
    [
        ("pending", "admit"),
        ("active", "dispute"),
        ("active", "supersede"),
        ("disputed", "resolve"),
        ("active", "archive"),
        ("disputed", "archive"),
        ("superseded", "archive"),
        ("archived", "restore"),
        ("pending", "reject"),
        ("disputed", "reject"),
        ("superseded", "reverse_supersede"),
        ("pending", "erase"),
        ("active", "erase"),
        ("disputed", "erase"),
        ("superseded", "erase"),
        ("rejected", "erase"),
        ("archived", "erase"),
    ],
)
def test_can_transition_allowed(frm, effect):
    assert can_transition(Lifecycle(frm), effect) is True


@pytest.mark.parametrize(
    "frm,effect",
    [
        ("pending", "supersede"),
        ("pending", "dispute"),
        ("pending", "archive"),
        ("pending", "resolve"),
        ("pending", "restore"),
        ("pending", "reverse_supersede"),
        ("active", "admit"),
        ("active", "resolve"),
        ("active", "restore"),
        ("active", "reject"),
        ("active", "reverse_supersede"),
        ("disputed", "admit"),
        ("disputed", "supersede"),
        ("disputed", "dispute"),
        ("disputed", "restore"),
        ("disputed", "reverse_supersede"),
        ("superseded", "restore"),
        ("superseded", "admit"),
        ("superseded", "supersede"),
        ("rejected", "admit"),
        ("rejected", "restore"),
        ("rejected", "archive"),
        ("archived", "admit"),
        ("archived", "archive"),
        ("erased", "admit"),
        ("erased", "restore"),
        ("erased", "erase"),
        ("erased", "archive"),
    ],
)
def test_can_transition_forbidden(frm, effect):
    assert can_transition(Lifecycle(frm), effect) is False


def test_can_transition_unknown_effect():
    assert can_transition(Lifecycle.ACTIVE, "explode") is False
    assert can_transition(Lifecycle.ACTIVE, "") is False


# ---------------------------------------------------------------------------
# admit
# ---------------------------------------------------------------------------


def test_admit_pending_with_evidence(store, scope):
    cid, rev = seed_claim(store, scope, state="pending")
    seq = _apply(store, _machine(store), _cmd(cid, rev, "admit", reason="approved"))
    assert _state(store, cid) == Lifecycle.ACTIVE
    with store.tx() as conn:
        head = read_claim_head(conn, cid)
        assert head.revision == rev + 1
        assert head.recorded_from == seq
        old = conn.execute(
            "SELECT recorded_until FROM claim_revisions WHERE claim_id=? AND revision=?",
            (cid, rev),
        ).fetchone()[0]
        assert old == seq
        # the audit event exists
        ev = conn.execute("SELECT kind, actor_id FROM events WHERE event_seq=?", (seq,)).fetchone()
        assert ev[0] == "claim_transition" and ev[1] == "operator"


def test_admit_requires_evidence(store, scope):
    cid, rev = seed_claim(store, scope, state="pending", with_evidence=False)
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(cid, rev, "admit"))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION


def test_missing_claim_is_indistinguishable(store):
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd("nonexistent", 1, "admit"))
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_stale_expected_revision(store, scope):
    cid, rev = seed_claim(store, scope, state="pending")
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(cid, rev + 5, "admit"))
    assert ei.value.code == ErrorCode.STALE_PROPOSAL
    # nothing changed
    assert _state(store, cid) == Lifecycle.PENDING


# ---------------------------------------------------------------------------
# dispute / resolve
# ---------------------------------------------------------------------------


def test_dispute_requires_conflict_edge(store, scope):
    cid, rev = seed_claim(store, scope, state="active")
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(cid, rev, "dispute"))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION


def test_dispute_with_conflict_edge(store, scope):
    cid, rev = seed_claim(store, scope, state="active")
    other, _ = seed_claim(store, scope, state="active", object_text="vscode")
    import verbatim.storage.repos as R

    with store.tx() as conn:
        R.EdgesRepo(store).add(
            conn, scope_id_of(store, scope), "claim", cid, "claim", other, "conflicts_with"
        )
    _apply(store, _machine(store), _cmd(cid, rev, "dispute"))
    assert _state(store, cid) == Lifecycle.DISPUTED


def test_resolve_disputed(store, scope):
    cid, rev = seed_claim(store, scope, state="disputed")
    _apply(store, _machine(store), _cmd(cid, rev, "resolve", reason="operator decided"))
    assert _state(store, cid) == Lifecycle.ACTIVE


# ---------------------------------------------------------------------------
# supersede + reverse_supersede
# ---------------------------------------------------------------------------


def test_supersede_requires_successor(store, scope):
    cid, rev = seed_claim(store, scope, state="active")
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(cid, rev, "supersede"))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION


def test_supersede_missing_successor(store, scope):
    cid, rev = seed_claim(store, scope, state="active")
    with pytest.raises(VerbatimError) as ei:
        _apply(
            store,
            _machine(store),
            _cmd(cid, rev, "supersede", successor_claim_id="ghost"),
        )
    assert ei.value.code == ErrorCode.NOT_FOUND_OR_FORBIDDEN


def test_supersede_successor_in_bad_state(store, scope):
    cid, rev = seed_claim(store, scope, state="active")
    dead, _ = seed_claim(store, scope, state="erased", object_text="x")
    with pytest.raises(VerbatimError) as ei:
        _apply(
            store,
            _machine(store),
            _cmd(cid, rev, "supersede", successor_claim_id=dead),
        )
    assert ei.value.code == ErrorCode.INVALID_TRANSITION


def test_supersede_writes_edge_and_truncates(store, scope):
    iv = TimeInterval(from_us=T0, until_us=None, precision=Precision.INSTANT, basis="asserted_current_at")
    cid, rev = seed_claim(store, scope, state="active", intervals=[iv])
    succ, _ = seed_claim(store, scope, state="active", object_text="helix")
    cut = TimeInterval(from_us=T0 + 5 * DAY, until_us=None, basis="explicit")
    _apply(
        store,
        _machine(store),
        _cmd(cid, rev, "supersede", successor_claim_id=succ, interval=cut),
    )
    assert _state(store, cid) == Lifecycle.SUPERSEDED
    with store.tx() as conn:
        head = read_claim_head(conn, cid)
        ivs = read_intervals(conn, cid, head.revision)
        assert len(ivs) == 1
        assert ivs[0].from_us == T0 and ivs[0].until_us == T0 + 5 * DAY
        edge = conn.execute(
            "SELECT source_id, target_id, edge_type, retired_event FROM edges"
            " WHERE edge_type='supersedes'"
        ).fetchone()
        assert edge == (succ, cid, "supersedes", None)


def test_supersede_cycle_rejected(store, scope):
    a, ra = seed_claim(store, scope, state="active")
    b, rb = seed_claim(store, scope, state="active", object_text="b")
    _apply(store, _machine(store), _cmd(a, ra, "supersede", successor_claim_id=b))
    # A is now superseded-by-B. Trying to supersede B by A would close the
    # cycle — the successor-state check (A is superseded) blocks it, and the
    # path check in the machine is the second line of defense.
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(b, rb, "supersede", successor_claim_id=a))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION
    assert _state(store, b) == Lifecycle.ACTIVE


def test_reverse_supersede_reopens_intervals(store, scope):
    iv = TimeInterval(from_us=T0, until_us=None, precision=Precision.INSTANT, basis="asserted_current_at")
    cid, rev = seed_claim(store, scope, state="active", intervals=[iv])
    succ, _ = seed_claim(store, scope, state="active", object_text="helix")
    cut = TimeInterval(from_us=T0 + 5 * DAY, until_us=None, basis="explicit")
    _apply(store, _machine(store), _cmd(cid, rev, "supersede", successor_claim_id=succ, interval=cut))
    head = None
    with store.tx() as conn:
        head = read_claim_head(conn, cid)
    # reverse: restores exactly the prior revision's intervals
    _apply(
        store,
        _machine(store),
        _cmd(cid, head.revision, "reverse_supersede", successor_claim_id=succ, reason="mistake"),
    )
    assert _state(store, cid) == Lifecycle.ACTIVE
    with store.tx() as conn:
        head = read_claim_head(conn, cid)
        ivs = read_intervals(conn, cid, head.revision)
        assert len(ivs) == 1
        assert ivs[0].from_us == T0 and ivs[0].until_us is None
        edge = conn.execute(
            "SELECT retired_event FROM edges WHERE edge_type='supersedes'"
        ).fetchone()
        assert edge[0] is not None  # compensating retirement recorded


def test_reverse_supersede_forbidden_from_active(store, scope):
    cid, rev = seed_claim(store, scope, state="active")
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(cid, rev, "reverse_supersede"))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION


def test_superseded_cannot_restore(store, scope):
    cid, rev = seed_claim(store, scope, state="superseded")
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(cid, rev, "restore"))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION


# ---------------------------------------------------------------------------
# archive / restore / reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frm", ["active", "disputed", "superseded"])
def test_archive(store, scope, frm):
    cid, rev = seed_claim(store, scope, state=frm)
    _apply(store, _machine(store), _cmd(cid, rev, "archive", reason="retention"))
    assert _state(store, cid) == Lifecycle.ARCHIVED


def test_restore_archived(store, scope):
    cid, rev = seed_claim(store, scope, state="archived")
    _apply(store, _machine(store), _cmd(cid, rev, "restore", reason="revalidated"))
    assert _state(store, cid) == Lifecycle.ACTIVE


def test_restore_requires_surviving_evidence(store, scope):
    cid, rev = seed_claim(store, scope, state="archived", with_evidence=False)
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(cid, rev, "restore"))
    assert ei.value.code == ErrorCode.EVIDENCE_UNAVAILABLE


@pytest.mark.parametrize("frm", ["pending", "disputed"])
def test_reject(store, scope, frm):
    cid, rev = seed_claim(store, scope, state=frm)
    _apply(store, _machine(store), _cmd(cid, rev, "reject", reason="denied"))
    assert _state(store, cid) == Lifecycle.REJECTED


# ---------------------------------------------------------------------------
# erase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frm", ["pending", "active", "disputed", "superseded", "rejected", "archived"]
)
def test_erase_only_purge_actor(store, scope, frm):
    cid, rev = seed_claim(store, scope, state=frm)
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(cid, rev, "erase", actor_id="alice"))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION
    _apply(store, _machine(store), _cmd(cid, rev, "erase", actor_id="purge"))
    assert _state(store, cid) == Lifecycle.ERASED


def test_erased_revision_carries_no_content(store, scope):
    cid, rev = seed_claim(store, scope, state="active")
    _apply(store, _machine(store), _cmd(cid, rev, "erase", actor_id="purge"))
    with store.tx() as conn:
        head = read_claim_head(conn, cid)
        assert head.state == Lifecycle.ERASED
        assert head.object_json is None
        assert head.condition_json is None
        assert read_intervals(conn, cid, head.revision) == []
        ev = conn.execute(
            "SELECT COUNT(*) FROM claim_evidence WHERE claim_id=? AND revision=?",
            (cid, head.revision),
        ).fetchone()[0]
        assert ev == 0


def test_erased_is_terminal(store, scope):
    cid, rev = seed_claim(store, scope, state="active")
    _apply(store, _machine(store), _cmd(cid, rev, "erase", actor_id="purge"))
    head = None
    with store.tx() as conn:
        head = read_claim_head(conn, cid)
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(cid, head.revision, "restore", actor_id="purge"))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION


# ---------------------------------------------------------------------------
# atomicity — no partial changes on failure
# ---------------------------------------------------------------------------


def test_atomicity_on_write_failure(store, scope, monkeypatch):
    cid, rev = seed_claim(store, scope, state="pending")
    machine = _machine(store)

    def boom(*a, **k):
        raise RuntimeError("simulated mid-write failure")

    monkeypatch.setattr(machine._repos.claims, "add_revision", boom)
    before_events = q(store, "SELECT COUNT(*) FROM events")[0][0]
    with pytest.raises(RuntimeError):
        _apply(store, machine, _cmd(cid, rev, "admit"))
    after_events = q(store, "SELECT COUNT(*) FROM events")[0][0]
    assert after_events == before_events  # savepoint rolled the event back
    with store.tx() as conn:
        head = read_claim_head(conn, cid)
        assert head.state == Lifecycle.PENDING
        assert head.revision == rev
        assert head.recorded_until is None


def test_reason_required(store, scope):
    cid, rev = seed_claim(store, scope, state="pending")
    with pytest.raises(VerbatimError) as ei:
        _apply(store, _machine(store), _cmd(cid, rev, "admit", reason=""))
    assert ei.value.code == ErrorCode.INVALID_TRANSITION
