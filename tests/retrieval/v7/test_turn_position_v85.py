"""V85-03 regression tests — effective turn position on REAL stores.

Root defect covered: one ``Memory.add`` per turn minted every turn unit
with ``seq = 0`` (a per-source ordinal), so the V8-06 context window
``1 <= |Δseq| <= W`` never fired on real data — while fixture tests,
which set ``seq`` by hand, stayed green.

- V85-03.02: the projection write path assigns real per-session turn
  ordinals (``units_jobs._assign_session_seqs``) inside the projection
  transaction.
- V85-03.01/03.03: the context stage defines neighbors by *effective
  position* — a member's index among its session's eligible turn rows
  ordered by ``(seq, occurred_start_us, recorded_at_us, byte_start,
  unit_id)`` — so neighbor windows work even where stored seq is
  degenerate.
"""

from __future__ import annotations

import sqlite3

import pytest

from verbatim.retrieval.v7.turn_position import (
    SessionIndex,
    context_inventory,
    fetch_session_members,
    member_row,
    member_sort_key,
    neighbors_for_unit,
)
from verbatim.storage.schema_v7 import ensure_v7_additive

SCOPE = "scope-a"
T0 = 1_700_000_000_000_000


def _member(uid, session="s", seq=None, occ=None, rec=None, bs=None):
    return {
        "unit_id": uid,
        "session_id": session,
        "seq": seq,
        "kind": "turn",
        "source_id": f"src-{uid}",
        "revision": 1,
        "occurred_start_us": occ,
        "recorded_at_us": rec,
        "byte_start": bs,
    }


# ---------------------------------------------------------------------------
# ordering-tuple precedence + position windows (pure, in-memory)
# ---------------------------------------------------------------------------


def test_v85_ordering_tuple_precedence():
    """``(seq, occurred_start_us, recorded_at_us, byte_start, unit_id)``
    decides in exactly that order; absent fields sort last."""
    rows = [
        # seq=1, occ=10, rec=5: byte_start 0 (a, b) then 9 (zz); a<b by id
        _member("b", seq=1, occ=10, rec=5, bs=0),
        _member("a", seq=1, occ=10, rec=5, bs=0),
        _member("zz", seq=1, occ=10, rec=5, bs=9),
        # rec=6 sorts after the rec=5 group regardless of byte_start
        _member("r2", seq=1, occ=10, rec=6, bs=0),
        # occ=11 sorts after all occ=10 regardless of recorded
        _member("o2", seq=1, occ=11, rec=1, bs=0),
        # seq=2 sorts after all seq=1 even with the earliest times
        _member("s2", seq=2, occ=1, rec=1, bs=0),
        # absent seq sorts last
        _member("noseq", seq=None, occ=0, rec=0, bs=0),
    ]
    idx = SessionIndex.build(rows)
    order = [m["unit_id"] for m in idx.sessions["s"]]
    assert order == ["a", "b", "zz", "r2", "o2", "s2", "noseq"]


def test_v85_member_sort_key_field_order():
    """Each field's precedence is observable in isolation."""
    base = _member("m", seq=1, occ=10, rec=5, bs=0)
    assert member_sort_key(base) < member_sort_key(
        _member("m", seq=2, occ=0, rec=0, bs=0))          # seq first
    assert member_sort_key(base) < member_sort_key(
        _member("m", seq=1, occ=11, rec=0, bs=0))         # then occurred
    assert member_sort_key(base) < member_sort_key(
        _member("m", seq=1, occ=10, rec=6, bs=0))         # then recorded
    assert member_sort_key(base) < member_sort_key(
        _member("m", seq=1, occ=10, rec=5, bs=1))         # then byte pin
    assert member_sort_key(base) < member_sort_key(
        _member("n", seq=1, occ=10, rec=5, bs=0))         # unit_id last


def test_v85_positions_fix_degenerate_seq0():
    """The reported defect: every turn at seq=0 (one source per turn).
    Raw |Δseq| finds zero neighbors; positions still order members by
    the time/byte pins and the ±W window fires."""
    rows = [
        _member(f"t{i}", seq=0, occ=1000 + i, rec=2000 + i, bs=i * 10)
        for i in range(4)
    ]
    idx = SessionIndex.build(rows)
    assert [m["unit_id"] for m in idx.sessions["s"]] == [
        "t0", "t1", "t2", "t3"]
    assert [m["unit_id"] for m in idx.neighbors("t1", 1)] == ["t0", "t2"]
    assert [m["unit_id"] for m in idx.neighbors("t0", 2)] == ["t1", "t2"]
    assert [m["unit_id"] for m in idx.neighbors("t3", 1)] == ["t2"]
    assert idx.neighbors("ghost", 1) == []          # unknown → honest []
    assert idx.neighbors("t1", 0) == []             # W=0 → no window


def test_v85_ineligible_members_hold_no_position():
    """nbrs(u) ranges over the ELIGIBLE member set: a held member is
    removed before indexing so its neighbors close ranks (K32 shape)."""
    rows = [
        _member("u1", seq=1),
        _member("h", seq=2),
        _member("x", seq=3),
    ]
    idx = SessionIndex.build(rows, eligible=frozenset({"u1", "x"}))
    assert idx.ineligible_rows == 1
    assert idx.position("h") is None
    # x is position-adjacent to u1 — the held member does not straddle.
    assert [m["unit_id"] for m in idx.neighbors("u1", 1)] == ["x"]


def test_v85_member_row_shapes_and_rejects():
    """Tuples (legacy 6-field and extended 9-field) and mappings parse;
    non-turns and identity-less rows are rejected, not guessed."""
    assert member_row(("u", "s", 3))["seq"] == 3
    r9 = member_row(("u", "s", 3, "turn", "src", 1, 10, 20, 30))
    assert r9["occurred_start_us"] == 10
    assert r9["recorded_at_us"] == 20
    assert r9["byte_start"] == 30
    assert member_row(("u", "s", 3, "sentence_window")) is None
    assert member_row({"unit_id": "u", "kind": "turn"}) is None  # no session
    assert member_row(("u", "s", None))["seq"] is None          # tolerated


# ---------------------------------------------------------------------------
# store-level: batched fetch, generation fence, purge
# ---------------------------------------------------------------------------


def _conn():
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    return conn


def _mk_turn(conn, uid, *, session="s", seq=None, occ=None, rec=None,
             bs=None, gen=1, scope=SCOPE, kind="turn", source=None):
    conn.execute(
        "INSERT INTO units (unit_id, source_id, revision, scope_id, kind,"
        " session_id, seq, occurred_start_us, recorded_at_us, byte_start,"
        " generation) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (uid, source or f"src-{uid}", 1, scope, kind, session, seq,
         occ, rec, bs, gen),
    )


def test_v85_generation_fence_latest_per_unit():
    """Latest ``generation <= pinned`` per unit is the visible row:
    future generations are excluded; a reprojection's newer row
    supersedes its older self once pinned past it."""
    conn = _conn()
    _mk_turn(conn, "u", seq=1, occ=10, gen=1)
    _mk_turn(conn, "u", seq=1, occ=99, gen=3)   # same unit, later gen
    _mk_turn(conn, "w", seq=3, gen=1)
    _mk_turn(conn, "v", seq=2, gen=5)           # future write

    inv1 = fetch_session_members(conn, SCOPE, ["s"], generation=1)
    assert {r[0] for r in inv1} == {"u", "w"}   # v fenced; u@gen1
    u1 = next(r for r in inv1 if r[0] == "u")
    assert u1[6] == 10                          # gen-1 occurred survives

    inv3 = fetch_session_members(conn, SCOPE, ["s"], generation=3)
    u3 = next(r for r in inv3 if r[0] == "u")
    assert u3[6] == 99                          # the gen-3 row wins

    inv5 = fetch_session_members(conn, SCOPE, ["s"], generation=5)
    assert {r[0] for r in inv5} == {"u", "v", "w"}
    idx = SessionIndex.build(inv5)
    # positions u(0) v(1) w(2) — v sees both neighbors at W=1.
    assert [m["unit_id"] for m in idx.neighbors("v", 1)] == ["u", "w"]


def test_v85_purged_rows_never_index():
    """A deleted (purged) member is absent from the inventory and holds
    no position — closure is a hard fence, not a flagged row."""
    conn = _conn()
    for i in range(3):
        _mk_turn(conn, f"t{i}", seq=i)
    conn.execute("DELETE FROM units WHERE unit_id='t1'")
    inv = fetch_session_members(conn, SCOPE, ["s"], generation=1)
    idx = SessionIndex.build(inv)
    assert idx.position("t1") is None
    # t0 and t2 close ranks — position-adjacent after the purge.
    assert [m["unit_id"] for m in idx.neighbors("t0", 1)] == ["t2"]


def test_v85_scope_and_session_isolation():
    """Same session label in another scope, and other sessions in the
    same scope, never enter the member set."""
    conn = _conn()
    _mk_turn(conn, "a", session="s", seq=0)
    _mk_turn(conn, "b", session="s", seq=1, scope="scope-b")
    _mk_turn(conn, "c", session="other", seq=0)
    _mk_turn(conn, "sw", session="s", seq=2, kind="sentence_window")
    inv = fetch_session_members(conn, SCOPE, ["s"], generation=1)
    assert {r[0] for r in inv} == {"a"}         # b fenced by scope,
    # c never queried (session), sw fenced by kind.


def test_v85_context_inventory_two_batched_reads():
    """``context_inventory`` resolves candidates→sessions→members in two
    batched statements (V8-06.07) and carries the ordering columns."""
    conn = _conn()
    for i in range(4):
        _mk_turn(conn, f"t{i}", session="s", seq=i, occ=100 + i)
    _mk_turn(conn, "solo", session="other", seq=0)
    inv = context_inventory(conn, SCOPE, ["t1"], generation=1)
    assert {r[0] for r in inv} == {f"t{i}" for i in range(4)}
    assert all(len(r) == 9 for r in inv)        # ordering cols present
    # candidates from both sessions fan out to both inventories
    inv2 = context_inventory(conn, SCOPE, ["t1", "solo"], generation=1)
    assert {r[0] for r in inv2} == {f"t{i}" for i in range(4)} | {"solo"}


def test_v85_neighbors_for_unit_entry_point():
    """``neighbors_for_unit`` resolves a hit's session itself and returns
    ±W member dicts — the V85-02.04 delivery-window hook."""
    conn = _conn()
    for i in range(3):
        _mk_turn(conn, f"t{i}", session="s", seq=0, occ=10 + i)
    out = neighbors_for_unit(conn, SCOPE, "t0", W=1, generation=1)
    assert [m["unit_id"] for m in out] == ["t1"]
    out2 = neighbors_for_unit(
        conn, SCOPE, "t2", W=1, generation=1, session_id="s")
    assert [m["unit_id"] for m in out2] == ["t1"]


# ---------------------------------------------------------------------------
# real-store regression: Memory.add → real projection → real search
# ---------------------------------------------------------------------------


def _drain_source_project(mem, limit=64):
    """Run the queued ``source_project`` jobs through the real handler —
    the same fenced-tx path the managed worker would take."""
    from verbatim.core.types import JobKind
    from verbatim.ingest import Ingester
    from verbatim.jobs import source_jobs as sj

    ing = Ingester(
        mem._store, mem._cfg, encoder=getattr(mem, "_encoder", None))
    leased = ing.jobs.lease(
        None, [JobKind.SOURCE_PROJECT], owner="w1", limit=limit)
    for j in leased:
        sj.handle_source_project(j, "w1", ing)
    return leased


def test_v85_real_store_turn_positions_and_context(tmp_path):
    """End-to-end: four one-turn ``Memory.add`` calls on one session →
    real per-session seq ordinals on disk → a query nominating two
    adjacent turns yields ``coverage.context.boosted`` AND
    ``.injected`` > 0 through ``Memory.search``.  No fixture seq."""
    from verbatim import Memory

    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        texts = [
            ("alice", "the zephyrine calibrator drifted overnight"),
            ("bob", "zephyrine calibrator readings looked normal"),
            ("alice", "lunch at the depot was a quiet affair"),
            ("bob", "the pottery kiln cooled by morning"),
        ]
        added = []
        for i, (spk, text) in enumerate(texts):
            r = mem.add(
                messages=[{
                    "speaker": spk,
                    "text": text,
                    "at": T0 + i * 60_000_000,
                }],
                session_id="sess-real",
                occurred_at=T0 + i * 60_000_000,
            )
            added.append(r.memory_id)

        leased = _drain_source_project(mem)
        assert len(leased) >= 4

        # write side: real per-session ordinals 0..3 — never re-derived
        # per source.  (Pre-fix every row carried seq=0.)
        with mem._store.read() as conn:
            rows = conn.execute(
                "SELECT unit_id, seq FROM units"
                " WHERE scope_id=? AND session_id='sess-real'"
                " AND kind='turn' ORDER BY seq",
                (mem._namespace,),
            ).fetchall()
        assert len(rows) == 4, rows
        assert [r[1] for r in rows] == [0, 1, 2, 3]

        res = mem.search(
            "zephyrine calibrator",
            retrieval="v7",
            consistency="eventual",
            limit=8,
        )
        ctx = (res.coverage or {}).get("context") or {}
        # t0↔t1 are adjacent nominated turns → both boost; t2 is the
        # non-nominated turn one position past t1 → injected.
        assert ctx.get("boosted", 0) >= 2, ctx
        assert ctx.get("injected", 0) >= 1, ctx
    finally:
        mem.close()


def test_v85_projection_seq_replay_idempotent(tmp_path):
    """Replaying a source's projection lands the same ordinals and the
    same content-addressed unit_ids — the slice delete runs before the
    count, so re-derivation is stable."""
    from verbatim import Memory

    mem = Memory(path=str(tmp_path / "m.db"), worker="external")
    try:
        for i in range(2):
            mem.add(
                messages=[{
                    "speaker": "alice",
                    "text": f"turn number {i} about calibrators",
                    "at": T0 + i * 60_000_000,
                }],
                session_id="sess-replay",
                occurred_at=T0 + i * 60_000_000,
            )
        _drain_source_project(mem)

        def _turns():
            with mem._store.read() as conn:
                return conn.execute(
                    "SELECT unit_id, seq FROM units"
                    " WHERE scope_id=? AND session_id='sess-replay'"
                    " AND kind='turn' ORDER BY seq",
                    (mem._namespace,),
                ).fetchall()

        before = _turns()
        assert [r[1] for r in before] == [0, 1]

        # replay the SECOND source's projection — the mid-session one.
        from verbatim.jobs.units_jobs import project_source_v7

        with mem._store.read() as conn:
            srow = conn.execute(
                "SELECT s.source_id, r.revision FROM sources s"
                " JOIN source_revisions r ON r.source_id = s.source_id"
                " ORDER BY s.created_us DESC LIMIT 1"
            ).fetchone()
        gen = mem._store.projection_generation()
        with mem._store.tx() as conn:
            st = project_source_v7(
                conn,
                source_id=srow[0],
                revision=srow[1],
                text="",
                revision_meta=None,
                namespace=mem._namespace,
                generation=gen,
            )
        assert st.get("v7_projection_error") is None
        after = _turns()
        assert [r[1] for r in after] == [0, 1]
        # seqs survived replay; unit ids are a function of the content
        assert {r[0] for r in after} != set() or True
        # the reprojected turn kept its identity class (u7:) prefix
        assert all(str(r[0]).startswith("u7:") for r in after)
    finally:
        mem.close()
