"""V8 context-propagation/injection verification — SPEC_V8 §06/§21.1,
§24 verification row, scenarios K30–K36.

The stage under test is ``verbatim/retrieval/v7/fusion.py::
propagate_context`` (the S2c stage ``rrf_fuse`` runs before lane ranks
are consumed) plus the pipeline's ``_context_inventory`` batched
neighbor read and the production purge closure for the erase scenario.
Everything runs on real ``LaneOutput``/``CandidateV7`` contract objects
and real SQLite — no mocks.

- K30: ``s'(u) = s(u) + w·max{s(v)}`` over nominated eligible neighbors;
  lane ranks recomputed before RRF.
- K31: a non-nominated same-session neighbor is injected at
  ``w·max{s(v)}`` with ``ctx_from`` set, strictly below its parent.
- K32: an ineligible neighbor is neither injected nor a propagation
  source.
- K33: source erasure sweeps the unit plane in the closure transaction;
  a search for the erased turn's unique words returns nothing. (The
  ``ctx`` doclen field is a declared-but-unadopted §19 arm —
  ``_DOCLEN_FIELDS`` excludes it — so the neighbor-``ctx`` rewrite
  clause is vacuous; the closure half is exercised for real.)
- K34: an injected turn delivers pinned to its own bytes; the neighbor's
  text appears only as decoration (``ctx_from`` / ``.context``).
- K35: no propagation/injection crosses session, scope, generation, or
  the ``W`` window; ``sentence_window`` units are never sources or
  targets.
- K36: ``coverage.context`` reports every field; the neighbor inventory
  is ≤ 2 SQL statements per request.
"""

from __future__ import annotations

import re
import sqlite3

import pytest

from verbatim.core.types_v7 import (
    BudgetClass,
    CandidateV7,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    PackItemV7,
    QueryViewV7,
    RetrievalPolicyV7,
    ScoredCandidate,
)
from verbatim.jobs import units_jobs
from verbatim.privacy import closure_v7
from verbatim.retrieval.v7 import fusion
from verbatim.retrieval.v7.fusion import propagate_context, rrf_fuse
from verbatim.retrieval.v7.pack import assemble_pack
from verbatim.retrieval.v7 import pipeline as pipeline_v7
from verbatim.storage.schema_v7 import ensure_v7_additive

SCOPE = "scope-a"
T0 = 1_700_000_000_000_000
DAY = 86_400_000_000


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _cand(
    uid: str,
    score: float,
    *,
    session: str = "s1",
    seq: int = 1,
    kind: str = "turn",
    source: str | None = None,
    revision: int = 1,
    lane: str = "lex",
) -> CandidateV7:
    return CandidateV7(
        unit_id=uid,
        source_id=source or f"src-{uid}",
        revision=revision,
        lane=lane,
        rank=0,
        raw_score=score,
        signals={"kind": kind, "session_id": session, "seq": seq},
    )


def _lane_out(cands, lane: str = "lex") -> LaneOutput:
    out = LaneOutput(lane=lane, status=LaneStatus.OK)
    for i, c in enumerate(cands, start=1):
        out.candidates.append(
            CandidateV7(
                unit_id=c.unit_id,
                source_id=c.source_id,
                revision=c.revision,
                lane=c.lane,
                rank=i,
                raw_score=c.raw_score,
                signals=dict(c.signals),
            )
        )
    return out


def _row(uid, session, seq, kind="turn", source=None, revision=1):
    """Flat-inventory neighbor row: ``(unit_id, session_id, seq, kind,
    source_id, revision)`` — one of the two shapes the pipeline's
    neighbor fetch produces."""
    return (uid, session, seq, kind, source or f"src-{uid}", revision)


def mk_qv(query: str = "q") -> QueryViewV7:
    terms = tuple(
        NormTerm(term=t, channel="text", byte_start=0, byte_end=len(t))
        for t in query.split()
    )
    return QueryViewV7(
        query=query,
        norm=NormAnalysis(analyzer_id="norm/v2", terms=terms,
                          identifiers=()),
        intent=IntentResult(
            primary=IntentClass.LOOKUP, classes=(IntentClass.LOOKUP,)
        ),
        query_time_us=T0,
    )


def _mk_ctx(conn, *, scope=SCOPE, generation=1, eligible=None):
    return LaneContextV7(
        store=conn,
        scope_id=scope,
        generation=generation,
        eligible=eligible,
        query_time_us=T0,
        profile="test",
        budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="test/policy",
            profile="test",
            lanes=(LaneName.LEX,),
            lane_weights={},
        ),
        manifest={},
    )


_UNITS_STMT = re.compile(
    r"\bFROM\s+units\b|\bINTO\s+units\b|\bUPDATE\s+units\b"
)


def _counting_trace(conn):
    counts = {"units": 0, "total": 0}

    def cb(sql):
        counts["total"] += 1
        if _UNITS_STMT.search(sql):
            counts["units"] += 1

    conn.set_trace_callback(cb)
    return counts


# ---------------------------------------------------------------------------
# K30 — propagation arithmetic + lane re-rank
# ---------------------------------------------------------------------------


def test_k30_propagation_arithmetic():
    """K30: nominated u (s=2) adjacent to nominated v (s=4), w=0.7 →
    s'(u) = 2 + 0.7·4 = 4.8, s'(v) = 4 + 0.7·2 = 5.4 — boosts read only
    the original pre-context scores (no cascading)."""
    u = _cand("u", 2.0, session="s1", seq=1)
    v = _cand("v", 4.0, session="s1", seq=2)
    outs, stats = propagate_context(
        [_lane_out([u, v])],
        neighbors=[_row("u", "s1", 1), _row("v", "s1", 2)],
        mode="combined", w=0.7, W=1,
    )
    (out,) = outs
    by_id = {c.unit_id: c for c in out.candidates}
    assert by_id["u"].raw_score == pytest.approx(4.8)
    assert by_id["v"].raw_score == pytest.approx(5.4)
    # boost provenance: original score + argmax neighbor
    assert by_id["u"].signals["pre_ctx_score"] == pytest.approx(2.0)
    assert by_id["u"].signals["ctx_from"] == "v"
    assert by_id["v"].signals["pre_ctx_score"] == pytest.approx(4.0)
    assert by_id["v"].signals["ctx_from"] == "u"
    # ranks recomputed before RRF: v outranks u post-propagation
    assert by_id["v"].rank == 1 and by_id["u"].rank == 2
    assert stats["model"] == fusion.CONTEXT_MODEL_ID
    assert stats["boosted"] == 2 and stats["injected"] == 0


def test_k30_propagation_reorders_rank():
    """K30 (recomputation): a mid-chain turn leapfrogs its weaker neighbor
    — u2 (s=2, between u1 s=4 and u3 s=10) boosts to 9.0 and takes rank
    2, so the re-rank is observable, not a no-op."""
    u1 = _cand("u1", 4.0, session="s1", seq=1)
    u2 = _cand("u2", 2.0, session="s1", seq=2)
    u3 = _cand("u3", 10.0, session="s1", seq=3)
    outs, _ = propagate_context(
        [_lane_out([u1, u2, u3])],
        neighbors=[
            _row("u1", "s1", 1), _row("u2", "s1", 2), _row("u3", "s1", 3)
        ],
        w=0.7, W=1,
    )
    (out,) = outs
    by_id = {c.unit_id: c for c in out.candidates}
    # u1 ← u2 (s=2): 4 + 1.4 = 5.4;  u2 ← max(u1,u3)=10: 2 + 7.0 = 9.0;
    # u3 ← u2 (s=2): 10 + 1.4 = 11.4 — scores read pre-context only.
    assert by_id["u1"].raw_score == pytest.approx(5.4)
    assert by_id["u2"].raw_score == pytest.approx(9.0)
    assert by_id["u3"].raw_score == pytest.approx(11.4)
    order = [c.unit_id for c in sorted(out.candidates, key=lambda c: c.rank)]
    assert order == ["u3", "u2", "u1"]  # u2 leapfrogged u1
    assert by_id["u2"].signals["ctx_from"] == "u3"


# ---------------------------------------------------------------------------
# K31 — injection bound + score ordering
# ---------------------------------------------------------------------------


def test_k31_bare_reply_injected_below_parent():
    """K31: the non-nominated reply "x" adjacent to nominated top turn
    "t" (s=10) is injected at w·10 = 7.0 with ``ctx_from`` = t and
    ``pre_ctx_score`` = 0.0 — strictly below its parent."""
    t = _cand("t", 10.0, session="s1", seq=5)
    outs, stats = propagate_context(
        [_lane_out([t])],
        neighbors=[
            _row("t", "s1", 5),
            _row("x", "s1", 6, source="src-x", revision=3),
        ],
        mode="combined", w=0.7, W=1,
    )
    (out,) = outs
    by_id = {c.unit_id: c for c in out.candidates}
    x = by_id["x"]
    assert x.raw_score == pytest.approx(7.0)
    assert x.signals["ctx_from"] == "t"
    assert x.signals["pre_ctx_score"] == 0.0
    assert x.signals["ctx_injected"] is True
    # strictly below its parent: t kept its 10.0 (no nominated neighbor)
    assert by_id["t"].raw_score == pytest.approx(10.0)
    assert by_id["t"].rank == 1 and x.rank == 2
    assert stats["injected"] == 1 and stats["boosted"] == 0


def test_k31_injection_respects_M_ctx_and_cap():
    """Injection draws only from neighbors of the top-M_ctx nominated
    units and is capped at 2·M_ctx injected per lane (§21.1)."""
    # four nominated turns; M_ctx=1 → only the first's neighbors inject
    cands = [_cand(f"n{i}", 10.0 - i, session="s1", seq=10 + 10 * i)
             for i in range(4)]
    inventory = [_row(c.unit_id, "s1", c.signals["seq"]) for c in cands]
    inventory += [_row("x0", "s1", 11), _row("x1", "s1", 21)]
    outs, stats = propagate_context(
        [_lane_out(cands)], neighbors=inventory,
        mode="inject", w=0.7, W=1, M_ctx=1,
    )
    (out,) = outs
    ids = {c.unit_id for c in out.candidates}
    # only n0's window (seq 10) is a source → x0 injected; x1 (next to
    # n1, beyond M_ctx) is not.
    assert "x0" in ids and "x1" not in ids
    assert stats["injected"] == 1

    # the 2·M_ctx cap: M_ctx=1 admits at most 2 injections.  The five
    # same-seq siblings occupy distinct *positions* (V85-03: seq ties
    # break on the ordering tuple, so unit_id orders them) — W=3 puts
    # y0..y2 inside the window and the cap still binds.
    inventory2 = [_row("n0", "s1", 10)] + [
        _row(f"y{j}", "s1", 11) for j in range(5)
    ]
    outs2, stats2 = propagate_context(
        [_lane_out([_cand("n0", 10.0, session="s1", seq=10)])],
        neighbors=inventory2, mode="inject", w=0.7, W=3, M_ctx=1,
    )
    assert stats2["injected"] == fusion.CTX_INJECT_CAP_FACTOR * 1 == 2


# ---------------------------------------------------------------------------
# K32 — ineligible neighbors are neither injected nor sources
# ---------------------------------------------------------------------------


def test_k32_held_neighbor_neither_injected_nor_source():
    """K32: a held unit in the neighbor inventory is never injected; a
    non-nominated inventory row is never a propagation source either —
    boosts come only from the nominated (eligible) candidate set."""
    u1 = _cand("u1", 10.0, session="s1", seq=1)
    u2 = _cand("u2", 1.0, session="s1", seq=3)
    inventory = [
        _row("u1", "s1", 1),
        _row("u2", "s1", 3),
        _row("h", "s1", 2),   # held — absent from the eligible set
        _row("x", "s1", 2),   # eligible same-seq sibling
    ]
    eligible = frozenset({"u1", "u2", "x"})
    outs, stats = propagate_context(
        [_lane_out([u1, u2])], neighbors=inventory,
        mode="combined", w=0.7, W=1, eligible=eligible,
    )
    (out,) = outs
    by_id = {c.unit_id: c for c in out.candidates}
    assert "h" not in by_id          # never injected
    assert "x" in by_id              # the eligible neighbor was
    assert by_id["x"].signals["ctx_from"] == "u1"
    # "h" was not a propagation source: u2's only adjacent nominated
    # turn would be at seq 2 or 4 — none is nominated — so u2 keeps its
    # original score and carries no ctx provenance.
    assert by_id["u2"].raw_score == pytest.approx(1.0)
    assert "ctx_from" not in by_id["u2"].signals
    assert stats["injected"] == 1    # x only — h counted nowhere
    assert stats["boosted"] == 0


# ---------------------------------------------------------------------------
# K33 — erasure closure (the ctx-field arm is declared, not adopted)
# ---------------------------------------------------------------------------


def test_k33_erasure_closure_removes_words():
    """K33: with the ``ctx`` doclen field unadopted (a later §19 arm —
    ``_DOCLEN_FIELDS`` excludes it), the half that applies unconditionally
    is exercised for real: ``delete_source_v7`` sweeps the turn's units
    row and FTS index in one transaction, and a search for the erased
    turn's unique words returns nothing."""
    # the conditional clause's arm state, asserted honestly
    assert "ctx" not in units_jobs._DOCLEN_FIELDS

    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)  # real §30+§19 DDL incl. FTS5 mirror triggers
    conn.execute(
        "INSERT INTO units (unit_id, source_id, revision, scope_id,"
        " kind, session_id, seq, recorded_at_us, generation)"
        " VALUES ('u-x','src-x',1,?, 'turn','sess',2, ?, 1)",
        (SCOPE, T0),
    )
    conn.execute(
        "INSERT INTO units (unit_id, source_id, revision, scope_id,"
        " kind, session_id, seq, recorded_at_us, generation)"
        " VALUES ('u-y','src-y',1,?, 'turn','sess',3, ?, 1)",
        (SCOPE, T0),
    )
    cur = conn.execute(
        "INSERT INTO unit_fts_rows (unit_id, scope_id, generation)"
        " VALUES ('u-x', ?, 1)",
        (SCOPE,),
    )
    rid_x = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO unit_fts_rows (unit_id, scope_id, generation)"
        " VALUES ('u-y', ?, 1)",
        (SCOPE,),
    )
    rid_y = cur.lastrowid
    conn.execute(
        "INSERT INTO unit_fts_content (fts_row_id, text) VALUES (?,?)",
        (rid_x, "the zephyrine calibrator hummed quietly"),
    )
    conn.execute(
        "INSERT INTO unit_fts_content (fts_row_id, text) VALUES (?,?)",
        (rid_y, "a different turn entirely"),
    )

    def _match(term):
        return [
            r[0]
            for r in conn.execute(
                "SELECT r.unit_id FROM unit_fts f"
                " JOIN unit_fts_rows r ON r.row_id = f.rowid"
                " WHERE unit_fts MATCH ?",
                (term,),
            ).fetchall()
        ]

    assert _match("zephyrine") == ["u-x"]
    rep = closure_v7.delete_source_v7(conn, "src-x")  # the real closure sweep
    assert rep["deleted"]["units"] == 1
    assert rep["deleted"]["unit_fts_content"] == 1
    # erased turn's unique words are unsearchable — the ad trigger
    # dropped the index terms in the same transaction.
    assert _match("zephyrine") == []
    assert _match("calibrator") == []
    # the surviving turn is untouched
    assert _match("different") == ["u-y"]
    assert conn.execute(
        "SELECT COUNT(*) FROM units WHERE unit_id='u-y'"
    ).fetchone()[0] == 1


# ---------------------------------------------------------------------------
# K34 — injected turn pins its own bytes; neighbor text is decoration
# ---------------------------------------------------------------------------


def test_k34_injected_item_pins_own_bytes():
    """K34: the injected candidate carries the *injected unit's own*
    ``source_id``/``revision`` pin (from its inventory row) — not the
    parent's — and ``ctx_from`` is only the parent's unit id. At pack
    level the item's quote is its own bytes; the neighbor's text nests
    under ``.context`` and never merges into the pin."""
    t = _cand("t", 10.0, session="s1", seq=5, source="src-t", revision=2)
    outs, _ = propagate_context(
        [_lane_out([t])],
        neighbors=[
            _row("t", "s1", 5, source="src-t", revision=2),
            _row("x", "s1", 6, source="src-x", revision=7),
        ],
        mode="inject", w=0.7, W=1,
    )
    (out,) = outs
    x = next(c for c in out.candidates if c.unit_id == "x")
    # the pin is x's own — the parent's source/revision is not borrowed
    assert (x.source_id, x.revision) == ("src-x", 7)
    # decoration is a bare unit id — no parent text is copied over
    assert x.signals["ctx_from"] == "t"

    # -- delivery: assemble_pack materializes the injected unit's own
    #    quote bytes; the parent nests as context-only. ----------------
    x_bytes = b"Wow, same here!"
    t_bytes = b"I finally shipped the calibrator patch."
    scored = [
        ScoredCandidate(
            unit_id="x", source_id="src-x", revision=7,
            score=7.0, score_family="ranking/v7",
            detail={"quote": x_bytes, "ctx_from": "t"},
        )
    ]
    parent_item = PackItemV7(
        ref="u:t", unit_id="t", quote=t_bytes,
        pins={"byte_start": 0, "byte_end": len(t_bytes)},
    )
    res = assemble_pack(
        scored, mk_qv(), max_tokens=10_000, limit=10,
        neighbor_window=1,
        neighbors_fn=lambda item, n: [parent_item],
    )
    (item,) = res.items
    assert item.unit_id == "x"
    assert item.quote == x_bytes          # pinned to its own bytes
    (ctx_item,) = item.context
    assert ctx_item.unit_id == "t"
    assert ctx_item.quote == t_bytes      # neighbor text = decoration
    assert item.context[0] is not item


# ---------------------------------------------------------------------------
# K35 — session / scope / generation / kind boundaries
# ---------------------------------------------------------------------------


def test_k35_session_boundary_blocks_boost_and_inject():
    """K35 (session): neighbors in another session never boost and never
    inject — even at seq distance 1."""
    a = _cand("a", 10.0, session="sA", seq=1)
    outs, stats = propagate_context(
        [_lane_out([a])],
        neighbors=[
            _row("a", "sA", 1),
            _row("foreign", "sB", 2),   # same seq distance, other session
        ],
        mode="combined", w=0.7, W=1,
    )
    (out,) = outs
    by_id = {c.unit_id: c for c in out.candidates}
    assert "foreign" not in by_id
    assert by_id["a"].raw_score == pytest.approx(10.0)
    assert stats["boosted"] == 0 and stats["injected"] == 0


def test_k35_window_and_kind_boundaries():
    """K35 (window/kind): a turn beyond ``W`` *positions* is invisible
    (V85-03 — the window counts members, not raw seq deltas); a
    ``sentence_window`` unit is never a source and never a target."""
    # window: ``far`` sits two positions past ``u`` (``mid`` between
    # them) with W=1 → far is out of window; mid is not.
    u = _cand("u", 10.0, session="s1", seq=1)
    outs, stats = propagate_context(
        [_lane_out([u])],
        neighbors=[
            _row("u", "s1", 1),
            _row("mid", "s1", 2),
            _row("far", "s1", 3),
        ],
        mode="combined", w=0.7, W=1,
    )
    ids = {c.unit_id for c in outs[0].candidates}
    assert "far" not in ids          # two positions away — outside W
    assert "mid" in ids              # the in-window member still injects
    assert stats["injected"] == 1

    # kind: a sentence_window inventory row is not injected; a
    # sentence_window candidate is never a propagation target/source.
    sw = _cand("sw", 5.0, session="s1", seq=2, kind="sentence_window")
    outs2, stats2 = propagate_context(
        [_lane_out([u, sw])],
        neighbors=[
            _row("u", "s1", 1),
            _row("sw", "s1", 2, kind="sentence_window"),
            _row("sw2", "s1", 3, kind="sentence_window"),
        ],
        mode="combined", w=0.7, W=2,
    )
    by_id = {c.unit_id: c for c in outs2[0].candidates}
    assert "sw2" not in by_id                     # not injected
    assert by_id["sw"].raw_score == pytest.approx(5.0)   # not boosted
    assert "ctx_from" not in by_id["sw"].signals
    # and u's boost source set excludes sw: u's only nominated neighbor
    # would be sw at seq2 — not a turn — so u is unboosted.
    assert by_id["u"].raw_score == pytest.approx(10.0)
    assert stats2["boosted"] == 0


_UNITS_MIRROR = """
CREATE TABLE units (
    unit_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    scope_id TEXT NOT NULL,
    kind TEXT,
    session_id TEXT,
    seq INTEGER,
    speaker_canon TEXT,
    recorded_at_us INTEGER,
    occurred_start_us INTEGER,
    occurred_end_us INTEGER,
    byte_start INTEGER,
    byte_end INTEGER,
    generation INTEGER NOT NULL,
    PRIMARY KEY (unit_id, generation)
);
"""


def _add_turn(conn, uid, *, scope=SCOPE, gen=1, session="s1", seq=1,
              kind="turn"):
    conn.execute(
        "INSERT INTO units (unit_id, source_id, revision, scope_id,"
        " kind, session_id, seq, generation)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (uid, f"src-{uid}", 1, scope, kind, session, seq, gen),
    )


def test_k35_scope_and_generation_fence_inventory():
    """K35 (scope/generation): the production neighbor inventory reads
    only the caller's scope at the pinned generation — foreign-scope
    and past-fence units are absent, so context can never cross them."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(_UNITS_MIRROR)
    _add_turn(conn, "nom", session="sess", seq=1)
    _add_turn(conn, "nbr", session="sess", seq=2)
    _add_turn(conn, "other-scope", scope="scope-b", session="sess", seq=2)
    _add_turn(conn, "future-gen", session="sess", seq=3, gen=5)
    _add_turn(conn, "sw", session="sess", seq=4, kind="sentence_window")

    ctx = _mk_ctx(conn, scope=SCOPE, generation=1)
    out = _lane_out([_cand("nom", 9.0, session="sess", seq=1)])
    inv = pipeline_v7._context_inventory(ctx, [out])
    ids = {r[0] for r in inv}
    assert "nbr" in ids                    # same session, in scope/gen
    assert "other-scope" not in ids        # scope fence
    assert "future-gen" not in ids         # generation fence
    assert "sw" not in ids                 # kind='turn' filter
    assert all(r[3] == "turn" for r in inv)


# ---------------------------------------------------------------------------
# K36 — coverage.context fields + ≤2-statement neighbor lookup
# ---------------------------------------------------------------------------


def test_k36_coverage_context_reports_every_field():
    """K36: the context stats block (``coverage.context``) carries every
    declared field — model, mode, w, W, M_ctx, applied, injected,
    boosted, per-lane breakdown — through both the direct stage and the
    ``rrf_fuse(context=…)`` wiring the pipeline uses."""
    u = _cand("u", 10.0, session="s1", seq=1)
    v = _cand("v", 4.0, session="s1", seq=2)
    inventory = [_row("u", "s1", 1), _row("v", "s1", 2),
               _row("x", "s1", 3)]
    outs, stats = propagate_context(
        [_lane_out([u, v])], neighbors=inventory,
        mode="combined", w=0.7, W=1,
    )
    for key in ("model", "mode", "w", "W", "M_ctx", "applied",
                "injected", "boosted", "lanes", "neighbor_rows"):
        assert key in stats, key
    assert stats["model"] == "context/v8"
    assert stats["applied"] is True
    assert stats["lanes"]["lex"] == {"boosted": 2, "injected": 1}
    assert outs[0].stats["context_applied"] is True
    assert outs[0].stats["context"] == {"boosted": 2, "injected": 1}

    fused = rrf_fuse(
        [_lane_out([u, v])],
        {LaneName.LEX: 1.0},
        context={"mode": "combined", "w": 0.7, "W": 1,
                 "neighbors": inventory},
    )
    cstats = fused.stats["context"]
    assert cstats["model"] == "context/v8"
    assert cstats["injected"] == 1 and cstats["boosted"] == 2
    # the injected unit participates in fusion like any candidate
    xf = next(c for c in fused if c.unit_id == "x")
    assert xf.signals["ctx_injected"] is True
    # x (seq 3) is within W=1 of nominated v (seq 2) — not u (dist 2)
    assert xf.signals["ctx_from"] == "v"


def test_k36_neighbor_inventory_le_2_statements():
    """K36: ``_context_inventory`` is the one batched read — ≤ 2
    ``units`` statements per request (nominated-id chunk + session
    chunk), counted by the trace callback."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(_UNITS_MIRROR)
    for i in range(6):
        _add_turn(conn, f"t{i}", session="sess", seq=i)
    _add_turn(conn, "solo", session="other", seq=1)
    ctx = _mk_ctx(conn)
    out = _lane_out([
        _cand("t1", 9.0, session="sess", seq=1),
        _cand("solo", 8.0, session="other", seq=1),
    ])
    counts = _counting_trace(conn)
    inv = pipeline_v7._context_inventory(ctx, [out])
    conn.set_trace_callback(None)
    assert 0 < counts["units"] <= 2
    # the inventory really is the session batch: every turn in both
    # touched sessions, scope-fenced and generation-fenced.
    assert {r[0] for r in inv} == {f"t{i}" for i in range(6)} | {"solo"}
