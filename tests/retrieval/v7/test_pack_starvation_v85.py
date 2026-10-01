"""V8.5 pack-starvation regression tests.

Root cause of the c6bc563 Track R collapse (any@10 = 0.154): on any
query whose lanes blew the 500 ms deadline, ``_materialize_details``
skipped every fill, every candidate carried an empty ``quote``, and
``_collapse`` folded 255/256 of them onto the identical empty
signature — so the never-empty path shipped exactly one item with no
text (``n_surfaced == 1``, ``tokens == 0`` on all 990 dev questions).

Two fixes are pinned here:

- ``pack._collapse`` — an empty signature means "unmaterialized", not
  "duplicate"; distinct units must never fold onto it.
- ``pipeline._materialize_details`` — the deliverable prefix
  (``scored[:limit]``) fills unconditionally; the deadline hook gates
  only the tail beyond ``limit`` that exists for collapse backfill.
"""

import sqlite3
import types

import pytest

from verbatim.core.types_v7 import ScoredCandidate
from verbatim.retrieval.v7.pack import assemble_pack
from verbatim.retrieval.v7.pipeline import _materialize_details


def _scored_bare(unit_id: str, score: float) -> ScoredCandidate:
    return ScoredCandidate(
        unit_id=unit_id,
        source_id="src",
        revision=1,
        score=score,
        score_family="ranking/v7",
        detail={},
    )


class _Q:
    intent = None


def test_unmaterialized_items_never_collapse():
    """Empty signatures are 'unknown', not 'duplicate': six candidates
    with no materialized text deliver as six items, not one."""
    cands = [_scored_bare(f"u{i}", 10.0 - i) for i in range(6)]
    res = assemble_pack(cands, _Q(), max_tokens=None, limit=10)
    assert len(res.items) == 6
    assert {i.unit_id for i in res.items} == {f"u{i}" for i in range(6)}


def test_real_duplicates_still_collapse():
    """Same normalized text + same lifecycle still folds — the collapse
    itself is untouched."""
    a = _scored_bare("uA", 2.0)
    b = _scored_bare("uB", 1.0)
    for c, t in ((a, b"Hello there."), (b, b"hello there!")):
        c.detail["quote"] = t
    res = assemble_pack([a, b], _Q(), max_tokens=None, limit=10)
    assert len(res.items) == 1
    assert res.items[0].unit_id == "uA"
    assert res.collapsed_duplicates == 1


def _fixture_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE units (unit_id TEXT, speaker_canon TEXT, "
        "session_id TEXT, recorded_at_us INTEGER, occurred_start_us "
        "INTEGER, occurred_end_us INTEGER, occurred_precision TEXT, "
        "occurred_source TEXT, byte_start INTEGER, byte_end INTEGER, "
        "source_id TEXT, revision INTEGER, scope_id TEXT, "
        "generation INTEGER)"
    )
    conn.execute(
        "CREATE TABLE unit_fts_rows (unit_id TEXT, row_id INTEGER, "
        "generation INTEGER)"
    )
    conn.execute(
        "CREATE TABLE unit_fts_content (fts_row_id INTEGER, text TEXT)"
    )
    for i in range(3):
        conn.execute(
            "INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"u{i}", "gina", "s1", 1, 1, 1, "instant", "explicit",
             0, 10, "src", 1, "scope", 1),
        )
        conn.execute(
            "INSERT INTO unit_fts_rows VALUES (?,?,?)", (f"u{i}", i, 1)
        )
        conn.execute(
            "INSERT INTO unit_fts_content VALUES (?,?)",
            (i, f"turn text {i}"),
        )
    return conn


def test_materialize_fills_deliverable_prefix_under_deadline():
    """Deadline already blown -> the top-`limit` prefix still fills;
    only the backfill tail is droppable (V8-14.04)."""
    conn = _fixture_conn()
    ctx = types.SimpleNamespace(
        store=conn, scope_id="scope", generation=99,
        manifest={"limit": 2}, policy=None,
    )
    cands = [_scored_bare(f"u{i}", 10.0 - i) for i in range(3)]
    unfilled = _materialize_details(
        ctx, cands, budget_exceeded=lambda: True
    )
    assert unfilled == 1                      # the tail item is droppable
    assert cands[0].detail["quote"] == b"turn text 0"
    assert cands[1].detail["quote"] == b"turn text 1"
    assert "quote" not in cands[2].detail     # tail beyond limit=2


def test_materialize_tail_fills_when_budget_remains():
    conn = _fixture_conn()
    ctx = types.SimpleNamespace(
        store=conn, scope_id="scope", generation=99,
        manifest={"limit": 2}, policy=None,
    )
    cands = [_scored_bare(f"u{i}", 10.0 - i) for i in range(3)]
    unfilled = _materialize_details(
        ctx, cands, budget_exceeded=lambda: False
    )
    assert unfilled == 0
    assert all("quote" in c.detail for c in cands)


def test_materialize_never_fetches_beyond_two_x_limit():
    """Items past 2*limit can never ship — they are not fetched."""
    conn = _fixture_conn()
    ctx = types.SimpleNamespace(
        store=conn, scope_id="scope", generation=99,
        manifest={"limit": 1}, policy=None,
    )
    cands = [_scored_bare(f"u{i}", 10.0 - i) for i in range(3)]
    _materialize_details(ctx, cands, budget_exceeded=lambda: False)
    assert "quote" in cands[0].detail and "quote" in cands[1].detail
    assert "quote" not in cands[2].detail
