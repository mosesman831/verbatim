"""V7.5 lifecycle currency — source-level supersession at retrieval
(V7-05.08 lifecycle leg; V5-14.12 window; V7-09.11 history semantics).

Regression: a ``replaces=``-superseded source's units flowed through the
unit lanes into the pack for current-value queries — the predecessor
answered alongside its successor (eval/v6 ``forbidden_hits`` leak).

Two layers, matching the implementation split:

* ``make_eligible`` withholds dispositions that can answer NO query
  (``recorded``/``corrected``/``retracted``/``archived``/``erased``,
  ``superseded`` with an unreadable boundary, not-yet-valid ``active``)
  — evaluated before rank on every lane;
* ``pipeline._apply_source_currency`` applies the intent-dependent
  window post-boost: non-history intents drop out-of-window sources;
  ``history_of`` keeps them and stamps ``detail["lifecycle"]`` so the
  pack labels them ``superseded``/``historical`` (V7-12.07).
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verbatim.core.types_v7 import IntentClass  # noqa: E402
from verbatim.retrieval.v7.eligibility import (  # noqa: E402
    _currency_verdict,
    _never_answerable,
    make_eligible,
)
from verbatim.retrieval.v7.pipeline import _apply_source_currency  # noqa: E402

from .test_eligibility import (  # noqa: E402
    _Db,
    _grant,
    _scope,
    _source,
    _unit,
    SC,
    ALICE,
)
from .test_pipeline import make_ctx, make_query  # noqa: E402


@pytest.fixture()
def db() -> _Db:
    d = _Db()
    yield d
    d.close()


# ---------------------------------------------------------------------------
# pure-function layer
# ---------------------------------------------------------------------------


def test_never_answerable_dispositions():
    now = 1_700_000_000.0
    for disp in ("recorded", "corrected", "retracted", "archived", "erased"):
        assert _never_answerable(disp, None, None, now), disp
    # superseded with an unreadable boundary never opens its window
    assert _never_answerable("superseded", "not-a-date", None, now)
    assert _never_answerable("superseded", None, None, now)
    # superseded with a real boundary stays eligible — history needs it
    assert not _never_answerable("superseded", "2020-01-01T00:00:00Z", None, now)
    # not-yet-valid active can answer nothing (not even history)
    assert _never_answerable("active", None, "2999-01-01T00:00:00Z", now)
    # active inside/at the end of its window stays eligible
    assert not _never_answerable("active", None, "2020-01-01T00:00:00Z", now)
    assert not _never_answerable("active", None, None, now)
    # unknown dispositions are admissible — not assertions of suppression
    assert not _never_answerable("disputed", None, None, now)
    assert not _never_answerable("", None, None, now)


def test_currency_verdict_window():
    now = 1_700_000_000.0
    past = "2020-01-01T00:00:00Z"
    future = "2999-01-01T00:00:00Z"

    # superseded, boundary passed: history only, always labeled
    assert _currency_verdict("superseded", past, None, None, now, False) == (
        False,
        None,
    )
    assert _currency_verdict("superseded", past, None, None, now, True) == (
        True,
        "superseded",
    )
    # superseded, window still open: delivers, labeled
    assert _currency_verdict("superseded", future, None, None, now, False) == (
        True,
        "superseded",
    )
    # active, closed window: history only, labeled historical
    assert _currency_verdict("active", None, past, past, now, False) == (
        False,
        None,
    )
    assert _currency_verdict("active", None, past, past, now, True) == (
        True,
        "historical",
    )
    # active, open window: delivers unlabeled
    assert _currency_verdict("active", None, past, future, now, False) == (
        True,
        None,
    )
    # never-current drops for every intent
    for history in (False, True):
        assert _currency_verdict("erased", None, None, None, now, history) == (
            False,
            None,
        )
    # unknown/empty dispositions: admissible, unlabeled
    for history in (False, True):
        assert _currency_verdict("", None, None, None, now, history) == (
            True,
            None,
        )
        assert _currency_verdict("weird-state", None, None, None, now, history) == (
            True,
            None,
        )


# ---------------------------------------------------------------------------
# eligibility — never-answerable withheld before rank
# ---------------------------------------------------------------------------


def _state_disp(
    conn,
    source_id: str,
    namespace: str,
    disposition: str,
    effective_at: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO source_state(source_id, namespace, control_version,"
        " mutation_head, disposition, effective_at, valid_from, valid_to,"
        " known_at, updated_at, producer)"
        " VALUES (?,?,1,'1',?,?,?,?,'t','t','test')",
        (source_id, namespace, disposition, effective_at, valid_from, valid_to),
    )


def test_eligibility_withholds_never_answerable(db):
    """An ``erased`` source's units never enter the eligible set; a
    ``superseded`` source with a real boundary stays eligible (history
    needs it — the currency stage applies the intent-dependent window)."""
    conn = db.conn
    _grant(conn)
    _source(conn, "src-erased", SC)
    _state_disp(conn, "src-erased", SC, "erased")
    _unit(conn, "u-erased", "src-erased")

    _source(conn, "src-sup", SC)
    _state_disp(conn, "src-sup", SC, "superseded", effective_at="2020-01-01T00:00:00Z")
    _unit(conn, "u-sup", "src-sup")

    _source(conn, "src-live", SC)
    _state_disp(conn, "src-live", SC, "active")
    _unit(conn, "u-live", "src-live")

    eligible = make_eligible(
        conn, db, scope_id=SC, generation=9, principal_id=ALICE
    )
    assert eligible.stats["mode"] == "ok"
    assert eligible.stats["lifecycle_blocked"] == 1
    assert not eligible({"unit_id": "u-erased"})
    assert eligible({"unit_id": "u-sup"})      # history's answer stays
    assert eligible({"unit_id": "u-live"})


def test_eligibility_blocks_unbounded_superseded(db):
    """``superseded`` without a readable ``effective_at`` can never open
    its window — withheld before rank for every intent."""
    conn = db.conn
    _grant(conn)
    _source(conn, "src-noeff", SC)
    _state_disp(conn, "src-noeff", SC, "superseded", effective_at=None)
    _unit(conn, "u-noeff", "src-noeff")

    eligible = make_eligible(
        conn, db, scope_id=SC, generation=9, principal_id=ALICE
    )
    assert eligible.stats["lifecycle_blocked"] == 1
    assert not eligible({"unit_id": "u-noeff"})


def test_eligibility_blocks_not_yet_valid(db):
    conn = db.conn
    _grant(conn)
    _source(conn, "src-future", SC)
    _state_disp(conn, "src-future", SC, "active", valid_from="2999-01-01T00:00:00Z")
    _unit(conn, "u-future", "src-future")

    eligible = make_eligible(
        conn, db, scope_id=SC, generation=9, principal_id=ALICE
    )
    assert eligible.stats["lifecycle_blocked"] == 1
    assert not eligible({"unit_id": "u-future"})


def test_eligibility_missing_state_row_admissible(db):
    """No ``source_state`` row is unresolved state, not suppression —
    the unit stays eligible (V5-14.16)."""
    conn = db.conn
    _grant(conn)
    _source(conn, "src-bare", SC)
    _unit(conn, "u-bare", "src-bare")

    eligible = make_eligible(
        conn, db, scope_id=SC, generation=9, principal_id=ALICE
    )
    assert eligible({"unit_id": "u-bare"})


# ---------------------------------------------------------------------------
# pipeline currency stage — the intent-dependent window
# ---------------------------------------------------------------------------


def _ctx_with_conn(conn, history: bool = False):
    q = make_query("when is the weekly sync")
    if history:
        q = dc_replace(
            q,
            intent=dc_replace(
                q.intent,
                primary=IntentClass.HISTORY_OF,
                classes=(IntentClass.HISTORY_OF,),
            ),
        )
    return dc_replace(make_ctx(lanes=()), store=conn), q


def _mk_scored(unit_id: str, source_id: str):
    from verbatim.core.types_v7 import ScoredCandidate

    return ScoredCandidate(
        unit_id=unit_id, source_id=source_id, revision=1, score=1.0,
        score_family="ranking/v7",
    )


def test_currency_drops_superseded_for_current_query(db):
    conn = db.conn
    _scope(conn)
    _source(conn, "src-v1", SC)
    _state_disp(conn, "src-v1", SC, "superseded", effective_at="2020-01-01T00:00:00Z")
    _source(conn, "src-v2", SC)
    _state_disp(conn, "src-v2", SC, "active")

    ctx, q = _ctx_with_conn(conn)
    scored = [_mk_scored("u1", "src-v1"), _mk_scored("u2", "src-v2")]
    kept, note = _apply_source_currency(ctx, scored, q)
    assert note["status"] == "ok" and note["dropped"] == 1
    assert [s.unit_id for s in kept] == ["u2"]


def test_currency_keeps_superseded_labeled_for_history(db):
    conn = db.conn
    _scope(conn)
    _source(conn, "src-v1", SC)
    _state_disp(conn, "src-v1", SC, "superseded", effective_at="2020-01-01T00:00:00Z")

    ctx, q = _ctx_with_conn(conn, history=True)
    scored = [_mk_scored("u1", "src-v1")]
    kept, note = _apply_source_currency(ctx, scored, q)
    assert note["dropped"] == 0 and note["labeled"] == 1
    assert kept[0].detail["lifecycle"] == "superseded"


def test_currency_open_window_superseded_delivers_labeled(db):
    """Window still open (now < effective_at): delivers, labeled
    ``superseded`` — the honest mid-flight state (e40 semantics)."""
    conn = db.conn
    _scope(conn)
    _source(conn, "src-v1", SC)
    _state_disp(conn, "src-v1", SC, "superseded", effective_at="2999-01-01T00:00:00Z")

    ctx, q = _ctx_with_conn(conn)
    kept, note = _apply_source_currency(ctx, [_mk_scored("u1", "src-v1")], q)
    assert note["dropped"] == 0 and note["labeled"] == 1
    assert kept[0].detail["lifecycle"] == "superseded"


def test_currency_closed_active_window(db):
    conn = db.conn
    _scope(conn)
    _source(conn, "src-old", SC)
    _state_disp(
        conn, "src-old", SC, "active",
        valid_from="2020-01-01T00:00:00Z", valid_to="2020-06-01T00:00:00Z",
    )

    ctx, q = _ctx_with_conn(conn)
    kept, note = _apply_source_currency(ctx, [_mk_scored("u1", "src-old")], q)
    assert note["dropped"] == 1 and kept == []

    ctx, q = _ctx_with_conn(conn, history=True)
    s = _mk_scored("u1", "src-old")
    kept, note = _apply_source_currency(ctx, [s], q)
    assert note["dropped"] == 0
    assert kept[0].detail["lifecycle"] == "historical"


def test_currency_missing_state_row_admissible(db):
    conn = db.conn
    _scope(conn)
    _source(conn, "src-bare", SC)
    ctx, q = _ctx_with_conn(conn)
    kept, note = _apply_source_currency(ctx, [_mk_scored("u1", "src-bare")], q)
    assert note["dropped"] == 0 and len(kept) == 1


def test_currency_no_state_table_skips():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(x)")
    ctx, q = _ctx_with_conn(conn)
    kept, note = _apply_source_currency(ctx, [_mk_scored("u1", "src-x")], q)
    assert note["status"] == "skipped" and len(kept) == 1


def test_currency_no_snapshot_skips():
    ctx, q = _ctx_with_conn(None)
    kept, note = _apply_source_currency(ctx, [_mk_scored("u1", "src-x")], q)
    assert note["status"] == "skipped" and note["reason"] == "no_read_snapshot"
    assert len(kept) == 1


def test_currency_does_not_clobber_disputed(db):
    """A fact-level ``disputed`` mark survives a closed-window ``active``
    label — only ``current``/blank is overwritten by ``historical``."""
    conn = db.conn
    _scope(conn)
    _source(conn, "src-old", SC)
    _state_disp(
        conn, "src-old", SC, "active",
        valid_from="2020-01-01T00:00:00Z", valid_to="2020-06-01T00:00:00Z",
    )
    ctx, q = _ctx_with_conn(conn, history=True)
    s = _mk_scored("u1", "src-old")
    s.detail["lifecycle"] = "disputed"
    kept, _ = _apply_source_currency(ctx, [s], q)
    assert kept[0].detail["lifecycle"] == "disputed"
