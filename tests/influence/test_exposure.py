"""Source-lane exposure sink tests (SPEC_V6 V6-03.14,
docs/v6_contracts.md §8).

Covers the ``source_exposure`` journal through
``verbatim.influence.exposure``: row-per-delivery writes inside the
caller's write tx, ``(receipt_id, ord)`` idempotence, receipt-less
emission ('' stored, never fabricated), the ``emit_deliveries``
own-tx discipline, namespace/receipt filters, newest-first probes,
absent-table honesty, and delivery-shape validation. Real on-disk
``Store.create`` fixtures throughout — including the lazy
first-writer-creates path, since ``Store.create`` never runs the
migrations ensure phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.influence import exposure as _exposure
from verbatim.storage.repos import has_table
from verbatim.storage.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store.create(str(tmp_path / "v.db"))
    yield s
    s.close()


_DELIVERIES = [
    {"source_id": "src_a", "revision": 1, "score_family": "ranking/v1"},
    {"source_id": "src_b", "revision": 2, "score_family": "lexical"},
    {"source_id": "src_c", "revision": 1, "score_family": "vector"},
]


# ---------------------------------------------------------------------------
# writes + idempotence
# ---------------------------------------------------------------------------


def test_record_source_deliveries_writes_rows(store):
    """One row per delivered item; the additive table is created lazily
    inside the same write tx (fresh Store.create store — no apply ran)."""
    with store.read() as conn:
        assert not has_table(conn, "source_exposure")
    with store.tx() as conn:
        n = _exposure.record_source_deliveries(
            conn, receipt_id="rc_1", namespace="ns1",
            deliveries=_DELIVERIES,
        )
    assert n == 3
    with store.read() as conn:
        rows = conn.execute(
            "SELECT receipt_id, ord, source_id, revision, score_family,"
            " namespace, delivered_at_us FROM source_exposure"
            " ORDER BY ord"
        ).fetchall()
    assert [r[1] for r in rows] == [0, 1, 2]
    assert [r[2] for r in rows] == ["src_a", "src_b", "src_c"]
    assert all(r[0] == "rc_1" for r in rows)
    assert rows[0][4] == "ranking/v1" and rows[1][4] == "lexical"
    assert all(r[5] == "ns1" for r in rows)
    assert all(r[6] > 0 for r in rows)


def test_record_source_deliveries_idempotent(store):
    """A replayed batch under the same receipt is an insert-ignore no-op;
    the first write's rows stand."""
    with store.tx() as conn:
        n1 = _exposure.record_source_deliveries(
            conn, receipt_id="rc_1", namespace="ns1",
            deliveries=_DELIVERIES,
        )
    with store.tx() as conn:
        n2 = _exposure.record_source_deliveries(
            conn, receipt_id="rc_1", namespace="ns1",
            deliveries=_DELIVERIES,
        )
    assert (n1, n2) == (3, 0)
    with store.read() as conn:
        assert _exposure.exposure_count(conn, receipt_id="rc_1") == 3


def test_record_source_deliveries_partial_replay(store):
    """A shorter replay under the same receipt only counts the rows that
    were actually new — never double-counts delivered items."""
    with store.tx() as conn:
        _exposure.record_source_deliveries(
            conn, receipt_id="rc_1", namespace="ns1",
            deliveries=_DELIVERIES[:1],
        )
    with store.tx() as conn:
        n = _exposure.record_source_deliveries(
            conn, receipt_id="rc_1", namespace="ns1",
            deliveries=_DELIVERIES,
        )
    assert n == 2  # ord 0 replayed, ords 1-2 new


def test_record_source_deliveries_empty(store):
    with store.tx() as conn:
        assert _exposure.record_source_deliveries(
            conn, receipt_id="rc_1", namespace="ns1", deliveries=[],
        ) == 0


def test_record_source_deliveries_receipt_none(store):
    """receipt_id=None is honest '' — never a fabricated identifier."""
    with store.tx() as conn:
        _exposure.record_source_deliveries(
            conn, receipt_id=None, namespace="ns1",
            deliveries=_DELIVERIES[:1],
        )
    with store.read() as conn:
        row = conn.execute(
            "SELECT receipt_id FROM source_exposure"
        ).fetchone()
    assert row[0] == ""


def test_record_source_deliveries_validation(store):
    """Malformed delivery dicts refuse with typed VALIDATION — a bad
    row is never coerced into the journal."""
    with store.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            _exposure.record_source_deliveries(
                conn, receipt_id="rc_1", namespace="ns1",
                deliveries=[{"revision": 1}],
            )
        assert exc.value.code == ErrorCode.VALIDATION
        with pytest.raises(VerbatimError):
            _exposure.record_source_deliveries(
                conn, receipt_id="rc_1", namespace="ns1",
                deliveries=[{"source_id": "s1"}],
            )
        with pytest.raises(VerbatimError):
            _exposure.record_source_deliveries(
                conn, receipt_id="rc_1", namespace="ns1",
                deliveries=[{"source_id": "s1", "revision": "x"}],
            )
        with pytest.raises(VerbatimError):
            _exposure.record_source_deliveries(
                conn, receipt_id="rc_1", namespace="ns1",
                deliveries=[
                    {"source_id": "s1", "revision": 1, "score_family": ""}
                ],
            )


# ---------------------------------------------------------------------------
# emit_deliveries — own short write tx (contract §7.4)
# ---------------------------------------------------------------------------


def test_emit_deliveries_own_tx(store):
    """The hook opens its own store.tx() — callable with any receipt
    (search-minted token or None) and never inside a read tx."""
    n = _exposure.emit_deliveries(
        store, receipt_id="rc_search_1", namespace="ns1",
        deliveries=_DELIVERIES,
    )
    assert n == 3
    with store.read() as conn:
        assert _exposure.exposure_count(conn, receipt_id="rc_search_1") == 3
        assert _exposure.exposure_count(conn, namespace="ns1") == 3
        assert _exposure.exposure_count(conn, namespace="other") == 0


def test_emit_deliveries_no_receipt(store):
    n = _exposure.emit_deliveries(
        store, receipt_id=None, namespace="ns1",
        deliveries=_DELIVERIES[:2],
    )
    assert n == 2
    with store.read() as conn:
        assert _exposure.exposure_count(conn) == 2


def test_emit_deliveries_empty_short_circuits(store):
    """Empty delivery batches write nothing — and don't even open a tx
    (validation/normalization happens before admission)."""
    assert _exposure.emit_deliveries(store, deliveries=[]) == 0


def test_emit_deliveries_duck_typed_hits(store):
    """The facade may hand Hit-like objects straight over — memory_id /
    revision / score_family attributes normalize into rows."""

    class _Hit:
        def __init__(self, sid, rev, fam):
            self.memory_id = sid
            self.revision = rev
            self.score_family = fam

    hits = [_Hit("src_x", 3, "ranking/v1"), _Hit("src_y", 1, "typed/v1")]
    n = _exposure.emit_deliveries(
        store, receipt_id="rc_2", namespace="ns1", deliveries=hits,
    )
    assert n == 2
    with store.read() as conn:
        rows = conn.execute(
            "SELECT source_id, revision, score_family FROM source_exposure"
            " ORDER BY ord"
        ).fetchall()
    assert rows[0] == ("src_x", 3, "ranking/v1")
    assert rows[1] == ("src_y", 1, "typed/v1")


# ---------------------------------------------------------------------------
# reads — counts + probes, absent-table honesty
# ---------------------------------------------------------------------------


def test_exposure_count_filters(store):
    with store.tx() as conn:
        _exposure.record_source_deliveries(
            conn, receipt_id="rc_a", namespace="ns1",
            deliveries=_DELIVERIES[:2],
        )
        _exposure.record_source_deliveries(
            conn, receipt_id="rc_b", namespace="ns2",
            deliveries=_DELIVERIES[2:],
        )
    with store.read() as conn:
        assert _exposure.exposure_count(conn) == 3
        assert _exposure.exposure_count(conn, receipt_id="rc_a") == 2
        assert _exposure.exposure_count(conn, receipt_id="rc_b") == 1
        assert _exposure.exposure_count(conn, namespace="ns1") == 2
        assert _exposure.exposure_count(conn, namespace="ns2") == 1
        assert _exposure.exposure_count(
            conn, receipt_id="rc_b", namespace="ns2") == 1
        assert _exposure.exposure_count(
            conn, receipt_id="rc_a", namespace="ns2") == 0


def test_exposure_count_absent_table(store):
    """A store that never emitted reports 0 — absence is zero deliveries,
    not an error (fresh store, table never created)."""
    with store.read() as conn:
        assert _exposure.exposure_count(conn) == 0


def test_recent_exposures(store):
    with store.tx() as conn:
        _exposure.record_source_deliveries(
            conn, receipt_id="rc_a", namespace="ns1",
            deliveries=_DELIVERIES,
        )
    with store.read() as conn:
        rows = _exposure.recent_exposures(conn, "ns1", limit=10)
    assert len(rows) == 3
    assert rows[0]["source_id"] == "src_c"  # newest ord last-written wins
    assert all(
        set(r) == {"receipt_id", "ord", "source_id", "revision",
                   "score_family", "delivered_at_us", "namespace"}
        for r in rows
    )
    with store.read() as conn:
        assert _exposure.recent_exposures(conn, "ns1", limit=2)
        assert len(_exposure.recent_exposures(conn, "ns1", limit=2)) == 2
        assert _exposure.recent_exposures(conn, "empty_ns") == []


def test_recent_exposures_absent_table(store):
    with store.read() as conn:
        assert _exposure.recent_exposures(conn, "ns1") == []


# ---------------------------------------------------------------------------
# migration ensure-phase interaction
# ---------------------------------------------------------------------------


def test_existing_store_gains_table_on_open(tmp_path):
    """A store created before the additive table existed gains it on the
    next writable open (migrations._ensure_v6_additive) — no lazy write
    needed."""
    db = str(tmp_path / "pre.db")
    Store.create(db).close()  # create() never runs apply()
    s = Store.open(db)        # apply() → ensure phase creates the table
    try:
        with s.read() as conn:
            assert has_table(conn, "source_exposure")
    finally:
        s.close()


# ---------------------------------------------------------------------------
# eval/v5/feedback.py probe surface — the source-lane count + check verdict
# ---------------------------------------------------------------------------


def _fake_env(store):
    """Minimal stand-in for harness.ConsumerEnv — the probe only reads
    ``env.memory._store``."""
    from types import SimpleNamespace

    return SimpleNamespace(memory=SimpleNamespace(_store=store))


def test_feedback_probe_counts_source_exposure(store):
    """eval.v5.feedback._count_source_exposure reports rows the emit
    hook minted — the consumer-path twin of the influence count."""
    from eval.v5 import feedback as _fb

    env = _fake_env(store)
    assert _fb._count_source_exposure(env) == 0
    n = _exposure.emit_deliveries(
        store, receipt_id="rc_probe", namespace="ns", deliveries=_DELIVERIES
    )
    assert n == 3
    assert _fb._count_source_exposure(env) == 3


def test_feedback_exposure_check_truth_table():
    """exposure_rows_recorded passes when EITHER accounting surface
    demonstrates exposure (V6-03.14 "or an equivalent disclosed
    accounting"); measurable-zero stays inconclusive; unavailable only
    when neither count could be read."""
    from eval.v5 import feedback as _fb

    assert _fb._exposure_check(None, None) == "unavailable"
    assert _fb._exposure_check(0, 0) == "inconclusive"
    assert _fb._exposure_check(0, None) == "inconclusive"
    assert _fb._exposure_check(None, 0) == "inconclusive"
    assert _fb._exposure_check(2, 0) == "passed"
    assert _fb._exposure_check(0, 2) == "passed"   # the source-lane case
    assert _fb._exposure_check(None, 2) == "passed"
