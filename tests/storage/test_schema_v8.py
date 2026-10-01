"""V8 additive-schema tests (SPEC_V8 §19, V8-19.01/19.04).

Covers the §19 additive set riding the ``ensure_v7_additive``
lazy-additive path: ``unit_time_mentions`` (write-time resolved
mentions, V8-09.04), ``unit_doclen`` (maintained BM25F field lengths,
V8-07.04), the two further ``units`` indexes (V8-09.06/V8-06.07), the
``events_v7.subject_source`` additive column via the PRAGMA-probed
idempotent-ALTER (V8-09.07), and the ``meta.schema_v8`` version marker.
Real on-disk databases throughout — same conventions as
``test_schema_v7``.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verbatim.core.types import safe_json_loads
from verbatim.storage.schema_v7 import (
    DDL_V8,
    SCHEMA_V8_TAG,
    V8_ALTER_COLUMNS,
    V8_INDEXES,
    V8_TABLES,
    ensure_v7_additive,
    ensure_v8_additive,
    v7_statements,
    v8_statements,
    v8_tables_present,
)
from verbatim.storage.store import Store


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _names(conn: sqlite3.Connection, kind: str) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type=?", (kind,)
        )
    }


def _cols(conn: sqlite3.Connection, table: str) -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _marker(conn: sqlite3.Connection):
    row = conn.execute(
        "SELECT value_json FROM meta WHERE key = 'schema_v8'"
    ).fetchone()
    return safe_json_loads(row[0]) if row else None


@pytest.fixture()
def ensured(store):
    """Fresh store with the additive schema ensured in one tx — the
    chained ensure leaves the V8 objects beside the V7 plane."""
    with store.tx() as conn:
        ensure_v7_additive(conn)
    return store


# ---------------------------------------------------------------------------
# presence / idempotence (V8-19.01)
# ---------------------------------------------------------------------------


def test_fresh_store_lacks_v8_objects_then_ensure(store):
    """Nothing V8 is eager: a fresh store reports absent; the chained
    ``ensure_v7_additive`` creates tables, indexes, column, and marker."""
    with store.read() as conn:
        assert v8_tables_present(conn) is False
        names = _names(conn, "table")
        for t in V8_TABLES:
            assert t not in names
        assert "events_v7" not in names  # the whole v7 plane is lazy too
        assert _marker(conn) is None
    with store.tx() as conn:
        ensure_v7_additive(conn)
    with store.read() as conn:
        assert v8_tables_present(conn) is True
        names = _names(conn, "table")
        for t in V8_TABLES:
            assert t in names
        assert _marker(conn) == SCHEMA_V8_TAG


def test_ensure_idempotent(ensured):
    """Repeated ensures are pure no-ops — CREATE IF NOT EXISTS for the
    tables/indexes, a probed ALTER for the column (a second unguarded
    ALTER would raise ``duplicate column name``)."""
    s = ensured
    with s.tx() as conn:
        ensure_v7_additive(conn)
        ensure_v8_additive(conn)
        ensure_v8_additive(conn)
    with s.read() as conn:
        assert v8_tables_present(conn) is True


def test_ensure_v8_standalone_provisions_base(store):
    """``ensure_v8_additive`` on a store missing the V7 plane ensures the
    base first — the V8 objects stand on ``units``/``events_v7``, so the
    call stays safe on any store."""
    with store.tx() as conn:
        ensure_v8_additive(conn)
    with store.read() as conn:
        assert v8_tables_present(conn) is True
        assert "units" in _names(conn, "table")
        assert _marker(conn) == SCHEMA_V8_TAG


def test_ensure_rolls_back_with_caller_tx(store):
    """The lazy ensure is transactional — a caller rollback takes the
    V8 CREATEs, the ALTER, and the marker with it."""
    with pytest.raises(RuntimeError):
        with store.tx() as conn:
            ensure_v7_additive(conn)
            raise RuntimeError("caller abort")
    with store.read() as conn:
        assert v8_tables_present(conn) is False
        assert _marker(conn) is None
        names = _names(conn, "table")
        for t in V8_TABLES:
            assert t not in names


def test_alter_column_on_preexisting_events_table(store):
    """The idempotent-ALTER path: an ``events_v7`` shaped like the §30
    original (no ``subject_source``) gains the column on ensure; an
    already-migrated table keeps its rows and the column stays NULL
    (NULL | 'extracted' | 'speaker_backfill' — no default rewrite)."""
    with store.tx() as conn:
        # v7 statements minus nothing — then drop the column's presence
        # by rebuilding events_v7 at its pre-V8 shape.
        ensure_v7_additive(conn)
        conn.execute("ALTER TABLE events_v7 RENAME TO events_v7_old")
        conn.execute(
            "CREATE TABLE events_v7 ("
            " event_id TEXT PRIMARY KEY, unit_id TEXT NOT NULL,"
            " scope_id TEXT NOT NULL, subject_canon TEXT,"
            " predicate_lemma TEXT, object_text TEXT, polarity TEXT,"
            " occurred_start_us INTEGER, occurred_end_us INTEGER,"
            " precision TEXT, pins_json TEXT, rule_id TEXT,"
            " generation INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT INTO events_v7(event_id, unit_id, scope_id,"
            " subject_canon, generation)"
            " VALUES ('e-old', 'u-old', 's1', 'alice', 1)"
        )
        conn.execute("DROP TABLE events_v7_old")
        assert "subject_source" not in _cols(conn, "events_v7")
        ensure_v8_additive(conn)
        assert "subject_source" in _cols(conn, "events_v7")
        row = conn.execute(
            "SELECT subject_source FROM events_v7 WHERE event_id='e-old'"
        ).fetchone()
        assert row == (None,)  # existing rows untouched — NULL provenance


# ---------------------------------------------------------------------------
# §19 columns / keys / indexes — the normative shapes
# ---------------------------------------------------------------------------


def test_unit_time_mentions_shape(ensured):
    with ensured.read() as conn:
        assert {
            "unit_id", "generation", "scope_id", "ord", "start_us",
            "end_us", "precision", "span_start", "span_end", "anchor_us",
            "resolver_version",
        } <= _cols(conn, "unit_time_mentions")
        pk = [
            r[1]
            for r in conn.execute("PRAGMA table_info(unit_time_mentions)")
            if r[5] > 0
        ]
        # §19 key: (unit_id, generation, ord)
        assert pk == ["unit_id", "generation", "ord"]


def test_unit_doclen_shape(ensured):
    with ensured.read() as conn:
        assert {
            "unit_id", "generation", "scope_id", "field", "len",
        } <= _cols(conn, "unit_doclen")
        pk = [
            r[1]
            for r in conn.execute("PRAGMA table_info(unit_doclen)")
            if r[5] > 0
        ]
        # §19 key: (unit_id, generation, field)
        assert pk == ["unit_id", "generation", "field"]


def test_v8_check_constraints(ensured):
    """The §19 CHECKs are real: ``end_us > start_us`` on mentions and
    ``len >= 0`` plus the precision enum on doclen/mentions."""
    s = ensured
    with s.tx() as conn:
        conn.execute(
            "INSERT INTO units(unit_id, source_id, revision, scope_id,"
            " kind, generation) VALUES ('u1', 'src-1', 1, 's1',"
            " 'turn', 1)"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO unit_time_mentions(unit_id, generation,"
                " scope_id, ord, start_us, end_us, precision,"
                " span_start, span_end, anchor_us, resolver_version)"
                " VALUES ('u1', 1, 's1', 0, 10, 10, 'day', 0, 4, 5,"
                " 'r1')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO unit_time_mentions(unit_id, generation,"
                " scope_id, ord, start_us, end_us, precision,"
                " span_start, span_end, anchor_us, resolver_version)"
                " VALUES ('u1', 1, 's1', 0, 5, 10, 'fortnight', 0, 4,"
                " 5, 'r1')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO unit_doclen(unit_id, generation, scope_id,"
                " field, len) VALUES ('u1', 1, 's1', 'text', -1)"
            )
        # the good rows land
        conn.execute(
            "INSERT INTO unit_time_mentions(unit_id, generation,"
            " scope_id, ord, start_us, end_us, precision, span_start,"
            " span_end, anchor_us, resolver_version)"
            " VALUES ('u1', 1, 's1', 0, 5, 10, 'day', 0, 4, 5, 'r1')"
        )
        conn.execute(
            "INSERT INTO unit_doclen(unit_id, generation, scope_id,"
            " field, len) VALUES ('u1', 1, 's1', 'ctx', 0)"
        )


def test_v8_indexes_exist(ensured):
    with ensured.read() as conn:
        names = _names(conn, "index")
        for idx in V8_INDEXES:
            assert idx in names


def test_v8_index_columns(ensured):
    """The §19 index definitions land verbatim — scope+generation fence
    leading, then the ordering columns."""
    with ensured.read() as conn:
        cols = [
            r[2]
            for r in conn.execute("PRAGMA index_list(units)")
            if r[1] == "idx_units_scope_occurred"
            for r in conn.execute(
                "PRAGMA index_info(idx_units_scope_occurred)"
            )
        ]
        assert cols == ["scope_id", "generation", "occurred_start_us"]
        cols = [
            r[2]
            for r in conn.execute(
                "PRAGMA index_info(idx_units_session_seq)"
            )
        ]
        assert cols == ["scope_id", "generation", "session_id", "seq"]
        cols = [
            r[2]
            for r in conn.execute("PRAGMA index_info(idx_utm_scope_time)")
        ]
        assert cols == ["scope_id", "generation", "start_us", "end_us"]


def test_v8_ddl_and_tag():
    assert SCHEMA_V8_TAG == "schema_v8/v1"
    stmts = v8_statements()
    assert len(stmts) == 5  # 2 tables + 3 indexes — the ALTER rides V8_ALTER_COLUMNS
    assert all(isinstance(s, str) and s for s in stmts)
    # the v7 statement tuple stays pure v7 — the additions never edit it
    assert all("unit_time_mentions" not in s for s in v7_statements())
    assert "unit_time_mentions" in DDL_V8
    assert "unit_doclen" in DDL_V8
    # the additive ALTER is declared for the probed path, verbatim §19
    assert V8_ALTER_COLUMNS == (
        (
            "events_v7",
            "subject_source",
            "ALTER TABLE events_v7 ADD COLUMN subject_source TEXT",
        ),
    )


def test_subject_source_column_nullable_and_named(ensured):
    """subject_source is the §19 additive column: present, nullable,
    TEXT — NULL | 'extracted' | 'speaker_backfill' per the enum comment."""
    with ensured.read() as conn:
        info = {
            r[1]: r
            for r in conn.execute("PRAGMA table_info(events_v7)")
        }
        col = info["subject_source"]
        assert col[2].upper() == "TEXT"
        assert col[3] == 0  # notnull — NULL is a legal provenance
    with ensured.tx() as conn:
        conn.execute(
            "INSERT INTO events_v7(event_id, unit_id, scope_id,"
            " subject_canon, subject_source, generation)"
            " VALUES ('e-bf', 'u-bf', 's1', 'melanie',"
            " 'speaker_backfill', 1)"
        )
    with ensured.read() as conn:
        row = conn.execute(
            "SELECT subject_source FROM events_v7"
            " WHERE event_id = 'e-bf'"
        ).fetchone()
        assert row == ("speaker_backfill",)
