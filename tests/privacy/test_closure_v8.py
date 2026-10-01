"""V8 derived-plane closure tests (SPEC_V8 §19, V8-19.03, V8-04.02).

The §19 additions are unit-keyed derived rows like the rest of the
plane: ``unit_time_mentions`` (V8-09.04) and ``unit_doclen`` (V8-07.04)
die with their unit at every generation — by source, by scope, and by
the post-flip stale-generation sweep.  ``events_v7.subject_source``
needs no handling of its own: it rides the event row, already a
closure member.  ``verify_unit_closure_v7`` is the residual check the
closure verifier runs — zero rows for any purged ``unit_id``.

Same fixture conventions as ``test_closure_v7``.
"""

from __future__ import annotations

import sqlite3

import pytest

from verbatim.privacy import closure_v7
from verbatim.privacy.closure_v7 import (
    delete_scope_v7,
    delete_source_v7,
    sweep_stale_generations_v7,
    v7_closure_coverage,
    verify_unit_closure_v7,
)
from verbatim.purge import execute_purge, plan_purge
from verbatim.storage.schema_v7 import (
    V8_TABLES,
    ensure_v7_additive,
)
from verbatim.storage.store import Store


# ---------------------------------------------------------------------------
# fixtures + builders
# ---------------------------------------------------------------------------


@pytest.fixture
def v7(tmp_path):
    """Store with the additive schema ensured (chains V8) + two scopes."""
    s = Store.create(str(tmp_path / "v8.db"))
    with s.tx() as conn:
        ensure_v7_additive(conn)
        for sid in ("scope:v7", "scope:other"):
            conn.execute(
                "INSERT INTO scopes (scope_id, profile_id, visibility)"
                " VALUES (?, 'prof', 'owner')",
                (sid,),
            )
    yield s
    s.close()


SID = "scope:v7"
OTHER = "scope:other"


def qscalar(conn, sql, params=()):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def make_source(conn, store, sid, src_id, rev=1, payload=b"raw bytes"):
    conn.execute(
        "INSERT OR IGNORE INTO sources (source_id, origin, source_kind,"
        " scope_id, created_us)"
        " VALUES (?, 'test', 'user_message', ?, 1)",
        (src_id, sid),
    )
    conn.execute(
        "INSERT INTO source_revisions"
        "(source_id, revision, payload, payload_hmac, event_us,"
        " captured_us, provenance, metadata_json)"
        " VALUES (?, ?, ?, ?, 1, 1, 'direct_user', '{}')",
        (src_id, rev, payload, store.hmac(payload)),
    )


def seed_unit_v8(conn, sid, src_id, uid, *, rev=1, gen=1,
                 subject_source="speaker_backfill"):
    """One unit plus its full V8 derived set — a time mention, a doclen
    row per fielded lane, and a subject-backfilled event row."""
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id,"
        " kind, generation) VALUES (?,?,?,?,'turn',?)",
        (uid, src_id, rev, sid, gen),
    )
    conn.execute(
        "INSERT INTO unit_time_mentions(unit_id, generation, scope_id,"
        " ord, start_us, end_us, precision, span_start, span_end,"
        " anchor_us, resolver_version)"
        " VALUES (?,?,?,?,100,200,'day',4,12,50,'resolver/v1')",
        (uid, gen, sid, 0),
    )
    for field in ("text", "entities", "when", "speaker", "session"):
        conn.execute(
            "INSERT INTO unit_doclen(unit_id, generation, scope_id,"
            " field, len) VALUES (?,?,?,?,7)",
            (uid, gen, sid, field),
        )
    conn.execute(
        "INSERT INTO events_v7(event_id, unit_id, scope_id,"
        " subject_canon, predicate_lemma, subject_source, generation)"
        " VALUES (?,?,?,'alice','met',?,?)",
        (f"ev-{uid}", uid, sid, subject_source, gen),
    )


def residual_counts(conn, uid):
    return {
        t: qscalar(
            conn, f"SELECT COUNT(*) FROM {t} WHERE unit_id = ?", (uid,)
        )
        for t in ("unit_time_mentions", "unit_doclen", "events_v7")
    }


# ---------------------------------------------------------------------------
# coverage contract — the §19 tables are enumerated, never unhandled
# ---------------------------------------------------------------------------


def test_v8_tables_in_closure_sets_and_coverage(v7):
    """The V8-19.03 registration: both §19 tables are swept members —
    coverage enumerates them and unhandled stays empty (V8-04.02's
    'enumerated by the closure verifier')."""
    assert set(closure_v7.V8_CLOSURE_TABLES) == set(V8_TABLES)
    with v7.read() as conn:
        cov = v7_closure_coverage(conn)
    assert cov["complete"] is True
    assert cov["unhandled"] == []
    assert set(cov["swept_v8"]) == set(V8_TABLES)
    wiring = closure_v7.register_v7_erasers()
    assert wiring["tables_v8"] == closure_v7.V8_CLOSURE_TABLES


# ---------------------------------------------------------------------------
# source erasure — per-unit rows die with the source's bytes
# ---------------------------------------------------------------------------


def test_source_delete_removes_v8_unit_artifacts(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit_v8(conn, SID, "src-1", "u8:1:src1:a")
        seed_unit_v8(conn, SID, "src-2", "u8:1:src2:a")  # sibling source
        conn.execute(
            "INSERT INTO sources (source_id, origin, source_kind,"
            " scope_id, created_us)"
            " VALUES ('src-2', 'test', 'user_message', ?, 1)",
            (SID,),
        )
        # verifier sees the rows before the sweep
        pre = verify_unit_closure_v7(conn, ["u8:1:src1:a"])
        assert pre["closed"] is False
        assert pre["residual"]["unit_time_mentions"] == 1
        assert pre["residual"]["unit_doclen"] == 5
        out = delete_source_v7(conn, "src-1")
        assert out["units_removed"] == 1
        assert out["deleted"]["unit_time_mentions"] == 1
        assert out["deleted"]["unit_doclen"] == 5
        assert out["deleted"]["events_v7"] == 1
    with v7.read() as conn:
        assert residual_counts(conn, "u8:1:src1:a") == {
            "unit_time_mentions": 0,
            "unit_doclen": 0,
            "events_v7": 0,
        }
        # sibling source's V8 plane untouched
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM unit_time_mentions"
            " WHERE unit_id = 'u8:1:src2:a'",
        ) == 1
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM unit_doclen"
            " WHERE unit_id = 'u8:1:src2:a'",
        ) == 5


def test_source_delete_all_generations_v8(v7):
    """Erasure is generation-blind on the V8 rows too: a stale-generation
    copy dies with the live one (V7-30.02 erasure form carried)."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit_v8(conn, SID, "src-1", "u8:g1:a", gen=1)
        seed_unit_v8(conn, SID, "src-1", "u8:g2:a", gen=2)
        out = delete_source_v7(conn, "src-1")
        assert out["units_removed"] == 2
        assert out["deleted"]["unit_time_mentions"] == 2
        assert out["deleted"]["unit_doclen"] == 10
    with v7.read() as conn:
        assert qscalar(conn, "SELECT COUNT(*) FROM unit_time_mentions") == 0
        assert qscalar(conn, "SELECT COUNT(*) FROM unit_doclen") == 0


def test_verifier_reports_zero_after_sweep(v7):
    """The V8-19.03 verifier check — zero rows for any purged unit_id —
    closes on the swept set and still flags a surviving unit."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        make_source(conn, v7, SID, "src-2")
        seed_unit_v8(conn, SID, "src-1", "u8:1:a")
        seed_unit_v8(conn, SID, "src-2", "u8:1:b")
        delete_source_v7(conn, "src-1")
        verdict = verify_unit_closure_v7(conn, ["u8:1:a"])
        assert verdict == {"closed": True, "residual": {}}
        # a unit that still lives reports open with named residuals
        verdict = verify_unit_closure_v7(conn, ["u8:1:b"])
        assert verdict["closed"] is False
        assert verdict["residual"]["units"] == 1
        assert verdict["residual"]["unit_time_mentions"] == 1
        assert verdict["residual"]["unit_doclen"] == 5
        assert verdict["residual"]["events_v7"] == 1
        # empty input closes vacuously; absent tables are skipped
        assert verify_unit_closure_v7(conn, []) == {
            "closed": True,
            "residual": {},
        }


def test_subject_source_rides_event_row_closure(v7):
    """events_v7.subject_source is removed with the event row — the
    existing event closure needs no new handling (V8-19.03 row 3)."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit_v8(conn, SID, "src-1", "u8:1:a",
                     subject_source="speaker_backfill")
        row = conn.execute(
            "SELECT subject_source FROM events_v7 WHERE unit_id='u8:1:a'"
        ).fetchone()
        assert row == ("speaker_backfill",)
        delete_source_v7(conn, "src-1")
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM events_v7 WHERE unit_id = 'u8:1:a'",
        ) == 0


# ---------------------------------------------------------------------------
# scope erasure + stale-generation sweep
# ---------------------------------------------------------------------------


def test_scope_delete_removes_v8_plane(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        make_source(conn, v7, OTHER, "src-2")
        seed_unit_v8(conn, SID, "src-1", "u8:1:a")
        seed_unit_v8(conn, OTHER, "src-2", "u8:1:b")
        out = delete_scope_v7(conn, SID)
        assert out["deleted"]["unit_time_mentions"] == 1
        assert out["deleted"]["unit_doclen"] == 5
    with v7.read() as conn:
        assert qscalar(conn, "SELECT COUNT(*) FROM unit_time_mentions") == 1
        assert qscalar(conn, "SELECT COUNT(*) FROM unit_doclen") == 5
        # the sibling scope's V8 rows survive
        assert residual_counts(conn, "u8:1:b") == {
            "unit_time_mentions": 1,
            "unit_doclen": 5,
            "events_v7": 1,
        }


def test_scope_delete_generation_narrowed_v8(v7):
    """generation= narrows the V8 sweep exactly like the v7 plane —
    the newer generation's rows of the SAME scope survive."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit_v8(conn, SID, "src-1", "u8:g1:a", gen=1)
        seed_unit_v8(conn, SID, "src-1", "u8:g2:a", gen=2)
        delete_scope_v7(conn, SID, generation=1)
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM unit_time_mentions"
            " WHERE generation = 1",
        ) == 0
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM unit_doclen WHERE generation = 1",
        ) == 0
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM unit_time_mentions"
            " WHERE generation = 2",
        ) == 1
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM unit_doclen WHERE generation = 2",
        ) == 5


def test_stale_generation_sweep_reclaims_v8_rows(v7):
    """Post-flip: rows strictly below the keep generation go; rows AT it
    stay (the V8 tables ride _SCOPE_GENERATION_TABLES)."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit_v8(conn, SID, "src-1", "u8:g1:a", gen=1)
        seed_unit_v8(conn, SID, "src-1", "u8:g2:a", gen=2)
        out = sweep_stale_generations_v7(conn, keep_generation=2)
        assert out["deleted"]["unit_time_mentions"] == 1
        assert out["deleted"]["unit_doclen"] == 5
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM unit_time_mentions"
            " WHERE generation = 2",
        ) == 1
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM unit_doclen WHERE generation = 2",
        ) == 5


# ---------------------------------------------------------------------------
# end-to-end: the public purge path carries the V8 sweep
# ---------------------------------------------------------------------------


def test_execute_purge_sweeps_v8_plane(v7):
    """plan → execute erases the source's bytes AND its V8 projection in
    one transaction; the receipt reports the per-table counts and the
    residual verifier closes."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit_v8(conn, SID, "src-1", "u8:1:a")
    plan = plan_purge(v7, SID, [("source", "src-1")], actor="alice")
    result = execute_purge(v7, plan["purge_id"])
    assert result["state"] == "completed"
    with v7.read() as conn:
        assert qscalar(conn, "SELECT COUNT(*) FROM unit_time_mentions") == 0
        assert qscalar(conn, "SELECT COUNT(*) FROM unit_doclen") == 0
        assert qscalar(conn, "SELECT COUNT(*) FROM events_v7") == 0
        verdict = verify_unit_closure_v7(conn, ["u8:1:a"])
        assert verdict == {"closed": True, "residual": {}}
    # the purge receipt discloses the v7 sweep counts under derived.v7_*
    assert result["derived"].get("v7_unit_time_mentions", 0) >= 1
    assert result["derived"].get("v7_unit_doclen", 0) >= 1
