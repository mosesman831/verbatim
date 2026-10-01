"""V7 derived-plane closure tests (SPEC_V7 §30, V7-06.11/V7-19.09, V7-30.02).

The D7-class gap the schema audit flagged: a "deleted" memory leaking
through V7 artifacts.  These tests pin the closure contract — every
content-bearing V7 table dies with the unit's source revision, FTS
shadows drain through the mirror triggers (never a direct write),
lexical stats decrement inside the same transaction, graph edges close
in both directions, shared ``entity_canon`` rows survive while other
mentions stand, stale-generation rows outlive the sweep until the
projection fence flips, and the quarantine cascade hides a unit whose
covering chain (unit → source → revision → span → envelope) is held.

Audit retention is asserted, not assumed: ``screening_log`` rows are
counted under ``audit_retained`` and never deleted — the write-channel
journal survives evidence closure like the v1 ``events`` journal.
"""

from __future__ import annotations

import json
import sqlite3
import struct

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.privacy import closure_v7
from verbatim.privacy.closure_v7 import (
    delete_scope_v7,
    delete_source_v7,
    held_unit_ids,
    register_v7_erasers,
    sweep_stale_generations_v7,
    v7_closure_coverage,
)
from verbatim.purge import execute_purge, plan_purge
from verbatim.storage.repos import has_table
from verbatim.storage.schema_v7 import (
    V7_FTS_TABLES,
    V7_INTERNAL_TABLES,
    V7_TABLES,
    ensure_v7_additive,
    v7_fts_present,
)
from verbatim.storage.stats_v7 import corpus_stats, update_stats
from verbatim.storage.store import Store


# ---------------------------------------------------------------------------
# fixtures + builders
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """A plain store — the additive V7 schema is never ensured."""
    s = Store.create(str(tmp_path / "prev7.db"))
    yield s
    s.close()


@pytest.fixture
def v7(tmp_path):
    """Store with the additive V7 schema ensured + two scopes."""
    s = Store.create(str(tmp_path / "v7.db"))
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


def qrows(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


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


def seed_unit(conn, sid, src_id, uid, *, rev=1, gen=1,
              text="alice met bob at the club", speaker="alice",
              entities="bob", session="sess-1", when="2026-09-18"):
    """One unit plus its full per-unit derived set, exactly the shape
    ``project_units_v7`` commits (carrier rowid == units.rowid is a
    writer convenience the closure does not depend on)."""
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " generation) VALUES (?,?,?,?,'turn',?)",
        (uid, src_id, rev, sid, gen),
    )
    cur = conn.execute(
        "INSERT INTO unit_fts_rows(unit_id, scope_id, generation)"
        " VALUES (?,?,?)",
        (uid, sid, gen),
    )
    conn.execute(
        'INSERT INTO unit_fts_content(fts_row_id, text, speaker,'
        ' entities, session, "when") VALUES (?,?,?,?,?,?)',
        (cur.lastrowid, text, speaker, entities, session, when),
    )
    conn.execute(
        "INSERT INTO events_v7(event_id, unit_id, scope_id,"
        " subject_canon, predicate_lemma, generation)"
        " VALUES (?,?,?,'alice','met',?)",
        (f"ev-{uid}", uid, sid, gen),
    )
    conn.execute(
        "INSERT INTO state_facts(scope_id, state_key, unit_id,"
        " generation, value_text) VALUES (?,?,?,?, 'v1')",
        (sid, f"key-{uid}", uid, gen),
    )
    conn.execute(
        "INSERT INTO preferences(scope_id, subject_canon, unit_id,"
        " generation, object_text) VALUES (?,?,?,?,'sushi')",
        (sid, "alice", uid, gen),
    )
    conn.execute(
        "INSERT INTO standing_rules(rule_id, scope_id, unit_id,"
        " generation, status) VALUES (?,?,?,?,'active')",
        (f"rule-{uid}", sid, uid, gen),
    )
    return cur.lastrowid


def seed_stats(conn, sid, gen, rows):
    """Corpus stats seeded exactly as the writer counts them — the same
    ``_fields_for_stats`` field build, so a delete decrements what the
    add counted (V7-06.06 same-tx symmetry)."""
    update_stats(
        conn,
        sid,
        gen,
        [
            {
                "unit_id": uid,
                "fields": closure_v7._fields_for_stats(*fields),
            }
            for uid, fields in rows
        ],
    )


def seed_vector_block(conn, sid, gen, encoder, block_no, keys,
                      dims=4, quant="f32"):
    """A packed unit_vectors_block in the rowmap layout — u32 count,
    u32 len + utf8 per key; n_rows × dims × width data; 8-byte f64
    norm per row for f32."""
    rowmap = bytearray(struct.pack("<I", len(keys)))
    for k in keys:
        kb = k.encode("utf-8")
        rowmap += struct.pack("<I", len(kb))
        rowmap += kb
    n = len(keys)
    data = bytes(range(n * dims * 4))[: n * dims * 4]
    scale = struct.pack(f"<{n}d", *[1.0 + i for i in range(n)])
    conn.execute(
        "INSERT INTO unit_vectors_block(encoder_id, scope_id,"
        " generation, block_no, n_rows, dims, quant, scale_blob,"
        " data_blob, rowmap_blob)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (encoder, sid, gen, block_no, n, dims, quant,
         scale, data, bytes(rowmap)),
    )


def decode_rowmap(blob):
    (n,) = struct.unpack_from("<I", blob, 0)
    pos, keys = 4, []
    for _ in range(n):
        (klen,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        keys.append(blob[pos : pos + klen].decode("utf-8"))
        pos += klen
    return keys


def quarantine(conn, sid, kind, oid, rev=0, state="pending"):
    conn.execute(
        "INSERT INTO quarantine(object_kind, object_id, revision,"
        " scope_id, reason_codes_json, findings_json, state,"
        " opened_event, decision_json)"
        " VALUES (?,?,?,?, '[]', '[]', ?, 1, '{}')",
        (kind, oid, rev, sid, state),
    )


# ---------------------------------------------------------------------------
# coverage contract — the D7 registration must name every §30 table
# ---------------------------------------------------------------------------


def test_table_sets_cover_every_v7_table():
    """The allowlist is complete by construction: closure + FTS-via-
    triggers + audit-retained == V7_TABLES ∪ V7_INTERNAL_TABLES, exactly.
    A §30 schema addition that forgets closure lands in ``unhandled``
    and fails this test — the drift guard the audit asked for."""
    covered = (
        set(closure_v7.V7_CLOSURE_TABLES)
        | set(closure_v7.V7_FTS_INDEXES)
        | set(closure_v7.V7_AUDIT_TABLES)
    )
    assert covered == set(V7_TABLES) | set(V7_INTERNAL_TABLES)
    assert set(closure_v7.V7_FTS_INDEXES) == set(V7_FTS_TABLES)


def test_coverage_report_complete_on_ensured_store(v7):
    with v7.read() as conn:
        cov = v7_closure_coverage(conn)
    assert cov["complete"] is True
    assert cov["unhandled"] == []
    assert set(cov["swept"]) == set(closure_v7.V7_CLOSURE_TABLES)


def test_coverage_reports_unhandled_when_table_drifts(v7, monkeypatch):
    """If schema_v7 grows a table this module never claimed, every sweep
    receipt + the coverage report names it — reported, never silently
    swept and never silently skipped."""
    monkeypatch.setattr(
        closure_v7._schema_v7,
        "V7_TABLES",
        tuple(list(V7_TABLES) + ["future_v7_table"]),
    )
    with v7.tx() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS future_v7_table (x TEXT)"
        )
        cov = v7_closure_coverage(conn)
        assert "future_v7_table" in cov["unhandled"]
        assert cov["complete"] is False
        out = delete_scope_v7(conn, SID)
        assert "future_v7_table" in out["unhandled_v7"]


def test_register_v7_erasers_wiring_record(v7):
    wiring = register_v7_erasers()
    assert wiring["tables"] == closure_v7.V7_CLOSURE_TABLES
    assert wiring["fts_indexes_via_triggers"] == closure_v7.V7_FTS_INDEXES
    assert wiring["audit_retained"] == closure_v7.V7_AUDIT_TABLES
    hooks = wiring["hooks"]
    assert "verbatim/purge.py::_empty_revision" in hooks
    assert "verbatim/memory/controls.py::_sweep_derived" in hooks
    assert "verbatim/storage/store.py::_COUNT_TABLES" in hooks
    assert "verbatim/export.py::NEVER_EXPORTED" in hooks
    registry: dict = {}
    register_v7_erasers(registry)
    assert registry["v7"]["tables"] == closure_v7.V7_CLOSURE_TABLES


# ---------------------------------------------------------------------------
# source erasure — the whole per-source derived plane
# ---------------------------------------------------------------------------


def test_source_delete_removes_unit_keyed_artifacts(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:src1:a")
        seed_unit(conn, SID, "src-1", "u7:1:src1:b")
        seed_unit(conn, SID, "src-2", "u7:1:src2:a")  # sibling source
        conn.execute(
            "INSERT INTO sources (source_id, origin, source_kind,"
            " scope_id, created_us)"
            " VALUES ('src-2', 'test', 'user_message', ?, 1)",
            (SID,),
        )
        out = delete_source_v7(conn, "src-1")
        assert out["units_removed"] == 2
        for t in ("units", "unit_fts_rows", "unit_fts_content",
                  "entity_mentions", "events_v7", "state_facts",
                  "preferences", "standing_rules"):
            assert out["deleted"].get(t, 0) >= 1 or t in out["deleted"]
    with v7.read() as conn:
        for t in ("units", "unit_fts_rows", "events_v7", "state_facts",
                  "preferences", "standing_rules"):
            assert qscalar(
                conn, f"SELECT COUNT(*) FROM {t} WHERE unit_id LIKE"
                " 'u7:1:src1:%'"
            ) == 0
        # content rows join through the carrier (fts_row_id only)
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM unit_fts_content c"
            " JOIN unit_fts_rows r ON r.row_id = c.fts_row_id"
            " WHERE r.unit_id LIKE 'u7:1:src1:%'",
        ) == 0
        # sibling source's plane is untouched
        assert qscalar(
            conn, "SELECT COUNT(*) FROM units WHERE source_id = 'src-2'"
        ) == 1


def test_source_delete_revision_scoped(v7):
    """``revision=`` narrows to one revision's units — the sibling
    revision of the SAME source survives (span-purge semantics)."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1", rev=1)
        make_source(conn, v7, SID, "src-1", rev=2, payload=b"rev two")
        seed_unit(conn, SID, "src-1", "u7:1:r1", rev=1)
        seed_unit(conn, SID, "src-1", "u7:1:r2", rev=2)
        out = delete_source_v7(conn, "src-1", revision=1)
        assert out["units_removed"] == 1
    with v7.read() as conn:
        assert qscalar(
            conn, "SELECT COUNT(*) FROM units WHERE unit_id = 'u7:1:r1'"
        ) == 0
        assert qscalar(
            conn, "SELECT COUNT(*) FROM units WHERE unit_id = 'u7:1:r2'"
        ) == 1


def test_source_delete_all_generations(v7):
    """Erasure is generation-blind: a stale-generation copy of the
    source's units dies with the live one — no generation of a deleted
    memory may remain queryable (V7-30.02 erasure form)."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:g1:a", gen=1)
        seed_unit(conn, SID, "src-1", "u7:g2:a", gen=2)
        seed_unit(conn, SID, "src-2", "u7:g2:b", gen=2)
        out = delete_source_v7(conn, "src-1")
        assert out["units_removed"] == 2
    with v7.read() as conn:
        assert qscalar(conn, "SELECT COUNT(*) FROM units") == 1


def test_fts_pair_and_index_drain_through_triggers(v7):
    """Content rows go first, carriers second; the *_ad triggers mirror
    into the FTS5 indexes — MATCH returns nothing after the sweep and
    foreign_key_check stays clean (the carrier/content pair never
    dangles)."""
    with v7.tx() as conn:
        fts = v7_fts_present(conn)
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a", text="zebra quokka")
        seed_unit(conn, SID, "src-2", "u7:1:b", text="zebra quokka")
        delete_source_v7(conn, "src-1")
        assert qscalar(conn, "SELECT COUNT(*) FROM unit_fts_content") == 1
        assert qscalar(conn, "SELECT COUNT(*) FROM unit_fts_rows") == 1
        if fts.get("unit_fts"):
            assert qscalar(
                conn,
                "SELECT COUNT(*) FROM unit_fts WHERE unit_fts MATCH"
                " 'zebra'",
            ) == 1  # the surviving unit only
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_lex_stats_decrement_in_same_tx(v7):
    """n_units/df subtract exactly what the add counted, floored at 0 —
    the surviving unit's terms keep their counts."""
    text_a = "alpha beta gamma"
    text_b = "alpha delta"
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        make_source(conn, v7, SID, "src-2")
        seed_unit(conn, SID, "src-1", "u7:1:a", text=text_a,
                  speaker="alice")
        seed_unit(conn, SID, "src-2", "u7:1:b", text=text_b,
                  speaker="bob")
        seed_stats(conn, SID, 1, [
            ("u7:1:a", (text_a, "alice", "", "", "")),
            ("u7:1:b", (text_b, "bob", "", "", "")),
        ])
        before = corpus_stats(conn, SID, 1)
        assert before["text"]["n"] == 2
        delete_source_v7(conn, "src-1")
        after = corpus_stats(conn, SID, 1)
        assert after["text"]["n"] == 1
        df = qscalar(
            conn,
            "SELECT df FROM lex_df WHERE scope_id = ? AND generation = 1"
            " AND field = 'text' AND term = 'beta'",
            (SID,),
        )
        assert df == 0  # only unit A carried it — floored at zero
        assert qscalar(
            conn,
            "SELECT df FROM lex_df WHERE scope_id = ? AND generation = 1"
            " AND field = 'text' AND term = 'alpha'",
            (SID,),
        ) == 1  # unit B still does


def test_shared_entity_canon_survives_df_recomputed(v7):
    """A canon mentioned by a surviving unit stays — df_units is
    recomputed from remaining mentions, not decremented blind."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        make_source(conn, v7, SID, "src-2")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        seed_unit(conn, SID, "src-2", "u7:1:b")
        for uid in ("u7:1:a", "u7:1:b"):
            conn.execute(
                "INSERT INTO entity_mentions(scope_id, canon, unit_id,"
                " generation, byte_start)"
                " VALUES (?, 'bob', ?, 1, 0)",
                (SID, uid),
            )
        conn.execute(
            "INSERT INTO entity_canon(scope_id, canon, generation,"
            " df_units) VALUES (?, 'bob', 1, 2)",
            (SID,),
        )
        delete_source_v7(conn, "src-1")
        row = qrows(
            conn,
            "SELECT df_units FROM entity_canon WHERE canon = 'bob'",
        )
        assert row == [{"df_units": 1}]


def test_orphaned_canon_drops_rule_alias_dies_review_survives(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        conn.execute(
            "INSERT INTO entity_mentions(scope_id, canon, unit_id,"
            " generation, byte_start) VALUES (?, 'zed', ?, 1, 0)",
            (SID, "u7:1:a"),
        )
        conn.execute(
            "INSERT INTO entity_canon(scope_id, canon, generation,"
            " df_units) VALUES (?, 'zed', 1, 1)",
            (SID,),
        )
        conn.execute(
            "INSERT INTO entity_aliases_v7(scope_id, canon, alias_canon,"
            " generation, method) VALUES (?, 'zed', 'z', 1, 'rule')",
            (SID,),
        )
        conn.execute(
            "INSERT INTO entity_aliases_v7(scope_id, canon, alias_canon,"
            " generation, method, state)"
            " VALUES (?, 'zed', 'zeddy', 1, 'review', 'active')",
            (SID,),
        )
        delete_source_v7(conn, "src-1")
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM entity_canon WHERE canon = 'zed'",
        ) == 0
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM entity_aliases_v7"
            " WHERE alias_canon = 'z'",
        ) == 0
        # reviewer-minted state is never resurrected or deleted
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM entity_aliases_v7"
            " WHERE alias_canon = 'zeddy'",
        ) == 1


def test_graph_edges_removed_both_directions(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        make_source(conn, v7, SID, "src-2")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        seed_unit(conn, SID, "src-2", "u7:1:b")
        seed_unit(conn, SID, "src-2", "u7:1:c")
        conn.execute(
            "INSERT INTO graph_edges(scope_id, src_unit, type, dst_unit,"
            " generation, weight)"
            " VALUES (?, 'u7:1:a', 'supports', 'u7:1:b', 1, 1.0)",
            (SID,),
        )
        conn.execute(
            "INSERT INTO graph_edges(scope_id, src_unit, type, dst_unit,"
            " generation, weight)"
            " VALUES (?, 'u7:1:c', 'mentions', 'u7:1:a', 1, 1.0)",
            (SID,),
        )
        conn.execute(
            "INSERT INTO graph_edges(scope_id, src_unit, type, dst_unit,"
            " generation, weight)"
            " VALUES (?, 'u7:1:b', 'mentions', 'u7:1:c', 1, 1.0)",
            (SID,),
        )
        delete_source_v7(conn, "src-1")
        edges = qrows(
            conn, "SELECT src_unit, dst_unit FROM graph_edges"
        )
        assert edges == [{"src_unit": "u7:1:b", "dst_unit": "u7:1:c"}]


def test_unit_pinning_artifacts_retire_with_support(v7):
    """t2_facts/observations/profiles die with pinned support; an
    observation whose *contradict* ref died keeps the row minus the
    dead pin (support still stands — V7-14.08)."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        make_source(conn, v7, SID, "src-2")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        seed_unit(conn, SID, "src-2", "u7:1:b")
        conn.execute(
            "INSERT INTO t2_facts(fact_id, scope_id, unit_ids_json,"
            " generation, verified)"
            " VALUES ('f-gone', ?, ?, 1, 1)",
            (SID, json.dumps(["u7:1:a", "u7:1:b"])),
        )
        conn.execute(
            "INSERT INTO t2_facts(fact_id, scope_id, unit_ids_json,"
            " generation, verified)"
            " VALUES ('f-stays', ?, ?, 1, 1)",
            (SID, json.dumps(["u7:1:b"])),
        )
        conn.execute(
            "INSERT INTO observations_v7(obs_id, scope_id,"
            " support_refs_json, generation)"
            " VALUES ('obs-gone', ?, ?, 1)",
            (SID, json.dumps(["u7:1:a"])),
        )
        conn.execute(
            "INSERT INTO observations_v7(obs_id, scope_id,"
            " support_refs_json, contradict_refs_json, generation)"
            " VALUES ('obs-contra', ?, ?, ?, 1)",
            (SID, json.dumps(["u7:1:b"]), json.dumps(["u7:1:a"])),
        )
        conn.execute(
            "INSERT INTO profiles_v7(scope_id, subject_canon, slot,"
            " generation, value, support_refs_json)"
            " VALUES (?, 'alice', 'likes', 1, 'sushi', ?)",
            (SID, json.dumps(["u7:1:a"])),
        )
        delete_source_v7(conn, "src-1")
        assert qscalar(
            conn, "SELECT COUNT(*) FROM t2_facts WHERE fact_id='f-gone'"
        ) == 0
        assert qscalar(
            conn, "SELECT COUNT(*) FROM t2_facts WHERE fact_id='f-stays'"
        ) == 1
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM observations_v7 WHERE obs_id='obs-gone'",
        ) == 0
        contra = qrows(
            conn,
            "SELECT contradict_refs_json FROM observations_v7"
            " WHERE obs_id = 'obs-contra'",
        )
        assert json.loads(contra[0]["contradict_refs_json"]) == []
        assert qscalar(conn, "SELECT COUNT(*) FROM profiles_v7") == 0


def test_standing_queries_dirty_and_pack_scrubbed(v7):
    """The stored pack may embed the erased bytes verbatim — it is
    scrubbed and the query marked dirty for rebuild (V7-14.07); the
    query row itself is user state and survives."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        conn.execute(
            "INSERT INTO standing_queries(sq_id, scope_id, query,"
            " pack_blob, pack_digest, dirty, generation)"
            " VALUES ('sq-1', ?, 'q?', X'00DEAD', 'd1', 0, 1)",
            (SID,),
        )
        delete_source_v7(conn, "src-1")
        row = qrows(
            conn,
            "SELECT dirty, pack_blob, pack_digest FROM standing_queries"
            " WHERE sq_id = 'sq-1'",
        )[0]
        assert row["dirty"] == 1
        assert row["pack_blob"] is None
        assert row["pack_digest"] is None


def test_vector_block_rowmap_surgery(v7):
    """Deleting one unit rewrites the block minus that row — surviving
    rows keep byte-exact data (no requantization); deleting the last
    key drops the block row entirely."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        make_source(conn, v7, SID, "src-2")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        seed_unit(conn, SID, "src-2", "u7:1:b")
        seed_unit(conn, SID, "src-2", "u7:1:c")
        seed_vector_block(
            conn, SID, 1, "enc:1", 0, ["u7:1:a", "u7:1:b", "u7:1:c"]
        )
        delete_source_v7(conn, "src-1")
        row = qrows(
            conn,
            "SELECT n_rows, rowmap_blob, data_blob FROM"
            " unit_vectors_block WHERE encoder_id = 'enc:1'",
        )[0]
        assert row["n_rows"] == 2
        assert decode_rowmap(bytes(row["rowmap_blob"])) == [
            "u7:1:b", "u7:1:c",
        ]
        # bytes for rows 1..2 of the original f32×4 matrix survive
        assert bytes(row["data_blob"]) == bytes(range(16, 48))
        delete_source_v7(conn, "src-2")
        assert qscalar(
            conn, "SELECT COUNT(*) FROM unit_vectors_block"
        ) == 0


def test_audit_journal_retained_and_counted(v7):
    """screening_log rows are the write-channel journal — counted under
    audit_retained, never deleted (v1 events-journal rule)."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        conn.execute(
            "INSERT INTO screening_log(decision_id, unit_id, source_id,"
            " scope_id, outcome, generation)"
            " VALUES ('d-1', 'u7:1:a', 'src-1', ?, 'allow', 1)",
            (SID,),
        )
        conn.execute(
            "INSERT INTO screening_log(decision_id, unit_id, source_id,"
            " scope_id, outcome, generation)"
            " VALUES ('d-2', NULL, 'src-1', ?, 'label', 1)",
            (SID,),
        )
        out = delete_source_v7(conn, "src-1")
        assert out["audit_retained"]["screening_log"] == 2
        assert qscalar(conn, "SELECT COUNT(*) FROM screening_log") == 2


def test_delete_idempotent_second_run(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        first = delete_source_v7(conn, "src-1")
        second = delete_source_v7(conn, "src-1")
        assert first["units_removed"] == 1
        assert second["units_removed"] == 0
        assert second["deleted"].get("units", 0) == 0


def test_source_delete_validation_and_noop(v7):
    with v7.tx() as conn:
        with pytest.raises(VerbatimError) as exc:
            delete_source_v7(conn, "")
        assert exc.value.code == ErrorCode.VALIDATION
        out = delete_source_v7(conn, "src-absent")
        assert out["units_removed"] == 0


def test_noop_on_pre_v7_store(store):
    """Pre-V7 store: the sweep is a no-op, not an error — the same
    has_table gate every purge path relies on."""
    with store.tx() as conn:
        assert not has_table(conn, "units")
        out = delete_source_v7(conn, "src-1")
        assert out["units_removed"] == 0
        assert out["deleted"] == {}
        cov = v7_closure_coverage(conn)
        assert cov["present"] == []
        assert cov["complete"] is False


# ---------------------------------------------------------------------------
# scope erasure
# ---------------------------------------------------------------------------


def test_scope_delete_removes_whole_plane(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        conn.execute(
            "INSERT INTO entity_mentions(scope_id, canon, unit_id,"
            " generation, byte_start) VALUES (?, 'bob', ?, 1, 0)",
            (SID, "u7:1:a"),
        )
        conn.execute(
            "INSERT INTO entity_canon(scope_id, canon, generation,"
            " df_units) VALUES (?, 'bob', 1, 1)",
            (SID,),
        )
        conn.execute(
            "INSERT INTO entity_aliases_v7(scope_id, canon, alias_canon,"
            " generation, method) VALUES (?, 'bob', 'b', 1, 'rule')",
            (SID,),
        )
        conn.execute(
            "INSERT INTO graph_edges(scope_id, src_unit, type, dst_unit,"
            " generation) VALUES (?, 'u7:1:a', 'mentions', 'u7:1:a', 1)",
            (SID,),
        )
        seed_vector_block(conn, SID, 1, "enc:1", 0, ["u7:1:a"])
        conn.execute(
            "INSERT INTO t2_facts(fact_id, scope_id, unit_ids_json,"
            " generation) VALUES ('f-1', ?, '[\"u7:1:a\"]', 1)",
            (SID,),
        )
        conn.execute(
            "INSERT INTO observations_v7(obs_id, scope_id,"
            " support_refs_json, generation) VALUES ('o-1', ?, '[]', 1)",
            (SID,),
        )
        conn.execute(
            "INSERT INTO profiles_v7(scope_id, subject_canon, slot,"
            " generation) VALUES (?, 'alice', 'likes', 1)",
            (SID,),
        )
        conn.execute(
            "INSERT INTO standing_queries(sq_id, scope_id, dirty,"
            " generation) VALUES ('sq-1', ?, 0, 1)",
            (SID,),
        )
        seed_stats(conn, SID, 1, [
            ("u7:1:a", ("alpha beta", "alice", "", "", "")),
        ])
        conn.execute(
            "INSERT INTO screening_log(decision_id, scope_id, outcome,"
            " generation) VALUES ('d-1', ?, 'allow', 1)",
            (SID,),
        )
        out = delete_scope_v7(conn, SID)
        assert out["deleted"]["units"] == 1
        assert out["audit_retained"]["screening_log"] == 1
    with v7.read() as conn:
        for t in closure_v7.V7_CLOSURE_TABLES:
            n = qscalar(conn, f"SELECT COUNT(*) FROM {t}")
            assert n == 0, f"{t} still has {n} rows"
        assert qscalar(conn, "SELECT COUNT(*) FROM screening_log") == 1


def test_scope_delete_leaves_sibling_scope(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        make_source(conn, v7, OTHER, "src-2")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        seed_unit(conn, OTHER, "src-2", "u7:1:b")
        delete_scope_v7(conn, SID)
        assert qscalar(
            conn, "SELECT COUNT(*) FROM units WHERE scope_id = ?", (SID,)
        ) == 0
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM units WHERE scope_id = ?",
            (OTHER,),
        ) == 1


def test_scope_delete_generation_narrowed(v7):
    """``generation=`` removes exactly that plane — the newer
    generation's rows of the SAME scope survive (rebuild coexistence)."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:g1:a", gen=1)
        seed_unit(conn, SID, "src-1", "u7:g2:a", gen=2)
        delete_scope_v7(conn, SID, generation=1)
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM units WHERE scope_id = ?"
            " AND generation = 1",
            (SID,),
        ) == 0
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM units WHERE scope_id = ?"
            " AND generation = 2",
            (SID,),
        ) == 1


# ---------------------------------------------------------------------------
# generation fencing — the post-flip sweeper
# ---------------------------------------------------------------------------


def test_stale_generations_respect_committed_fence(v7):
    """While generation N is the committed fence, rows at N survive the
    sweeper — only generations strictly below are reclaimed.  After the
    fence flips (same-tx meta bump, V7-30.02), the retired plane goes."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:g1:a", gen=1)
        conn.execute(
            "INSERT INTO meta(key, value_json)"
            " VALUES ('projection_generation', '1')"
            " ON CONFLICT(key) DO UPDATE SET"
            "   value_json = excluded.value_json"
        )
        out = sweep_stale_generations_v7(conn)
        assert out["keep_generation"] == 1
        assert out["deleted"].get("units", 0) == 0
        assert qscalar(
            conn, "SELECT COUNT(*) FROM units WHERE generation = 1"
        ) == 1
        # flip the fence — the same call now reclaims gen 1
        conn.execute(
            "UPDATE meta SET value_json = '2'"
            " WHERE key = 'projection_generation'"
        )
        out = sweep_stale_generations_v7(conn)
        assert out["keep_generation"] == 2
        assert out["deleted"]["units"] == 1
        assert qscalar(conn, "SELECT COUNT(*) FROM units") == 0


def test_sweep_stale_explicit_keep_and_scope_narrowing(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        make_source(conn, v7, OTHER, "src-2")
        seed_unit(conn, SID, "src-1", "u7:g1:a", gen=1)
        seed_unit(conn, SID, "src-1", "u7:g3:a", gen=3)
        seed_unit(conn, OTHER, "src-2", "u7:g1:b", gen=1)
        out = sweep_stale_generations_v7(
            conn, scope_id=SID, keep_generation=3
        )
        assert out["deleted"]["units"] == 1
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM units WHERE scope_id = ?"
            " AND generation = 3",
            (SID,),
        ) == 1
        # sibling scope's stale plane is untouched by the narrowed sweep
        assert qscalar(
            conn,
            "SELECT COUNT(*) FROM units WHERE scope_id = ?",
            (OTHER,),
        ) == 1


def test_sweep_stale_skips_audit_journals(v7):
    """Audit journals carry generation=0 by design — the stale sweep
    must not eat them (they are not a generation plane)."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:g1:a", gen=1)
        conn.execute(
            "INSERT INTO screening_log(decision_id, scope_id, outcome,"
            " generation) VALUES ('d-1', ?, 'allow', 0)",
            (SID,),
        )
        conn.execute(
            "INSERT INTO run_manifests(digest, manifest_json,"
            " generation) VALUES ('m-1', '{}', 0)",
        )
        out = sweep_stale_generations_v7(conn, keep_generation=5)
        assert qscalar(
            conn, "SELECT COUNT(*) FROM screening_log"
        ) == 1
        assert qscalar(
            conn, "SELECT COUNT(*) FROM run_manifests"
        ) == 1
        assert "screening_log" not in out["deleted"]


# ---------------------------------------------------------------------------
# quarantine / suppression cascade — the eligibility seam
# ---------------------------------------------------------------------------


def test_held_unit_ids_direct_unit_hold(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        seed_unit(conn, SID, "src-1", "u7:1:b")
        quarantine(conn, SID, "unit", "u7:1:a")
        assert held_unit_ids(conn, SID) == {"u7:1:a"}
        # released holds stop hiding
        conn.execute(
            "UPDATE quarantine SET state = 'released'"
            " WHERE object_id = 'u7:1:a'"
        )
        assert held_unit_ids(conn, SID) == set()


def test_held_unit_ids_source_hold_cascades(v7):
    """A hold on the source (revision 0 = whole object) withholds every
    unit the source projected — the V3-14.10 cascade on the V7 plane."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a", rev=1)
        seed_unit(conn, SID, "src-1", "u7:1:b", rev=2)
        quarantine(conn, SID, "source", "src-1", rev=0)
        assert held_unit_ids(conn, SID) == {"u7:1:a", "u7:1:b"}


def test_held_unit_ids_source_revision_exact(v7):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a", rev=1)
        seed_unit(conn, SID, "src-1", "u7:1:b", rev=2)
        quarantine(conn, SID, "source_revision", "src-1:2", rev=2)
        assert held_unit_ids(conn, SID) == {"u7:1:b"}
        # explicit ref form ("source", sid, exact rev) is equivalent
        assert held_unit_ids(
            conn, SID, hold_refs=[("source", "src-1", 2)]
        ) == {"u7:1:b"}


def test_held_unit_ids_span_and_envelope_resolve(v7):
    """Span/envelope holds resolve through their covering (source_id,
    revision) — a hold on the envelope hides the revision's units."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        make_source(conn, v7, SID, "src-1", rev=2, payload=b"rev two")
        seed_unit(conn, SID, "src-1", "u7:1:a", rev=1)
        seed_unit(conn, SID, "src-1", "u7:1:b", rev=2)
        conn.execute(
            "INSERT INTO spans(span_id, source_id, revision,"
            " start_byte, end_byte, excerpt_hmac, harvester_version)"
            " VALUES ('span-1', 'src-1', 2, 0, 4, X'00', 'h1')",
        )
        conn.execute(
            "INSERT INTO source_envelopes(envelope_id, source_id,"
            " revision, scope_id, envelope_kind, event_us, receipt_us,"
            " media_type, trust_class, redaction_status,"
            " adapter_version, host_id, session_id, task_id, step_id,"
            " metadata_json)"
            " VALUES ('env-1', 'src-1', 1, ?, 'user_message', 1, 1,"
            " 'text/plain', 'external_content', 'none', 'a1', 'h1', 's1',"
            " 't1', 'st1', '{}')",
            (SID,),
        )
        quarantine(conn, SID, "envelope", "env-1")
        assert held_unit_ids(conn, SID) == {"u7:1:a"}
        quarantine(conn, SID, "span", "span-1", rev=2)
        assert held_unit_ids(conn, SID) == {"u7:1:a", "u7:1:b"}


def test_held_unit_ids_purge_tombstone_hides(v7):
    """The suppress→execute window: a suppressed purge target hides its
    units before the sweep lands (V2-41.03 tombstone semantics)."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        conn.execute(
            "INSERT INTO purges(purge_id, selection_digest, scope_id,"
            " state, requested_us) VALUES ('p-1', 'd', ?, 'suppressed', 1)",
            (SID,),
        )
        conn.execute(
            "INSERT INTO purge_targets(purge_id, object_kind, object_id)"
            " VALUES ('p-1', 'source', 'src-1')",
        )
        assert held_unit_ids(conn, SID) == {"u7:1:a"}
        # a released (previewed) purge no longer suppresses
        conn.execute(
            "UPDATE purges SET state = 'previewed' WHERE purge_id = 'p-1'"
        )
        assert held_unit_ids(conn, SID) == set()


def test_held_unit_ids_scope_isolated_and_validation(v7, store):
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        quarantine(conn, SID, "unit", "u7:1:a")
        assert held_unit_ids(conn, OTHER) == set()
        with pytest.raises(VerbatimError):
            held_unit_ids(conn, "")
    # pre-V7 store: no units table → empty, never an error
    with store.tx() as conn:
        assert held_unit_ids(conn, "scope:v7") == set()


# ---------------------------------------------------------------------------
# end-to-end: the wired purge path carries the V7 sweep
# ---------------------------------------------------------------------------


def test_execute_purge_sweeps_v7_plane(v7):
    """The public purge path — plan → execute — erases the source's
    bytes AND its whole V7 projection in one transaction; the receipt
    discloses the per-table counts under ``derived.v7_*``."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        conn.execute(
            "INSERT INTO entity_mentions(scope_id, canon, unit_id,"
            " generation, byte_start) VALUES (?, 'bob', ?, 1, 0)",
            (SID, "u7:1:a"),
        )
        conn.execute(
            "INSERT INTO entity_canon(scope_id, canon, generation,"
            " df_units) VALUES (?, 'bob', 1, 1)",
            (SID,),
        )
    plan = plan_purge(v7, SID, [("source", "src-1")], actor="alice")
    result = execute_purge(v7, plan["purge_id"])
    assert result["state"] == "completed"
    with v7.read() as conn:
        assert qscalar(conn, "SELECT COUNT(*) FROM units") == 0
        assert qscalar(conn, "SELECT COUNT(*) FROM unit_fts_rows") == 0
        assert qscalar(conn, "SELECT COUNT(*) FROM unit_fts_content") == 0
        assert qscalar(conn, "SELECT COUNT(*) FROM entity_mentions") == 0
        assert qscalar(conn, "SELECT COUNT(*) FROM entity_canon") == 0
        assert qscalar(conn, "SELECT COUNT(*) FROM events_v7") == 0
    assert result["derived"].get("v7_units", 0) >= 1


def test_integrity_counts_report_v7_tables(v7):
    """check_integrity names every V7 table — None only on stores where
    the additive schema was never ensured."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
    counts = v7.check_integrity()["counts"]
    for t in V7_TABLES + V7_INTERNAL_TABLES:
        assert t in counts, f"{t} missing from integrity counts"
        assert counts[t] is not None
    assert counts["units"] == 1
    assert counts["unit_fts_rows"] == 1


def test_export_never_exported_lists_v7():
    """The export exclusion documents the whole §30 plane — bundles ship
    evidence sections; the derived plane is recomputable, never shipped."""
    from verbatim.export import NEVER_EXPORTED

    listed = set(NEVER_EXPORTED)
    for t in V7_TABLES + V7_INTERNAL_TABLES:
        assert t in listed, f"{t} undocumented in NEVER_EXPORTED"


def test_result_shape_discloses_everything(v7):
    """The sweep receipt carries per-table counts, audit-retained
    counts, the unhandled list, and the affected scopes — nothing is
    silently claimed."""
    with v7.tx() as conn:
        make_source(conn, v7, SID, "src-1")
        seed_unit(conn, SID, "src-1", "u7:1:a")
        out = delete_source_v7(conn, "src-1")
        assert out["units_removed"] == 1
        assert out["scopes"] == [SID]
        assert out["source_id"] == "src-1"
        assert out["unhandled_v7"] == []
        assert out["deleted"]["units"] == 1
