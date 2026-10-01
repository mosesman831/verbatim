"""Lane tests for ``verbatim/retrieval/v7/exact_id.py`` (w-lane-exactid).

Mirror-DDL approach per the wave brief: the §30 ``units`` /
``unit_fts_rows`` / ``unit_fts_content`` wiring plus the v1 ``sources`` /
``source_revisions`` id tables are created inline at their real shapes.
``QueryViewV7`` is hand-built with ``norm.identifiers`` already populated
— the analyzer (``norm/v2``) is never called here. These tests unit-test
the LANE.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from verbatim.core.types_v7 import (
    BudgetClass,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneSlice,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    RetrievalPolicyV7,
)
from verbatim.retrieval.v7 import exact_id as xlane
from verbatim.retrieval.v7.lanes_base import LANE_REGISTRY, run_one

# ---------------------------------------------------------------------------
# §30 mirror DDL is built inline in ``mk_conn`` at the real schema_v7 shapes
# (units + unit_fts_rows carrier + unit_fts_content external content, plus
# the v1 sources / source_revisions id tables) so each fixture can drop a
# piece and exercise the honest-degradation branches.
# ---------------------------------------------------------------------------

SCOPE = "s1"
GEN = 1


def us(y: int, m: int = 1, d: int = 1) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1_000_000)


def mk_conn(with_fts: bool = True, with_sources: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE units ("
        " unit_id TEXT NOT NULL, source_id TEXT NOT NULL,"
        " revision INTEGER NOT NULL, scope_id TEXT NOT NULL, kind TEXT,"
        " parent_unit_id TEXT, session_id TEXT, seq INTEGER,"
        " speaker_canon TEXT, perspective TEXT, recorded_at_us INTEGER,"
        " occurred_start_us INTEGER, occurred_end_us INTEGER,"
        " occurred_precision TEXT, occurred_source TEXT, byte_start INTEGER,"
        " byte_end INTEGER, generation INTEGER NOT NULL,"
        " PRIMARY KEY (unit_id, generation))"
    )
    if with_fts:
        conn.executescript(
            "CREATE TABLE unit_fts_rows ("
            " row_id INTEGER PRIMARY KEY, unit_id TEXT NOT NULL,"
            " scope_id TEXT NOT NULL, generation INTEGER NOT NULL,"
            " UNIQUE (unit_id, generation));"
            "CREATE TABLE unit_fts_content ("
            " fts_row_id INTEGER PRIMARY KEY REFERENCES unit_fts_rows(row_id),"
            " text TEXT, speaker TEXT, entities TEXT, session TEXT,"
            ' "when" TEXT);'
        )
    if with_sources:
        conn.executescript(
            "CREATE TABLE sources ("
            " source_id TEXT PRIMARY KEY, origin TEXT, external_id TEXT,"
            " source_kind TEXT, scope_id TEXT, speaker_id TEXT,"
            " created_us INTEGER);"
            "CREATE TABLE source_revisions ("
            " source_id TEXT NOT NULL, revision INTEGER NOT NULL,"
            " payload BLOB, payload_hmac BLOB, event_us INTEGER,"
            " captured_us INTEGER, PRIMARY KEY (source_id, revision));"
        )
    return conn


def add_unit(
    conn,
    unit_id,
    *,
    text=None,
    source="src",
    rev=1,
    scope=SCOPE,
    gen=GEN,
    kind="turn",
):
    conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " recorded_at_us, generation) VALUES (?,?,?,?,?,?,?)",
        (unit_id, source, rev, scope, kind, us(2024, 1, 1), gen),
    )
    if text is not None:
        cur = conn.execute(
            "INSERT INTO unit_fts_rows(unit_id, scope_id, generation)"
            " VALUES (?,?,?)",
            (unit_id, scope, gen),
        )
        conn.execute(
            'INSERT INTO unit_fts_content(fts_row_id, text) VALUES (?,?)',
            (cur.lastrowid, text),
        )


def add_source(conn, source_id, *, external_id=None, scope=SCOPE):
    conn.execute(
        "INSERT INTO sources(source_id, origin, external_id, source_kind,"
        " scope_id, created_us) VALUES (?,?,?,?,?,?)",
        (source_id, "test", external_id, "import", scope, us(2024, 1, 1)),
    )


def mk_qv(identifiers=(), *, q="test query", primary=IntentClass.IDENTIFIER):
    norm = NormAnalysis(
        analyzer_id="norm/v2",
        terms=(),
        identifiers=tuple(
            NormTerm(s, "identifier", 0, len(s)) for s in identifiers
        ),
    )
    intent = IntentResult(primary=primary, classes=(primary,))
    return QueryViewV7(
        query=q, norm=norm, intent=intent, query_time_us=us(2024, 1, 1)
    )


def mk_ctx(conn, *, eligible=None, gen=GEN, scope=SCOPE):
    if eligible is None:
        eligible = lambda row: True  # noqa: E731 - allow-all fixture
    policy = RetrievalPolicyV7(
        policy_id="retrieval_policy/v7",
        profile="test",
        lanes=(LaneName.EXACT_ID,),
        lane_weights={},
    )
    return LaneContextV7(
        store=conn,
        scope_id=scope,
        generation=gen,
        eligible=eligible,
        query_time_us=us(2024, 1, 1),
        profile="test",
        budget=BudgetClass.MID,
        policy=policy,
    )


def sl(cap=50, ms=10_000):
    return LaneSlice(deadline_ms=ms, cap=cap)


def by_id(out):
    return {c.unit_id: c for c in out.candidates}


# ---------------------------------------------------------------------------
# byte-exact substring matching
# ---------------------------------------------------------------------------


def test_version_string_matches_byte_exact():
    conn = mk_conn()
    add_unit(conn, "u_hit", text="upgrade to v2.4.1 today please")
    add_unit(conn, "u_miss", text="no version mentioned here")

    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2.4.1",)), sl())
    assert out.status == LaneStatus.OK
    ids = by_id(out)
    assert set(ids) == {"u_hit"}
    assert ids["u_hit"].signals["identifier"] == "v2.4.1"
    assert ids["u_hit"].signals["match"] == "instr"
    assert ids["u_hit"].signals["df"] == 1
    assert ids["u_hit"].rank == 1
    assert out.examined == 2 and out.eligible == 1


def test_identifier_channel_discipline_v2_never_v3():
    """The channel exists because folding loses v2 vs v3: an identifier
    ``v2`` must not match a unit that only contains ``v3``."""
    conn = mk_conn()
    add_unit(conn, "u_v3", text="we are on v3 now")
    add_unit(conn, "u_v2", text="still pinned to v2 here")

    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2",)), sl())
    ids = by_id(out)
    assert "u_v3" not in ids
    assert "u_v2" in ids  # byte-exact substring hit


def test_no_folding_case_is_byte_exact():
    """``V2.4.1`` (uppercase V) must NOT match lowercase ``v2.4.1`` —
    unicode61 folds case; the identifier channel never does."""
    conn = mk_conn()
    add_unit(conn, "u_lower", text="release v2.4.1 shipped")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("V2.4.1",)), sl())
    assert "u_lower" not in by_id(out)


def test_no_separator_normalization():
    """FTS5 phrase ``"v2.4.1"`` tokenizes v2/4/1 and wrongly matches
    ``v2-4-1``; the byte-exact path must not."""
    conn = mk_conn()
    add_unit(conn, "u_dash", text="the v2-4-1 build was yanked")
    add_unit(conn, "u_dot", text="the v2.4.1 build shipped")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2.4.1",)), sl())
    assert set(by_id(out)) == {"u_dot"}


def test_glued_token_boundary_still_byte_exact():
    """``xv2.4.1`` contains the byte-exact surface — a case FTS5 phrase
    misses (tokenizes as xv2/4/1). Substring semantics are honored."""
    conn = mk_conn()
    add_unit(conn, "u_glued", text="see build-xv2.4.1 artifact")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2.4.1",)), sl())
    assert "u_glued" in by_id(out)


def test_path_identifier_matches():
    conn = mk_conn()
    add_unit(conn, "u_path", text="edit /etc/foo.conf then restart")
    add_unit(conn, "u_other", text="edit /etc/bar.conf instead")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("/etc/foo.conf",)), sl())
    assert set(by_id(out)) == {"u_path"}


def test_code_span_matches_wrapped_and_core():
    """`` `foo()` `` matches both the backtick-wrapped source occurrence
    and the bare ``foo()`` core — the backticks are quotation."""
    conn = mk_conn()
    add_unit(conn, "u_wrapped", text="run `foo()` after init")
    add_unit(conn, "u_bare", text="call foo() whenever ready")
    add_unit(conn, "u_none", text="call bar() instead")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("`foo()`",)), sl())
    ids = by_id(out)
    assert {"u_wrapped", "u_bare"} <= set(ids)
    assert "u_none" not in ids


def test_ticket_and_hash_identifiers():
    conn = mk_conn()
    add_unit(conn, "u_ticket", text="tracked under ABC-123 since May")
    add_unit(conn, "u_hash", text="commit 0xdeadbeef fixed it")
    out = xlane.lane_exact_id(
        mk_ctx(conn), mk_qv(("ABC-123", "0xdeadbeef")), sl()
    )
    ids = by_id(out)
    assert set(ids) == {"u_ticket", "u_hash"}
    assert ids["u_ticket"].signals["identifier"] == "ABC-123"
    assert ids["u_hash"].signals["identifier"] == "0xdeadbeef"


# ---------------------------------------------------------------------------
# direct id lookups (D7-20)
# ---------------------------------------------------------------------------


def test_unit_id_lookup_emits_rank1():
    conn = mk_conn()
    add_unit(conn, "doc-77", text="contents irrelevant here")
    add_unit(conn, "u_other", text="some other text with doc-77 inside")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("doc-77",)), sl())
    assert out.status == LaneStatus.OK
    top = out.candidates[0]
    assert top.unit_id == "doc-77"
    assert top.rank == 1
    assert top.signals["match"] == "id_lookup"
    assert top.signals["id_kind"] == "unit_id"


def test_source_id_lookup_emits_its_units():
    conn = mk_conn()
    add_unit(conn, "u_a1", text="first rev", source="src-doc-9", rev=1)
    add_unit(conn, "u_a2", text="second rev", source="src-doc-9", rev=2)
    add_unit(conn, "u_b", text="unrelated", source="src-other", rev=1)
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("src-doc-9",)), sl())
    ids = by_id(out)
    assert set(ids) == {"u_a1", "u_a2"}
    assert all(c.signals["match"] == "id_lookup" for c in out.candidates)
    assert all(c.signals["id_kind"] == "source_id" for c in out.candidates)


def test_external_id_lookup_resolves_source():
    conn = mk_conn()
    add_source(conn, "src-int", external_id="ext-doc-42")
    add_unit(conn, "u_x1", text="alpha", source="src-int", rev=1)
    add_unit(conn, "u_x2", text="beta", source="src-int", rev=2)
    add_unit(conn, "u_y", text="gamma", source="src-ext", rev=1)
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("ext-doc-42",)), sl())
    ids = by_id(out)
    assert set(ids) == {"u_x1", "u_x2"}
    assert all(c.signals["id_kind"] == "external_id" for c in out.candidates)


def test_source_revision_pattern():
    conn = mk_conn()
    add_unit(conn, "u_r1", text="old", source="chat-log", rev=1)
    add_unit(conn, "u_r2", text="new", source="chat-log", rev=2)
    add_unit(conn, "u_r3", text="other", source="chat-log", rev=3)
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("chat-log@2",)), sl())
    ids = by_id(out)
    assert set(ids) == {"u_r2"}
    assert ids["u_r2"].signals["id_kind"] == "source_revision"


def test_bare_version_never_parses_as_revision():
    """``v2`` / ``ABC-123`` must never be read as (source, revision)."""
    conn = mk_conn()
    add_unit(conn, "u1", text="we pinned v2", source="v2", rev=2)
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2",)), sl())
    # the source_id equality hit is legitimate; what must NOT exist is a
    # "source_revision" interpretation of the bare token
    for c in out.candidates:
        assert c.signals["id_kind"] != "source_revision"


# ---------------------------------------------------------------------------
# honesty: skipped / unavailable / fences / eligibility
# ---------------------------------------------------------------------------


def test_no_identifiers_skipped():
    conn = mk_conn()
    add_unit(conn, "u1", text="v2.4.1 everywhere")
    qv = mk_qv((), primary=IntentClass.LOOKUP)
    out = xlane.lane_exact_id(mk_ctx(conn), qv, sl())
    assert out.status == LaneStatus.SKIPPED
    assert out.reason == "no_identifiers"
    assert out.candidates == [] and out.examined == 0


def test_unavailable_without_read_snapshot():
    ctx = mk_ctx(mk_conn())
    ctx.store = object()  # no pinned connection anywhere
    out = xlane.lane_exact_id(ctx, mk_qv(("v2",)), sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_read_snapshot"


def test_unavailable_generation_unpinned():
    conn = mk_conn()
    ctx = mk_ctx(conn)
    ctx.generation = None
    out = xlane.lane_exact_id(ctx, mk_qv(("v2",)), sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "generation_unpinned"


def test_unavailable_units_missing():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE unit_fts_rows (row_id INTEGER PRIMARY KEY)")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2",)), sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "units_table_missing"


def test_unavailable_text_index_missing_no_hits():
    """units present but no fts carrier/content: a pure text identifier
    cannot be matched and resolves no id -> honest unavailable."""
    conn = mk_conn(with_fts=False)
    add_unit(conn, "u1", text=None)  # units only, no text rows
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2.4.1",)), sl())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "text_index_missing"


def test_partial_text_index_missing_with_id_hit():
    """id lookups still emit when the text index is absent — real
    evidence, honestly labeled partial."""
    conn = mk_conn(with_fts=False)
    add_unit(conn, "doc-5", text=None, source="srcA", rev=1)
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("doc-5",)), sl())
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "text_index_missing"
    assert by_id(out)["doc-5"].signals["match"] == "id_lookup"


def test_eligibility_gate_held_unit_never_returned():
    conn = mk_conn()
    add_unit(conn, "u_ok", text="pinned at v2.4.1")
    add_unit(conn, "u_held", text="also pinned at v2.4.1")
    held = {"u_held"}
    ctx = mk_ctx(conn, eligible=lambda row: row["unit_id"] not in held)
    out = xlane.lane_exact_id(ctx, mk_qv(("v2.4.1",)), sl())
    ids = by_id(out)
    assert "u_ok" in ids and "u_held" not in ids
    # df is a corpus statistic: counted over the fenced set regardless
    assert out.stats["df"] == {"v2.4.1": 2}
    assert out.eligible == 1


def test_eligibility_set_object():
    conn = mk_conn()
    add_unit(conn, "u_a", text="/etc/foo.conf here")
    add_unit(conn, "u_b", text="/etc/foo.conf there")
    ctx = mk_ctx(conn, eligible={"u_a"})
    out = xlane.lane_exact_id(ctx, mk_qv(("/etc/foo.conf",)), sl())
    assert set(by_id(out)) == {"u_a"}


def test_generation_fence():
    """V7-30.02: ``generation <= pinned`` — a row written at gen 1 stays
    visible at the gen-2 pin (distinct unit_ids, not superseded); rows
    ABOVE the pin stay fenced out."""
    conn = mk_conn()
    add_unit(conn, "u_g1", text="v2.4.1 gen one", gen=1)
    add_unit(conn, "u_g2", text="v2.4.1 gen two", gen=2)
    out = xlane.lane_exact_id(mk_ctx(conn, gen=1), mk_qv(("v2.4.1",)), sl())
    assert set(by_id(out)) == {"u_g1"}  # u_g2 is above the pin
    out2 = xlane.lane_exact_id(mk_ctx(conn, gen=2), mk_qv(("v2.4.1",)), sl())
    assert set(by_id(out2)) == {"u_g1", "u_g2"}


def test_scope_fence():
    conn = mk_conn()
    add_unit(conn, "u_s1", text="v2.4.1 in scope", scope=SCOPE)
    add_unit(conn, "u_s2", text="v2.4.1 elsewhere", scope="s2")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2.4.1",)), sl())
    assert set(by_id(out)) == {"u_s1"}


# ---------------------------------------------------------------------------
# scoring / ranking determinism
# ---------------------------------------------------------------------------


def test_df_rarity_bonus_orders_rare_first():
    """``10.0 + 1/(1+df)``: the rare identifier's hit outranks the
    common identifier's hit."""
    conn = mk_conn()
    add_unit(conn, "u_rare", text="rare tag ERR-909 only here")
    for i in range(5):
        add_unit(conn, f"u_common_{i}", text="everyone knows ERR-500")
    out = xlane.lane_exact_id(
        mk_ctx(conn), mk_qv(("ERR-909", "ERR-500")), sl()
    )
    ids = by_id(out)
    assert out.candidates[0].unit_id == "u_rare"
    assert ids["u_rare"].raw_score > ids["u_common_0"].raw_score
    assert ids["u_rare"].signals["df"] == 1
    assert ids["u_common_0"].signals["df"] == 5
    assert out.candidates[0].raw_score == pytest.approx(10.0 + 1.0 / 2.0)


def test_multi_identifier_match_bonus():
    conn = mk_conn()
    add_unit(conn, "u_both", text="v2.4.1 and ERR-909 together")
    add_unit(conn, "u_one", text="just v2.4.1 alone")
    out = xlane.lane_exact_id(
        mk_ctx(conn), mk_qv(("v2.4.1", "ERR-909")), sl()
    )
    ids = by_id(out)
    assert out.candidates[0].unit_id == "u_both"
    assert ids["u_both"].raw_score > ids["u_one"].raw_score
    assert sorted(ids["u_both"].signals["identifiers"]) == ["ERR-909", "v2.4.1"]


def test_deterministic_tiebreak_and_repeat():
    conn = mk_conn()
    for i in range(6):
        add_unit(conn, f"u{i:02d}", text="all at v2.4.1")
    qv = mk_qv(("v2.4.1",))
    ctx = mk_ctx(conn)
    first = [(c.unit_id, c.rank, c.raw_score) for c in
             xlane.lane_exact_id(ctx, qv, sl(cap=4)).candidates]
    second = [(c.unit_id, c.rank, c.raw_score) for c in
              xlane.lane_exact_id(ctx, qv, sl(cap=4)).candidates]
    assert first == second
    assert [u for u, _r, _s in first] == ["u00", "u01", "u02", "u03"]
    assert [r for _u, r, _s in first] == [1, 2, 3, 4]


def test_cap_truncated_stats():
    conn = mk_conn()
    for i in range(5):
        add_unit(conn, f"u{i}", text="v2.4.1")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2.4.1",)), sl(cap=2))
    assert len(out.candidates) == 2
    assert out.stats["cap_truncated"] == 3
    assert out.stats["overflow"] == 3
    assert out.eligible == 5


def test_deadline_partial():
    conn = mk_conn()
    add_unit(conn, "u1", text="v2.4.1")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2.4.1",)), sl(ms=0))
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "deadline"
    assert out.examined == 0  # honest: nothing was scanned


def test_zero_cap_is_honest_empty():
    conn = mk_conn()
    add_unit(conn, "u1", text="v2.4.1")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2.4.1",)), sl(cap=0))
    assert out.status == LaneStatus.OK
    assert out.candidates == []
    assert out.stats["mode"] == "capped"


def test_deadline_mid_scan_partial(monkeypatch):
    """A deadline that lands mid-scan reports partial with honest
    examined counts — the scan does not run to completion."""
    conn = mk_conn()
    for i in range(xlane.DEADLINE_CHECK_ROWS + 10):
        add_unit(conn, f"u{i:04d}", text="no identifier here")
    # scripted clock: t_end fix + the pre-gate/id-loop checks read 0.0,
    # then the scan-loop check reads past the deadline
    calls = iter([0.0, 0.0, 0.0] + [999.0] * 10)
    monkeypatch.setattr(xlane, "_monotonic", lambda: next(calls, 999.0))
    out = xlane.lane_exact_id(
        mk_ctx(conn), mk_qv(("v2.4.1",)), sl(ms=10_000)
    )
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "deadline"
    assert out.examined == xlane.DEADLINE_CHECK_ROWS
    assert out.examined < xlane.DEADLINE_CHECK_ROWS + 10


def test_scan_bound_partial(monkeypatch):
    monkeypatch.setattr(xlane, "SCAN_ROW_LIMIT", 5)
    conn = mk_conn()
    for i in range(8):
        add_unit(conn, f"u{i}", text="v2.4.1")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2.4.1",)), sl())
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "scan_bound"
    assert out.examined == 5  # bound counts rows examined, not emitted
    assert out.candidates  # the scanned matches still ship


# ---------------------------------------------------------------------------
# registry / pipeline integration
# ---------------------------------------------------------------------------


def test_lane_registered_at_import():
    assert LANE_REGISTRY.get(LaneName.EXACT_ID) is xlane.lane_exact_id


def test_run_one_enforces_cap_and_status():
    conn = mk_conn()
    for i in range(4):
        add_unit(conn, f"u{i}", text="v2.4.1")
    ctx = mk_ctx(conn)
    out = run_one(ctx, mk_qv(("v2.4.1",)), LaneName.EXACT_ID, sl(cap=3))
    assert out.status == LaneStatus.OK
    assert len(out.candidates) == 3
    assert all(c.lane == "exact_id" for c in out.candidates)


def test_examined_counts_scan_and_lookup_rows():
    conn = mk_conn()
    add_unit(conn, "doc-1", text="nothing matching", source="srcZ")
    add_unit(conn, "u2", text="unrelated", source="srcZ")
    # "doc-1" resolves as a unit_id (1 row) AND its source scan covers both
    # fenced text rows
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("doc-1",)), sl())
    assert out.examined >= 3  # 1 unit_id hit + 2 source_id hits + 2 text rows
    assert out.stats["text_path"] in ("fts_content", "fts_table")


def test_identifier_truncation_is_honest():
    conn = mk_conn()
    add_unit(conn, "u1", text="v2.4.1")
    many = tuple(f"ident-{i:03d}" for i in range(xlane.MAX_IDENTIFIERS + 5))
    qv = mk_qv(many + ("v2.4.1",))
    out = xlane.lane_exact_id(mk_ctx(conn), qv, sl())
    assert out.stats["identifiers_truncated"] is True
    assert out.stats["n_identifiers"] == xlane.MAX_IDENTIFIERS


def test_null_text_never_matches():
    conn = mk_conn()
    add_unit(conn, "u_null", text=None)  # fts carrier row w/ NULL content
    conn.execute(
        "INSERT INTO unit_fts_rows(unit_id, scope_id, generation)"
        " VALUES ('u_null2', 's1', 1)"
    )
    conn.execute(
        'INSERT INTO unit_fts_content(fts_row_id, text)'
        " VALUES (LAST_INSERT_ROWID(), NULL)"
    )
    add_unit(conn, "u_ok", text="v2.4.1 present")
    out = xlane.lane_exact_id(mk_ctx(conn), mk_qv(("v2.4.1",)), sl())
    assert set(by_id(out)) == {"u_ok"}


def test_lane_class_wrapper():
    conn = mk_conn()
    add_unit(conn, "u1", text="v2.4.1")
    lane = xlane.ExactIdLane()
    out = lane.run(mk_ctx(conn), mk_qv(("v2.4.1",)), sl())
    assert out.status == LaneStatus.OK and "u1" in by_id(out)


def test_store_wrapper_with_conn_attribute():
    conn = mk_conn()
    add_unit(conn, "u1", text="v2.4.1")
    ctx = mk_ctx(conn)
    ctx.store = SimpleNamespace(conn=conn)
    out = xlane.lane_exact_id(ctx, mk_qv(("v2.4.1",)), sl())
    assert "u1" in by_id(out)
