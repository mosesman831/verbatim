"""V8 lexical-lane verification tests — SPEC_V8 §07 / §21.4 / §24 / §25
scenarios K37–K42.

These exercise ``verbatim/retrieval/v7/lexical.py`` against the sibling
module's real SQLite mirror (``units`` + FTS5 shadows + maintained
``lex_stats``/``lex_df``) plus the V8 ``unit_doclen`` mirror — no mocks.
The lane-miss forensic (K37) is exercised through its real pure
classifier ``eval.v7.forensics.lane_miss.classify_lane_miss`` fed with
evidence assembled from real lane runs and real FTS postings (the same
primitives ``_LaneProbe`` replays).

- K37: every constructed lane_miss evidence record earns exactly one
  class from ``a``–``g``; a healthy run's eligibility class count is 0
  and a denied gold is provably caught as ``f``.
- K38: df-gate exemptions — speaker/entity canons, identifiers, the
  rarest content term, and the last-nominating-term floor — plus honest
  df reporting for gated terms and the ``off`` arm.
- K39: bounded rescue — fires on starved nomination, scans ≤ 2 rarest
  gated/dropped terms, honors ``rescue_rows`` and the deadline, fences
  to eligible rows, and reports ``coverage.lexical.rescue``.
- K40: with ``unit_doclen`` populated, zero ``unit_fts_docsize``
  statements issue and BM25F scores are byte-equal; doclen reads are
  generation-fenced; the ``single_match`` arm yields identical
  nomination sets on the vocab and fields paths.
- K41: warm-process stem computation is bounded (≤ 50/query) by the
  process LRU; memoized stems equal uncached stems per token; the
  ``stem_lru`` bound is real and score-neutral.
- K42: ``PoolProfile.prf_round`` removal (V8-07.07) — strict xfail
  while the field still ships.
"""

from __future__ import annotations

import dataclasses
import inspect
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verbatim.core.types_v7 import (  # noqa: E402
    BudgetClass,
    LaneContextV7,
    LaneName,
    LaneStatus,
    QueryViewV7,
    RetrievalPolicyV7,
)
from verbatim.retrieval.v7 import lexical as lex  # noqa: E402
from verbatim.retrieval.v7.lexical import lane_lexical  # noqa: E402

from tests.retrieval.v7 import test_lexical as fx  # noqa: E402
from eval.v7.forensics.lane_miss import (  # noqa: E402
    CLASS_LABELS,
    CLASS_ORDER,
    classify_lane_miss,
)

SCOPE = "s"
GEN = 1

# SPEC_V8 §19 — the V8-07.04 maintained field-length table, mirrored.
_DOCLEN_DDL = """
CREATE TABLE unit_doclen (
    unit_id    TEXT    NOT NULL,
    generation INTEGER NOT NULL,
    scope_id   TEXT    NOT NULL,
    field      TEXT    NOT NULL,
    len        INTEGER NOT NULL CHECK (len >= 0),
    PRIMARY KEY (unit_id, generation, field)
);
"""
_DOCLEN_FIELDS = ("text", "entities", "when", "speaker", "session")


def mkctx8(conn, *, eligible=None, scope=SCOPE, gen=GEN, manifest=None,
           nominate_terms_max=None, df_theta=None):
    """Sibling ``mkctx`` extended with the V8 arm carriers: policy
    ``nominate_terms_max``/``nominate_df_theta`` and the manifest keys
    ``_lexical_arms`` resolves (``lexical.df_floor`` etc.)."""
    kw = {}
    if nominate_terms_max is not None:
        kw["nominate_terms_max"] = nominate_terms_max
    if df_theta is not None:
        kw["nominate_df_theta"] = df_theta
    return LaneContextV7(
        store=conn, scope_id=scope, generation=gen, eligible=eligible,
        query_time_us=0, profile="test", budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="test",
            lanes=(LaneName.LEX, LaneName.FUZZY), lane_weights={}, **kw),
        manifest=dict(manifest or {}))


def seed_df(conn, term, df, *, field="text", scope=SCOPE, gen=GEN):
    """A maintained ``lex_df`` row — the pre-fetch gate's df input."""
    conn.execute(
        "INSERT INTO lex_df(scope_id,generation,field,term,"
        "stats_version,df) VALUES(?,?,?,?,?,?)"
        " ON CONFLICT(scope_id,generation,field,term,stats_version)"
        " DO UPDATE SET df=excluded.df",
        (scope, gen, field, term, fx.STATS_VERSION, df))


def write_doclen(conn, unit_id, fields, *, scope=SCOPE, gen=GEN,
                 overrides=None):
    """Mirror ``units_jobs._write_doclen``: one row per populated field,
    ``len`` = the ``_stat_terms`` whitespace fold; ``overrides`` lets a
    test write a deliberate newer-generation value."""
    overrides = overrides or {}
    for f in _DOCLEN_FIELDS:
        txt = fields.get(f) or ""
        n = overrides.get(f, len(str(txt).split()))
        conn.execute(
            "INSERT INTO unit_doclen(unit_id,generation,scope_id,"
            "field,len) VALUES(?,?,?,?,?)",
            (unit_id, gen, scope, f, n))


def _gate(out):
    return (out.stats.get("coverage_lexical") or {}).get("df_gate") or {}


def _gated_terms(out):
    return {e["term"] for e in _gate(out).get("gated", ())}


def _exempt(out):
    return {e["term"]: e["reason"] for e in _gate(out).get("exempt", ())}


def _rescue_info(out):
    return (out.stats.get("coverage_lexical") or {}).get("rescue") or {}


def _rowids(conn):
    return {str(u): int(r) for u, r in
            conn.execute("SELECT unit_id, rowid FROM units")}


# ---------------------------------------------------------------------------
# K37 — lane-miss forensic: exactly one class per question, f-count = 0
# ---------------------------------------------------------------------------


def _probe_evidence(conn, out, qv, gold_uids, *, eligible=None,
                    lane_present=True, probe_error=None, in_items=()):
    """Assemble the classifier's evidence mapping from a REAL lane run —
    real ``out.stats``, real unit rows, and postings re-measured with the
    lane's own MATCH primitives (exactly what ``_LaneProbe.postings``
    replays)."""
    qterms, idents = lex._query_terms(qv)
    nominating = sorted(
        {q.term for q in idents}
        | ({q.term for q in qterms if q.content}
           if any(q.content for q in qterms)
           else {q.term for q in qterms}))
    posts = {}
    for t in {q.term for q in qterms}:
        ex = lex._match_rowids(conn, "unit_fts", lex._fts_quote(t))
        st = lex._match_rowids(conn, "unit_fts_stem", lex._fts_quote(t))
        posts[t] = (ex or set()) | (st or set())
    for t in {q.term for q in idents}:
        posts[t] = lex._match_rowids(conn, "unit_fts",
                                     lex._fts_quote(t)) or set()
    rid_of = _rowids(conn)
    admitted = {c.unit_id for c in out.candidates}
    rescued = {rid_of[c.unit_id] for c in out.candidates
               if c.signals.get("rescue") and c.unit_id in rid_of}
    items = set(in_items)
    units = []
    for uid in gold_uids:
        rid = rid_of.get(str(uid))
        if rid is None:
            continue
        units.append({
            "ref": str(uid), "unit_id": str(uid), "rowid": rid,
            "generation": GEN,
            "eligible": (True if eligible is None
                         else str(uid) in eligible),
            "matching": sorted(t for t in nominating
                               if rid in posts.get(t, ())),
            "rescued": rid in rescued,
            "readable": True,
            "admitted": str(uid) in admitted,
            "in_items": str(uid) in items,
        })
    return {
        "lane_present": lane_present,
        "explain_present": True,
        "lane_status": out.status.value,
        "lane_reason": out.reason,
        "stats": out.stats,
        "terms": sorted(posts),
        "nominating": nominating,
        "post_pool": None,
        "gold_units": units,
        "recheck_dropped": None,
        "probe_error": probe_error,
    }


def _forensic_fixture(conn):
    """Docs so each first-loss class is reachable by construction:
    floodterm floods (gate), budgeted terms drop, a capped gold sits
    below the admitted cut, an unrelated doc shares no nominating
    term."""
    fx.seed_unit(conn, "f1", text="floodterm shared words")
    fx.seed_unit(conn, "f2", text="floodterm shared words")
    fx.seed_unit(conn, "f3", text="floodterm shared words")
    fx.seed_unit(conn, "floodgold", text="floodterm unique gold words")
    fx.seed_unit(conn, "keepdoc", text="keepterm note")
    fx.seed_unit(conn, "capgold", text="capterm once here")
    fx.seed_unit(conn, "capdoc", text="capterm capterm capterm many")
    fx.seed_unit(conn, "droppedgold", text="dropt only here")
    fx.seed_unit(conn, "droppedextra", text="dropt and more")
    fx.seed_unit(conn, "nomatch", text="utterly unrelated content")


def test_k37_one_class_per_miss_and_zero_eligibility():
    """K37 / V8-07.01: the forensic assigns exactly one first-loss class
    to every answerable lane_miss evidence record, and on a healthy run
    the eligibility class count is 0 — while a genuinely denied gold is
    provably classified ``f`` (the defect detector works)."""
    conn = fx._db()
    _forensic_fixture(conn)
    seed_df(conn, "floodterm", 1500)

    # class a — gold's only matching nominating term was df-gated and
    # the bounded rescue (row-capped at 2) never reached it.
    qv_a = fx.mkqv("floodterm keepterm")
    out_a = lane_lexical(
        mkctx8(conn, manifest={"lexical.df_floor": 20,
                               "lexical.rescue_rows": 2}),
        qv_a, fx.mkslice())
    assert out_a.status == LaneStatus.OK
    assert _gated_terms(out_a) == {"floodterm"}
    assert _rescue_info(out_a)["fired"] is True
    ev_a = _probe_evidence(conn, out_a, qv_a, ["floodgold"])
    letter_a, detail_a = classify_lane_miss(ev_a)
    assert letter_a == "a", detail_a
    assert detail_a["note"] == "all_matching_gated"

    # class b — nomination budget dropped the gold's only term
    # (rescue disarmed by the K_rescue=0 arm).
    qv_b = fx.mkqv("keepterm dropt")
    out_b = lane_lexical(
        mkctx8(conn, nominate_terms_max=1,
               manifest={"lexical.K_rescue": 0}),
        qv_b, fx.mkslice())
    dropped = {e["term"]: e["reason"]
               for e in out_b.stats.get("nomination_dropped", ())}
    assert dropped.get("dropt") == "budget"
    ev_b = _probe_evidence(conn, out_b, qv_b, ["droppedgold"])
    letter_b, detail_b = classify_lane_miss(ev_b)
    assert letter_b == "b", detail_b
    assert detail_b["note"] == "all_matching_dropped"

    # class c — gold scored but the lane cap admitted only the top doc.
    qv_c = fx.mkqv("capterm")
    out_c = lane_lexical(mkctx8(conn), qv_c, fx.mkslice(cap=1))
    assert out_c.stats["overflow"] == 1
    ev_c = _probe_evidence(conn, out_c, qv_c, ["capgold"])
    letter_c, detail_c = classify_lane_miss(ev_c)
    assert letter_c == "c", detail_c
    assert detail_c["note"] == "cap_cut"

    # class d — zero nominating overlap with the gold.
    ev_d = _probe_evidence(conn, out_a, qv_a, ["nomatch"])
    letter_d, detail_d = classify_lane_miss(ev_d)
    assert letter_d == "d", detail_d
    assert detail_d["note"] == "no_nominating_overlap"

    # class e — the slice expired at lane entry.
    out_e = lane_lexical(mkctx8(conn), qv_a, fx.mkslice(deadline_ms=0.0))
    assert out_e.status == LaneStatus.DEADLINE
    ev_e = _probe_evidence(conn, out_e, qv_a, ["floodgold"])
    letter_e, detail_e = classify_lane_miss(ev_e)
    assert letter_e == "e", detail_e
    assert detail_e["deadline_phase"] == "entry"

    # class g — probe unavailable / no gold turn unit.
    ev_g1 = _probe_evidence(conn, out_a, qv_a, ["floodgold"],
                            probe_error="read_snapshot:boom")
    assert classify_lane_miss(ev_g1)[0] == "g"
    ev_g2 = _probe_evidence(conn, out_a, qv_a, ["not_a_unit"])
    assert classify_lane_miss(ev_g2)[0] == "g"

    # exactly one class, always a member of the declared order
    letters = [letter_a, letter_b, letter_c, letter_d, letter_e]
    assert all(isinstance(l, str) and l in CLASS_ORDER for l in letters)
    assert set(CLASS_LABELS) == set(CLASS_ORDER)

    # the healthy-run eligibility class count is 0 …
    healthy = [letter_a, letter_b, letter_c, letter_d, letter_e]
    assert sum(1 for l in healthy if l == "f") == 0

    # … and a real eligibility denial is classified f — the lane's own
    # fail-closed path produces the evidence honestly.
    elig = {"f1", "f2", "f3", "keepdoc"}   # floodgold held out
    out_f = lane_lexical(
        mkctx8(conn, eligible=elig,
               manifest={"lexical.df_floor": 20}),
        qv_a, fx.mkslice())
    ev_f = _probe_evidence(conn, out_f, qv_a, ["floodgold"],
                           eligible=elig)
    letter_f, detail_f = classify_lane_miss(ev_f)
    assert letter_f == "f", detail_f
    assert detail_f["note"] == "unit_denied"

    # lane-level shape defect: an unrecognized eligibility shape.
    out_f2 = lane_lexical(mkctx8(conn, eligible=42), qv_a, fx.mkslice())
    assert out_f2.status == LaneStatus.UNAVAILABLE
    ev_f2 = _probe_evidence(conn, out_f2, qv_a, ["floodgold"])
    letter_f2, detail_f2 = classify_lane_miss(ev_f2)
    assert letter_f2 == "f", detail_f2
    assert detail_f2["note"] == "eligibility_shape_unknown"


def test_k37_multi_unit_deepest_survivor_decides():
    """K37: with several gold units the question dies where its last
    live candidate died — the deepest-surviving unit's stage wins."""
    conn = fx._db()
    _forensic_fixture(conn)
    seed_df(conn, "floodterm", 1500)
    qv = fx.mkqv("floodterm keepterm")
    out = lane_lexical(
        mkctx8(conn, manifest={"lexical.df_floor": 20,
                               "lexical.rescue_rows": 2}),
        qv, fx.mkslice())
    elig = {"f1", "f2", "f3", "keepdoc"}   # f-denied + a-gated golds
    ev = _probe_evidence(conn, out, qv, ["floodgold", "capgold"],
                         eligible=elig)
    # capgold was denied by the fixture's eligible set too? no — it is
    # absent from the set, so both units are f-denied → class f.
    letter, detail = classify_lane_miss(ev)
    assert letter == "f"
    assert detail["note"] == "unit_denied"

    # denied (depth 1) + cap-cut (depth 7) → the deeper c stage decides
    elig2 = {"f1", "f2", "f3", "keepdoc", "capgold", "capdoc"}
    out_c = lane_lexical(mkctx8(conn, eligible=elig2), fx.mkqv("capterm"),
                         fx.mkslice(cap=1))
    ev2 = _probe_evidence(conn, out_c, fx.mkqv("capterm"),
                          ["capgold"], eligible=elig2)
    # combine a denied gold (from run 1's gate) with the cap-cut gold —
    # hand-join the unit lists exactly as _assemble_evidence would for
    # one question with two gold refs.
    ev_join = _probe_evidence(conn, out_c, fx.mkqv("capterm"),
                            ["capgold"], eligible=elig2)
    denied_unit = dict(ev["gold_units"][0])
    ev_join["gold_units"] = [denied_unit] + ev_join["gold_units"]
    letter2, detail2 = classify_lane_miss(ev_join)
    assert letter2 == "c", detail2
    assert detail2["note"] == "cap_cut"
    stages = {u["unit_id"]: u["class"] for u in detail2["unit_stages"]}
    assert stages["floodgold"] == "f" and stages["capgold"] == "c"


# ---------------------------------------------------------------------------
# K38 — df-gate exemptions (V8-07.02)
# ---------------------------------------------------------------------------


def test_k38_speaker_canon_df1500_not_gated():
    """K38a: a speaker canon whose maintained df is 1,500 is a protected
    nominating surface — exempt from the pre-fetch gate."""
    conn = fx._db()
    fx.seed_unit(conn, "u1", text="caroline review notes")
    fx.seed_unit(conn, "u2", text="review board minutes")
    seed_df(conn, "caroline", 1500)

    qv = dataclasses.replace(fx.mkqv("caroline review"),
                             speaker_canon="Caroline")
    out = lane_lexical(
        mkctx8(conn, manifest={"lexical.df_floor": 20}), qv, fx.mkslice())
    assert out.status == LaneStatus.OK
    gate = _gate(out)
    assert gate["floor"] == 20
    assert _gated_terms(out) == set()
    assert _exempt(out) == {"caroline": "speaker_canon"}
    # the canon term really nominated — its docs are candidates
    assert "u1" in {c.unit_id for c in out.candidates}


def test_k38_entity_canon_exempt():
    """K38b: every folded token of a resolved entity canon is exempt."""
    conn = fx._db()
    fx.seed_unit(conn, "u1", text="alice chen reviewed the proposal")
    fx.seed_unit(conn, "u2", text="the proposal review")
    seed_df(conn, "alice", 1500)
    seed_df(conn, "chen", 1600)

    qv = dataclasses.replace(fx.mkqv("alice chen review"),
                             entity_canons=("Alice Chen",))
    out = lane_lexical(
        mkctx8(conn, manifest={"lexical.df_floor": 20}), qv, fx.mkslice())
    exempt = _exempt(out)
    assert exempt["alice"] == "entity_canon"
    assert exempt["chen"] == "entity_canon"
    assert _gated_terms(out) == set()


def test_k38_identifier_exempt():
    """K38c: identifier-channel terms nominate by identity — never
    gated even when their maintained df exceeds the cut."""
    conn = fx._db()
    fx.seed_unit(conn, "u1", text="ticket jira-7 closed today")
    fx.seed_unit(conn, "u2", text="unrelated chatter")
    seed_df(conn, "jira-7", 1500)

    qv = fx.mkqv("what about jira-7", channels={"jira-7": "identifier"})
    out = lane_lexical(
        mkctx8(conn, manifest={"lexical.df_floor": 20}), qv, fx.mkslice())
    assert _exempt(out)["jira-7"] == "identifier"
    assert "jira-7" not in _gated_terms(out)


def test_k38_rarest_content_term_never_gated():
    """K38d: the rarest content term is the query's strongest nominating
    signal — it stays ungated while commoner terms gate."""
    conn = fx._db()
    fx.seed_unit(conn, "u1", text="floodx floody both here")
    fx.seed_unit(conn, "u2", text="floody alone here")
    seed_df(conn, "floodx", 1500)
    seed_df(conn, "floody", 1400)

    out = lane_lexical(
        mkctx8(conn, manifest={"lexical.df_floor": 20}),
        fx.mkqv("floodx floody"), fx.mkslice())
    assert _exempt(out)["floody"] == "rarest_content_term"
    assert _gated_terms(out) == {"floodx"}
    # floody's docs still nominate; floodx's gate consumed no budget
    assert {c.unit_id for c in out.candidates} == {"u1", "u2"}


def test_k38_only_high_df_terms_keep_a_nominator():
    """K38e: a query whose every term exceeds the cut still keeps ≥ 1
    nominating term — the last-nominating-term exemption fires."""
    conn = fx._db()
    fx.seed_unit(conn, "u1", text="the best of times")
    fx.seed_unit(conn, "u2", text="of mice and the rest")
    seed_df(conn, "the", 1500)
    seed_df(conn, "of", 1400)

    # "the"/"of" are both non-content terms (analyzer stop set) — the
    # rarest-content exemption cannot apply, so (d) must.
    out = lane_lexical(
        mkctx8(conn, manifest={"lexical.df_floor": 20}),
        fx.mkqv("the of"), fx.mkslice())
    exempt = _exempt(out)
    assert exempt.get("of") == "last_nominating_term", exempt
    assert _gated_terms(out) == {"the"}
    # the exempt term really nominated — candidates exist
    assert out.candidates
    assert out.stats["nominated_terms"] == 1
    dropped = {e["term"]: e["reason"]
               for e in out.stats["nomination_dropped"]}
    assert dropped["the"] == "df_threshold"


def test_k38_gated_terms_keep_honest_df():
    """K38: gated terms report their maintained df, not a fabricated 0 —
    ``stats.df`` and ``df_prefetch_gated`` carry the real count."""
    conn = fx._db()
    fx.seed_unit(conn, "u1", text="floodterm keepterm together")
    fx.seed_unit(conn, "u2", text="keepterm alone")
    seed_df(conn, "floodterm", 1500)

    out = lane_lexical(
        mkctx8(conn, manifest={"lexical.df_floor": 20}),
        fx.mkqv("floodterm keepterm"), fx.mkslice())
    assert _gated_terms(out) == {"floodterm"}
    assert out.stats["df"]["floodterm"] == 1500        # not 0
    assert out.stats["df_prefetch_gated"] == [
        {"term": "floodterm", "df": 1500}]


def test_k38_df_floor_off_disarms_gate():
    """K38 arm surface: ``lexical.df_floor = off`` disarms the gate —
    nothing gates, the report says so honestly."""
    conn = fx._db()
    fx.seed_unit(conn, "u1", text="floodterm here")
    seed_df(conn, "floodterm", 1500)
    out = lane_lexical(
        mkctx8(conn, manifest={"lexical.df_floor": "off"}),
        fx.mkqv("floodterm"), fx.mkslice())
    gate = _gate(out)
    assert gate["floor"] == "off"
    assert gate["gated"] == [] and gate["exempt"] == []
    assert {c.unit_id for c in out.candidates} == {"u1"}


# ---------------------------------------------------------------------------
# K39 — bounded exhaustive rescue (V8-07.03, §21.4)
# ---------------------------------------------------------------------------


def test_k39_rescue_fires_on_starved_nomination():
    """K39: when nomination yields fewer than K_rescue eligible
    candidates the lane runs the bounded rescue — the ≤2 rarest dropped
    terms are scanned, produced rows carry ``signals.rescue``, and the
    coverage report is complete."""
    conn = fx._db()
    fx.seed_unit(conn, "a1", text="alpha common words")
    fx.seed_unit(conn, "a2", text="alpha other words")
    fx.seed_unit(conn, "b1", text="beta note")

    out = lane_lexical(
        mkctx8(conn, nominate_terms_max=1),   # drops "alpha" (df 2)
        fx.mkqv("alpha beta"), fx.mkslice())
    assert out.status == LaneStatus.OK
    dropped = {e["term"]: e["reason"]
               for e in out.stats["nomination_dropped"]}
    assert dropped == {"alpha": "budget"}
    rescue = _rescue_info(out)
    assert rescue["fired"] is True
    assert rescue["terms"] == ["alpha"]
    assert rescue["rows"] >= 2
    assert rescue["produced"] == 2
    by_id = {c.unit_id: c for c in out.candidates}
    assert by_id["a1"].signals["rescue"] is True
    assert by_id["a2"].signals["rescue"] is True
    assert "rescue" not in by_id["b1"].signals


def test_k39_rescue_row_budget():
    """K39: the ``rescue_rows`` arm bounds the scan — one row visited,
    one doc produced, and the report admits the cut."""
    conn = fx._db()
    fx.seed_unit(conn, "a1", text="alpha common words")
    fx.seed_unit(conn, "a2", text="alpha other words")
    fx.seed_unit(conn, "b1", text="beta note")

    out = lane_lexical(
        mkctx8(conn, nominate_terms_max=1,
               manifest={"lexical.rescue_rows": 1}),
        fx.mkqv("alpha beta"), fx.mkslice())
    rescue = _rescue_info(out)
    assert rescue["fired"] is True
    assert rescue["rows"] <= 1
    assert rescue["produced"] <= 1
    produced = [c.unit_id for c in out.candidates
                if c.signals.get("rescue")]
    assert len(produced) == rescue["produced"]


def test_k39_rescue_never_produces_ineligible():
    """K39 / §21.4: rescue rows are fenced by the request's eligible set
    — an ineligible doc matching a gated term is never produced."""
    conn = fx._db()
    fx.seed_unit(conn, "held", text="gatedterm unique words")
    fx.seed_unit(conn, "free", text="gatedterm other words")
    fx.seed_unit(conn, "keep", text="keepterm note")
    seed_df(conn, "gatedterm", 1500)

    elig = {"free", "keep"}                # "held" is ineligible
    out = lane_lexical(
        mkctx8(conn, eligible=elig,
               manifest={"lexical.df_floor": 20}),
        fx.mkqv("gatedterm keepterm"), fx.mkslice())
    assert _gated_terms(out) == {"gatedterm"}
    rescue = _rescue_info(out)
    assert rescue["fired"] is True
    assert rescue["produced"] == 1         # only the eligible match
    ids = {c.unit_id for c in out.candidates}
    assert "held" not in ids
    assert {c.unit_id for c in out.candidates
            if c.signals.get("rescue")} == {"free"}


def _rescue_call(conn, terms_text, *, dropped=(), gated=(), k_rescue=64,
                 rescue_rows=4096, deadline=None, nominated=None,
                 eligible=None):
    """Drive the real ``_rescue_candidates`` with real lane plumbing —
    real ``_query_terms``/``_universe``/postings; only the incoming
    stats gate report and deadline are set up for the scenario."""
    qv = fx.mkqv(terms_text)
    qterms, idents = lex._query_terms(qv)
    uni, _trunc = lex._universe(conn, SCOPE, GEN, lex._Deadline(60000))
    rid_of = _rowids(conn)
    elig_ids = (set(rid_of) if eligible is None else eligible)
    elig_rids = {rid_of[u] for u in elig_ids if u in rid_of}
    posts = {}
    for qt in qterms:
        ex = lex._match_rowids(conn, "unit_fts", lex._fts_quote(qt.term))
        st = lex._match_rowids(conn, "unit_fts_stem",
                               lex._fts_quote(qt.term))
        posts[qt.term] = ((ex or set()) | (st or set())) \
            & set(uni.by_rowid) & elig_rids
    stats = {"coverage_lexical": {"df_gate": {
        "floor": 20,
        "gated": [{"term": t, "df": d} for t, d in gated],
        "exempt": []}}}
    stats["nomination_dropped"] = [
        {"term": t, "df": d, "reason": "budget"} for t, d in dropped]
    if nominated is None:
        nominated = set()
    produced = lex._rescue_candidates(
        conn, qterms, posts, nominated, uni, elig_rids, True,
        deadline if deadline is not None else lex._Deadline(60000),
        stats, k_rescue=k_rescue, rescue_rows=rescue_rows)
    return produced, stats["coverage_lexical"]["rescue"], rid_of


def test_k39_rescue_all_gated_trigger():
    """K39: the every-content-term-gated trigger fires even when the
    nominated pool is already ≥ K_rescue — a coverage path the gate's
    rarest-term exemption normally makes unreachable, driven here by
    seeding the lane's own df_gate report."""
    conn = fx._db()
    fx.seed_unit(conn, "g1", text="gateone words")
    fx.seed_unit(conn, "g2", text="gatetwo words")
    rid_of = _rowids(conn)
    produced, info, _ = _rescue_call(
        conn, "gateone gatetwo",
        gated=(("gateone", 1500), ("gatetwo", 1400)),
        nominated=set(rid_of.values()),      # already ≥ K_rescue
        k_rescue=64)
    assert info["fired"] is True
    assert info["terms"] == ["gatetwo", "gateone"]   # df-ascending, ≤2
    assert info["rows"] > 0
    assert produced == set()                  # all already nominated


def test_k39_rescue_deadline_at_entry():
    """K39: an already-expired slice produces zero visited rows —
    fired/terms are still reported honestly."""
    conn = fx._db()
    fx.seed_unit(conn, "a1", text="alpha common words")
    fx.seed_unit(conn, "a2", text="alpha other words")
    produced, info, _ = _rescue_call(
        conn, "alpha beta", dropped=(("alpha", 2),),
        deadline=lex._Deadline(0.0))
    assert info["fired"] is True
    assert info["terms"] == ["alpha"]
    assert info["rows"] == 0 and info["produced"] == 0
    assert produced == set()


def test_k39_rescue_deadline_mid_scan(monkeypatch):
    """K39 / §21.4: the deadline is checked every ``_RESCUE_CHECK`` rows;
    a scripted expiry stops the scan at the cadence boundary with honest
    row counts."""
    conn = fx._db()
    for i in range(6):
        fx.seed_unit(conn, f"a{i}", text="alpha common words")
    fx.seed_unit(conn, "b1", text="beta note")

    calls = {"n": 0}

    class ScriptedDeadline:
        def expired(self):
            calls["n"] += 1
            return calls["n"] > 3

    monkeypatch.setattr(lex, "_RESCUE_CHECK", 1)   # check every row
    produced, info, rid_of = _rescue_call(
        conn, "alpha beta", dropped=(("alpha", 6),),
        deadline=ScriptedDeadline())
    assert info["fired"] is True
    assert info["rows"] == 3                     # cut at the third row
    assert info["produced"] == len(produced)
    # produced rows are the ascending-rowid prefix the scan visited
    seen = sorted(rid_of[f"a{i}"] for i in range(6))
    assert produced <= set(seen[:3])


def test_k39_rescue_scans_at_most_two_terms():
    """K39: at most two of the rarest gated/dropped terms are scanned —
    the rest are reported untouched."""
    conn = fx._db()
    for i in range(3):
        fx.seed_unit(conn, f"x{i}", text="xterm words")
        fx.seed_unit(conn, f"y{i}", text="yterm words")
        fx.seed_unit(conn, f"z{i}", text="zterm words")
    produced, info, rid_of = _rescue_call(
        conn, "xterm yterm zterm",
        dropped=(("zterm", 3), ("xterm", 3), ("yterm", 3)),
        k_rescue=64)
    assert info["fired"] is True
    assert len(info["terms"]) == 2
    # df ties break on term text → xterm, yterm scanned; zterm untouched
    assert info["terms"] == ["xterm", "yterm"]


# ---------------------------------------------------------------------------
# K40 — maintained unit_doclen (V8-07.04) + single_match (V8-07.06)
# ---------------------------------------------------------------------------


_DOCSIZE_STMT = "unit_fts_docsize"


def _docsize_counting_trace(conn):
    counts = {"docsize": 0}

    def cb(sql):
        if _DOCSIZE_STMT in sql:
            counts["docsize"] += 1

    conn.set_trace_callback(cb)
    return counts


def _k40_corpus(conn):
    docs = {
        "u1": {"text": "the quick brown fox jumps", "speaker": "alice"},
        "u2": {"text": "quick dogs and quick cats", "speaker": "bob"},
        "u3": {"text": "a brown matter of record", "entities": "fox"},
        "u4": {"text": "nothing matching here", "session": "sync"},
        "u5": {"text": "quick brown quick brown quick"},
    }
    fields = {}
    for u, f in docs.items():
        fx.seed_unit(conn, u, **f)
        fields[u] = {"text": f.get("text", ""),
                     "speaker": f.get("speaker", ""),
                     "entities": f.get("entities", ""),
                     "session": f.get("session", ""),
                     "when": f.get("when", "")}
    return docs, fields


def test_k40_doclen_zero_docsize_statements_byte_equal_scores():
    """K40 / V8-07.04: once ``unit_doclen`` is populated the scorer reads
    field lengths from it — zero ``unit_fts_docsize`` statements per
    query — and every BM25F score is byte-equal to the pre-change run.
    An ineligible doc forces the corrected-stats path so both lens
    consumers (corpus correction + candidate scoring) are exercised."""
    conn = fx._db()
    _docs, fields = _k40_corpus(conn)
    elig = {"u1", "u2", "u3", "u4"}            # u5 held out → corrected
    qv = fx.mkqv("quick brown fox")

    counts = _docsize_counting_trace(conn)
    out_pre = lane_lexical(mkctx8(conn, eligible=elig), qv, fx.mkslice())
    conn.set_trace_callback(None)
    assert counts["docsize"] > 0              # pre-change: docsize reads
    assert out_pre.stats["stats"] == "corrected"

    # the projection transaction writes maintained lengths
    conn.execute(_DOCLEN_DDL)
    for uid, f in fields.items():
        write_doclen(conn, uid, f)

    counts2 = _docsize_counting_trace(conn)
    out_post = lane_lexical(mkctx8(conn, eligible=elig), qv, fx.mkslice())
    conn.set_trace_callback(None)
    assert counts2["docsize"] == 0            # V8-07.04: zero statements
    assert out_post.stats["stats"] == "corrected"

    # byte-equal scores — exact float equality, not approximate
    assert [(c.unit_id, c.raw_score) for c in out_pre.candidates] == \
        [(c.unit_id, c.raw_score) for c in out_post.candidates]


def test_k40_doclen_generation_fenced():
    """K40 / V8-19.04: ``unit_doclen`` reads honor the generation fence —
    the pinned run sees the latest row at/below its generation and
    future-generation lengths are invisible."""
    conn = fx._db()
    fx.seed_unit(conn, "u1", text="one two three four five")
    fx.seed_unit(conn, "u2", text="one")
    conn.execute(_DOCLEN_DDL)
    write_doclen(conn, "u1", {"text": "one two three four five"})
    write_doclen(conn, "u2", {"text": "one"})
    # a future-generation doclen claims a different length — invisible
    # to a generation-1 pin.
    write_doclen(conn, "u1", {"text": "one two three four five"},
                 gen=2, overrides={"text": 1})
    write_doclen(conn, "u2", {"text": "one"}, gen=2, overrides={"text": 1})

    qv = fx.mkqv("one")
    g1 = lane_lexical(mkctx8(conn, gen=1), qv, fx.mkslice())
    g2 = lane_lexical(mkctx8(conn, gen=2), qv, fx.mkslice())
    s1 = {c.unit_id: c.raw_score for c in g1.candidates}
    s2 = {c.unit_id: c.raw_score for c in g2.candidates}
    assert set(s1) == set(s2) == {"u1", "u2"}
    assert s1["u1"] != s2["u1"]      # len 5 vs gen-2 len 1 → different norm
    # the gen-1 run's score equals the hand-computed len-5 normalization
    # (both docs contain "one" → df = 2, avglen = (5 + 1)/2 = 3)
    import math as _m
    avg = 3.0
    idf = _m.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    tf_t = 1.0 / (1 - 0.75 + 0.75 * 5 / avg)
    want = idf * tf_t * 2.2 / (tf_t + 1.2)
    assert s1["u1"] == pytest.approx(want, abs=1e-9)


def _collect_for(conn, query, *, single_match, manifest=None):
    """``(posts, nominated, stats)`` from the real ``_collect_postings``
    with the ``single_match`` arm flipped — same plumbing the lane uses."""
    qv = fx.mkqv(query)
    qterms, idents = lex._query_terms(qv)
    uni, _ = lex._universe(conn, SCOPE, GEN, lex._Deadline(60000))
    elig_rids = set(uni.by_rowid)
    stats = {}
    res = lex._collect_postings(
        conn, qterms, idents, uni, elig_rids, True,
        lex._Deadline(60000), stats,
        nominate_budget=32, df_theta=None, n_eligible=len(elig_rids),
        df_maintained=lex._maintained_df(
            conn, SCOPE, GEN, [q.term for q in (*qterms, *idents)]),
        df_floor=None, single_match=single_match,
        canon_reasons=lex._canon_gate_reasons(qv))
    assert res is not None
    posts, nominated = res
    return posts, nominated, stats


def test_k40_single_match_vocab_path_byte_identical():
    """K40 / V8-07.06: with instance-mode ``fts5vocab`` postings
    maintained for both channels (the only mode whose ``doc`` is a
    rowid), the collapsed-nomination arm produces term→rowid sets
    identical to the per-term MATCH loop, in one statement per
    channel."""
    conn = fx._db()
    docs = {"u1": "pursue the matter today",
            "u2": "pursued it yesterday",
            "u3": "matter settled elsewhere"}
    for u, t in docs.items():
        fx.seed_unit(conn, u, text=t)
    conn.execute("DROP TABLE unit_fts_vocab")
    conn.execute("CREATE VIRTUAL TABLE unit_fts_vocab"
                 " USING fts5vocab('unit_fts','instance')")
    conn.execute("CREATE VIRTUAL TABLE unit_fts_stem_vocab"
                 " USING fts5vocab('unit_fts_stem','instance')")

    posts_a, nom_a, stats_a = _collect_for(conn, "pursue matter",
                                         single_match=False)
    posts_b, nom_b, stats_b = _collect_for(conn, "pursue matter",
                                         single_match=True)
    assert stats_b["single_match"] == "vocab"
    assert posts_b == posts_a                # byte-identical sets
    assert nom_b == nom_a

    out_off = lane_lexical(mkctx8(conn), fx.mkqv("pursue matter"),
                           fx.mkslice())
    out_on = lane_lexical(
        mkctx8(conn, manifest={"lexical.single_match": True}),
        fx.mkqv("pursue matter"), fx.mkslice())
    assert out_on.stats["single_match"] == "vocab"
    assert [(c.unit_id, c.raw_score) for c in out_off.candidates] == \
        [(c.unit_id, c.raw_score) for c in out_on.candidates]


def test_k40_single_match_fields_path_byte_identical():
    """K40 / V8-07.06: with no usable vocab the arm falls to one union
    MATCH per channel plus field-byte attribution — identical sets."""
    conn = fx._db()
    docs = {"u1": "pursue the matter today",
            "u2": "pursued it yesterday",
            "u3": "matter settled elsewhere"}
    for u, t in docs.items():
        fx.seed_unit(conn, u, text=t)
    conn.execute("DROP TABLE unit_fts_vocab")

    posts_a, nom_a, _ = _collect_for(conn, "pursue matter",
                                     single_match=False)
    posts_b, nom_b, stats_b = _collect_for(conn, "pursue matter",
                                         single_match=True)
    assert stats_b["single_match"] == "fields"
    assert posts_b == posts_a
    assert nom_b == nom_a


def test_k40_single_match_count_mode_vocab_degrades_honestly():
    """K40: an fts5vocab table in ``col`` or ``row`` mode is NOT usable
    for the collapsed path — their ``doc`` column is a document count,
    not a rowid; the arm must degrade to the union/field path instead
    of fabricating posting rowids."""
    docs = {"u1": "pursue the matter today",
            "u2": "pursued it yesterday",
            "u3": "matter settled elsewhere"}

    # 'col' mode — the sibling fixture's own shape
    conn = fx._db()
    for u, t in docs.items():
        fx.seed_unit(conn, u, text=t)
    posts_a, nom_a, _ = _collect_for(conn, "pursue matter",
                                     single_match=False)
    posts_b, nom_b, stats_b = _collect_for(conn, "pursue matter",
                                         single_match=True)
    assert stats_b["single_match"] == "fields"
    assert posts_b == posts_a
    assert nom_b == nom_a

    # 'row' mode — same count-shaped doc column
    conn = fx._db()
    for u, t in docs.items():
        fx.seed_unit(conn, u, text=t)
    conn.execute("DROP TABLE unit_fts_vocab")
    conn.execute("CREATE VIRTUAL TABLE unit_fts_vocab"
                 " USING fts5vocab('unit_fts','row')")
    posts_b, nom_b, stats_b = _collect_for(conn, "pursue matter",
                                         single_match=True)
    assert stats_b["single_match"] == "fields"
    assert posts_b == posts_a
    assert nom_b == nom_a


# ---------------------------------------------------------------------------
# K41 — bounded stem memoization (V8-07.05)
# ---------------------------------------------------------------------------


def _reset_stem_caches():
    lex._STEM_LRU.clear()
    lex._SC_MEMO.clear()
    lex._configure_stem_lru(lex._STEM_LRU_DEFAULT)


def test_k41_warm_process_stem_budget(monkeypatch):
    """K41 / V8-07.05: on a warm process a query computes ≤ 50 real
    ``porter_stem`` calls — the process LRU serves every repeat — while
    a cold corpus with > 50 distinct tokens computes more than 50."""
    conn = fx._db()
    for i in range(60):
        fx.seed_unit(conn, f"w{i}", text=f"token{i} runs and runs")
    _reset_stem_caches()

    orig = lex.porter_stem
    calls = {"n": 0}

    def counting(tok):
        calls["n"] += 1
        return orig(tok)

    monkeypatch.setattr(lex, "porter_stem", counting)

    qv = fx.mkqv("runs")
    out_cold = lane_lexical(mkctx8(conn), qv, fx.mkslice(cap=200))
    cold_n = calls["n"]
    assert cold_n > 50                      # 60 distinct tokens + query
    assert out_cold.status == LaneStatus.OK
    assert len(out_cold.candidates) == 60   # 'runs' stems every doc

    # warm process: a fresh ctx re-runs the identical query and computes
    # no more than the declared 50-stem budget — the LRU serves all.
    calls["n"] = 0
    out_warm = lane_lexical(mkctx8(conn), qv, fx.mkslice(cap=200))
    assert calls["n"] <= 50, calls["n"]
    assert [(c.unit_id, c.raw_score) for c in out_cold.candidates] == \
        [(c.unit_id, c.raw_score) for c in out_warm.candidates]


def test_k41_memoized_equals_uncached_every_token():
    """K41: ``_porter_memo`` returns the pure ``porter_stem`` value for
    every fixture token, and the disarmed arm (``stem_lru=0``) produces
    identical candidates — memoization never changes a score."""
    conn = fx._db()
    fx.seed_unit(conn, "u1", text="running races are running")
    fx.seed_unit(conn, "u2", text="the runner ran")
    _reset_stem_caches()

    toks = set(lex._tokenize("running races are running")
               + lex._tokenize("the runner ran")
               + ["runs", "runner", "races"])
    for t in toks:
        assert lex._porter_memo(t) == lex.porter_stem(t)

    qv = fx.mkqv("runs races")
    out_cached = lane_lexical(mkctx8(conn), qv, fx.mkslice())
    out_off = lane_lexical(
        mkctx8(conn, manifest={"lexical.stem_lru": 0}), qv, fx.mkslice())
    assert [(c.unit_id, c.raw_score) for c in out_cached.candidates] == \
        [(c.unit_id, c.raw_score) for c in out_off.candidates]
    lex._configure_stem_lru(lex._STEM_LRU_DEFAULT)


def test_k41_lru_bound_is_real():
    """K41: the ``stem_lru`` arm bounds the process cache — a cap of 3
    holds at most 3 entries while the lane still computes correctly."""
    conn = fx._db()
    for i in range(8):
        fx.seed_unit(conn, f"w{i}", text=f"tok{i} walking")
    _reset_stem_caches()
    out = lane_lexical(
        mkctx8(conn, manifest={"lexical.stem_lru": 3}),
        fx.mkqv("walking"), fx.mkslice())
    assert out.status == LaneStatus.OK
    assert len(lex._STEM_LRU) <= 3
    lex._configure_stem_lru(lex._STEM_LRU_DEFAULT)


# ---------------------------------------------------------------------------
# K42 — prf_round removal (V8-07.07)
# ---------------------------------------------------------------------------


def test_k42_prf_round_removed_from_poolprofile():
    """K42 / V8-07.07: ``prf_round`` no longer exists in ``PoolProfile``
    or ``POOLS`` — no field, no values, no references. The static
    flag-reader check additionally sweeps every production source for a
    reader of the dead flag (D8-25: declared-but-unexecuted)."""
    from verbatim.core import types_v7
    fields = {f.name for f in dataclasses.fields(types_v7.PoolProfile)}
    assert "prf_round" not in fields
    for pool in types_v7.POOLS.values():
        assert not hasattr(pool, "prf_round")
    assert "prf_round" not in inspect.getsource(types_v7)
    # Static flag-reader check: no production module may name the flag.
    root = Path(types_v7.__file__).resolve().parents[1]
    readers = [
        str(p)
        for p in root.rglob("*.py")
        if "prf_round" in p.read_text(encoding="utf-8")
    ]
    assert readers == []
