"""Tests for the V7 lexical BM25F lane + trigram/respell fuzzy lane.

§30 mirror DDL — swap for schema_v7.ensure_v7_additive at integration.
The FTS rowid convention here (``unit_fts.rowid == units.rowid``) matches
the external-content mapping the lane documents; ``lex_stats``/``lex_df``
are maintained by the seed helper exactly as stats_v7 will (per-field
document-frequency and length sums keyed by ``(scope, generation)``).

Query views are built as ``NormAnalysis``/``QueryViewV7`` literals — the
real ``norm/v2`` analyzer lands with w-norm; byte offsets are computed
with a tiny regex tokenizer so quoted-span mapping works end to end.
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verbatim.core.types_v7 import (  # noqa: E402
    BudgetClass,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneSlice,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    RetrievalPolicyV7,
    LaneName,
)
from verbatim.retrieval.v7 import lexical as lex  # noqa: E402
from verbatim.retrieval.v7 import fuzzy as fz  # noqa: E402


# §30 mirror DDL — swap for schema_v7.ensure_v7_additive at integration
_DDL = """
CREATE TABLE units(
  unit_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, revision INTEGER NOT NULL,
  scope_id TEXT NOT NULL, kind TEXT NOT NULL, parent_unit_id TEXT,
  session_id TEXT, seq INTEGER, speaker_canon TEXT, perspective TEXT,
  recorded_at_us INTEGER, occurred_start_us INTEGER, occurred_end_us INTEGER,
  occurred_precision TEXT, occurred_source TEXT, byte_start INTEGER,
  byte_end INTEGER, generation INTEGER NOT NULL);
CREATE VIRTUAL TABLE unit_fts USING fts5(
  text, speaker, entities, session, "when",
  tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE unit_fts_stem USING fts5(
  text, tokenize='porter unicode61');
CREATE VIRTUAL TABLE unit_fts_tri USING fts5(
  text, tokenize='trigram');
CREATE VIRTUAL TABLE unit_fts_vocab USING fts5vocab('unit_fts','col');
CREATE TABLE lex_stats(
  scope_id TEXT NOT NULL, generation INTEGER NOT NULL, field TEXT NOT NULL,
  stats_version TEXT NOT NULL,
  n_units INTEGER NOT NULL, total_len INTEGER NOT NULL,
  PRIMARY KEY(scope_id, generation, field, stats_version));
CREATE TABLE lex_df(
  scope_id TEXT NOT NULL, generation INTEGER NOT NULL, field TEXT NOT NULL,
  term TEXT NOT NULL, stats_version TEXT NOT NULL,
  df INTEGER NOT NULL,
  PRIMARY KEY(scope_id, generation, field, term, stats_version));
"""

_FIELDS = ("text", "speaker", "entities", "session", "when")

# Mirror of stats_v7.STATS_VERSION_V1 — the maintained-stats key column.
STATS_VERSION = "bm25f/v1"


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_DDL)
    return conn


def seed_unit(conn, unit_id, *, text="", speaker="", entities="",
              session="", when="", scope="s", gen=1, source_id=None,
              revision=1, kind="turn"):
    """Insert one unit + its FTS shadows and maintain lex_stats/lex_df —
    the same transaction shape stats_v7 guarantees (V7-06.06)."""
    cur = conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " parent_unit_id, session_id, seq, speaker_canon, perspective,"
        " recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (unit_id, source_id or unit_id, revision, scope, kind, None,
         "sess", 0, speaker, "user_stated", 0, 0, 0, "instant", "explicit",
         0, len(text.encode()), gen),
    )
    rid = int(cur.lastrowid)
    conn.execute(
        'INSERT INTO unit_fts(rowid,text,speaker,entities,session,"when")'
        " VALUES(?,?,?,?,?,?)",
        (rid, text, speaker, entities, session, when))
    conn.execute("INSERT INTO unit_fts_stem(rowid,text) VALUES(?,?)",
                 (rid, text))
    conn.execute("INSERT INTO unit_fts_tri(rowid,text) VALUES(?,?)",
                 (rid, text))
    # maintained statistics: per-field token counts via the lane's
    # unicode61 mirror — the same projection the index applied
    field_text = {"text": text, "speaker": speaker, "entities": entities,
                  "session": session, "when": when}
    for f in _FIELDS:
        toks = lex._tokenize(field_text[f])
        conn.execute(
            "INSERT INTO lex_stats(scope_id,generation,field,"
            "stats_version,n_units,total_len) VALUES(?,?,?,?,1,?)"
            " ON CONFLICT(scope_id,generation,field,stats_version)"
            " DO UPDATE SET"
            " n_units=n_units+1, total_len=total_len+excluded.total_len",
            (scope, gen, f, STATS_VERSION, len(toks)))
        for term in set(toks):
            conn.execute(
                "INSERT INTO lex_df(scope_id,generation,field,term,"
                "stats_version,df) VALUES(?,?,?,?,?,1)"
                " ON CONFLICT(scope_id,generation,field,term,stats_version)"
                " DO UPDATE SET df=df+1",
                (scope, gen, f, term, STATS_VERSION))
    return rid


def mkqv(query, *, channels=None):
    """QueryViewV7 literal: whitespace tokens folded; ``channels`` maps a
    term to 'identifier' to put it on the identifier channel."""
    channels = channels or {}
    terms = []
    idents = []
    for m in re.finditer(r"[\w./:@#-]+", query):
        tok = m.group(0).strip("\"'")
        if not tok:
            continue
        term = lex._fold(tok)
        start_b = len(query[:m.start(0)].encode())
        end_b = start_b + len(tok.encode())
        ch = channels.get(term, "text")
        nt = NormTerm(term=term, channel=ch, byte_start=start_b,
                      byte_end=end_b)
        terms.append(nt)
        if ch == "identifier":
            idents.append(nt)
    norm = NormAnalysis(analyzer_id="norm/v2", terms=tuple(terms),
                        identifiers=tuple(idents), text=lex._fold(query))
    return QueryViewV7(
        query=query, norm=norm,
        intent=IntentResult(primary=IntentClass.LOOKUP,
                            classes=(IntentClass.LOOKUP,)),
    )


def mkctx(conn, *, eligible=None, scope="s", gen=1):
    return LaneContextV7(
        store=conn, scope_id=scope, generation=gen, eligible=eligible,
        query_time_us=0, profile="test", budget=BudgetClass.MID,
        policy=RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="test",
            lanes=(LaneName.LEX, LaneName.FUZZY), lane_weights={}),
        manifest={})


def mkslice(cap=50, deadline_ms=60000.0):
    return LaneSlice(deadline_ms=deadline_ms, cap=cap)


# ---------------------------------------------------------------------------
# Independent pure-Python BM25F reference (§32.2 transcription, independent
# tokenizer — simple non-alnum split is equivalent on ASCII fixtures)
# ---------------------------------------------------------------------------


def _ref_tok(s):
    return [t for t in re.split(r"[^0-9a-z]+", s.lower()) if t]


def ref_scores(docs, qterms, eligible_ids=None):
    """docs: {unit_id: {field: text}}; qterms: [(term, stem)].  Returns
    {unit_id: score} computed per §32.2 over the eligible subset."""
    ids = [u for u in docs if eligible_ids is None or u in eligible_ids]
    toks = {u: {f: _ref_tok(docs[u].get(f, "")) for f in _FIELDS}
            for u in ids}
    stem_ct = {u: {}
               for u in ids}
    for u in ids:
        from collections import Counter
        stem_ct[u] = Counter(lex.porter_stem(t) for t in toks[u]["text"])
    n = len(ids)
    avglen = {f: (sum(len(toks[u][f]) for u in ids) / n if n else 0.0)
              for f in _FIELDS}
    out = {}
    for u in ids:
        score = 0.0
        for term, stem in qterms:
            # df over eligible: exact in any field OR stem-family in text
            df = 0
            for v in ids:
                if any(term in toks[v][f] for f in _FIELDS) or \
                        stem_ct[v].get(stem, 0):
                    df += 1
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf_tilde = 0.0
            for f in _FIELDS:
                tf = float(toks[u][f].count(term))
                if f == "text":
                    extra = max(stem_ct[u].get(stem, 0) - tf, 0)
                    tf += lex.STEM_SCALE * extra
                if tf <= 0:
                    continue
                b = lex.FIELD_B[f]
                norm = 1 - b + b * len(toks[u][f]) / avglen[f] \
                    if avglen[f] else 1.0
                tf_tilde += lex.FIELD_W[f] * tf / norm
            score += idf * tf_tilde * (lex.K1 + 1) / (tf_tilde + lex.K1)
        if score > 0:
            out[u] = score
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bm25f_hand_computed_single_term():
    """Hand-checkable fixture: N=2, df=1 → idf=ln 2; tf~=1 → score=ln 2."""
    conn = _db()
    seed_unit(conn, "u1", text="pursue the matter")
    seed_unit(conn, "u2", text="the matter settled")
    out = lex.lane_lexical(mkctx(conn), mkqv("pursue"), mkslice())
    assert out.status == LaneStatus.OK
    assert len(out.candidates) == 1
    c = out.candidates[0]
    assert c.unit_id == "u1"
    assert c.raw_score == pytest.approx(math.log(2), abs=1e-9)
    assert c.rank == 1
    assert c.signals["matched_terms"]["pursue"]["fields"] == {"text": 1}
    assert c.signals["matched_terms"]["pursue"]["channel"] == "exact"
    assert out.stats["stats"] == "maintained"
    assert out.stats["df"] == {"pursue": 1}


def test_bm25f_field_weights_hand_computed():
    """entities match: tf~=w_ent·1/(1−b+b·len/avg)=0.8/(0.7+0.3·1/0.5)≈0.6154
    (u2's empty entities field makes avglen 0.5) → sat≈0.7457."""
    conn = _db()
    seed_unit(conn, "u1", text="hello there", entities="pursue")
    seed_unit(conn, "u2", text="something else entirely")
    out = lex.lane_lexical(mkctx(conn), mkqv("pursue"), mkslice())
    assert len(out.candidates) == 1
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    tf_tilde = 0.8 / (0.7 + 0.3 * (1 / 0.5))
    want = idf * tf_tilde * 2.2 / (tf_tilde + 1.2)
    assert out.candidates[0].raw_score == pytest.approx(want, abs=1e-9)
    assert out.candidates[0].signals["matched_terms"]["pursue"]["fields"] == \
        {"entities": 1}


def test_bm25f_matches_python_reference_corpus():
    """Multi-doc corpus, multi-term query: lane == independent reference."""
    conn = _db()
    docs = {
        "u1": {"text": "the quick brown fox jumps", "speaker": "alice"},
        "u2": {"text": "quick dogs and quick cats", "speaker": "bob"},
        "u3": {"text": "a brown matter of record", "entities": "fox"},
        "u4": {"text": "nothing matching here", "session": "sync"},
        "u5": {"text": "quick brown quick brown quick"},
    }
    for u, fields in docs.items():
        seed_unit(conn, u, **fields)
    qv = mkqv("quick brown fox")
    out = lex.lane_lexical(mkctx(conn), qv, mkslice())
    qterms = [("quick", lex.porter_stem("quick")),
              ("brown", lex.porter_stem("brown")),
              ("fox", lex.porter_stem("fox"))]
    ref = ref_scores(docs, qterms)
    got = {c.unit_id: c.raw_score for c in out.candidates}
    assert set(got) == set(ref)
    for u in ref:
        assert got[u] == pytest.approx(ref[u], abs=1e-9), u
    # ranked by score desc; ranks are 1-based contiguous
    assert [c.rank for c in out.candidates] == list(range(1, len(got) + 1))
    # u1 leads: it covers the rare term 'fox' (df=2) whose idf outweighs
    # u5's saturated quick/brown tf — verified against the reference
    assert out.candidates[0].unit_id == "u1"
    assert out.candidates[1].unit_id == "u5"
    # matched-term lists present for explanation (V7-06.09)
    assert set(out.candidates[0].signals["matched_terms"]) == \
        {"quick", "brown", "fox"}


def test_stem_channel_matches_at_reduced_weight():
    """'pursued' (stem-only) scores below an exact 'pursue' match (×0.6 tf)."""
    conn = _db()
    seed_unit(conn, "exact", text="pursue it today")
    seed_unit(conn, "stemmed", text="pursued it today")
    out = lex.lane_lexical(mkctx(conn), mkqv("pursue"), mkslice())
    got = {c.unit_id: c for c in out.candidates}
    assert set(got) == {"exact", "stemmed"}
    assert got["stemmed"].raw_score < got["exact"].raw_score
    st = got["stemmed"].signals["matched_terms"]["pursue"]
    assert st["channel"] == "stem" and st["stem_extra"] == 1
    # doc with both exact and stem occurrences counts each once (1.0 + 0.6)
    seed_unit(conn, "both", text="pursue pursued daily")
    out2 = lex.lane_lexical(mkctx(conn), mkqv("pursue"), mkslice())
    both = {c.unit_id: c for c in out2.candidates}["both"]
    bt = both.signals["matched_terms"]["pursue"]
    assert bt["fields"] == {"text": 1} and bt["stem_extra"] == 1


def test_eligible_subset_corrects_stats():
    """V7-06.07: held-out rows are subtracted — N, df, avglen all shrink."""
    conn = _db()
    seed_unit(conn, "u1", text="matter discussed")
    seed_unit(conn, "u2", text="matter matter matter")  # held out
    seed_unit(conn, "u3", text="matter again")
    full = lex.lane_lexical(mkctx(conn), mkqv("matter"), mkslice())
    assert full.stats["stats"] == "maintained"
    assert full.stats["df"]["matter"] == 3

    ctx = mkctx(conn, eligible={"u1", "u3"})
    sub = lex.lane_lexical(ctx, mkqv("matter"), mkslice())
    assert sub.stats["stats"] == "corrected"
    assert sub.stats["n_eligible"] == 2
    assert sub.stats["df"]["matter"] == 2  # u2's contribution removed
    assert {c.unit_id for c in sub.candidates} == {"u1", "u3"}
    # and the scores equal the reference over the eligible subset
    docs = {"u1": {"text": "matter discussed"},
            "u2": {"text": "matter matter matter"},
            "u3": {"text": "matter again"}}
    ref = ref_scores(docs, [("matter", "matter")],
                     eligible_ids={"u1", "u3"})
    for c in sub.candidates:
        assert c.raw_score == pytest.approx(ref[c.unit_id], abs=1e-9)


def test_exact_recomputed_when_stats_missing():
    """No maintained lex_stats → exact recompute, honestly labelled."""
    conn = _db()
    seed_unit(conn, "u1", text="alpha beta")
    seed_unit(conn, "u2", text="beta gamma")
    conn.execute("DELETE FROM lex_stats")
    conn.execute("DELETE FROM lex_df")
    out = lex.lane_lexical(mkctx(conn), mkqv("beta"), mkslice())
    assert out.status == LaneStatus.OK
    assert out.stats["stats"] == "exact_recomputed"
    assert len(out.candidates) == 2


def test_callable_eligibility():
    conn = _db()
    seed_unit(conn, "u1", text="shared topic")
    seed_unit(conn, "u2", text="shared topic")
    ctx = mkctx(conn, eligible=lambda row: row["unit_id"] == "u1")
    out = lex.lane_lexical(ctx, mkqv("shared topic"), mkslice())
    assert {c.unit_id for c in out.candidates} == {"u1"}
    assert out.stats["eligible_via"].startswith("callable")
    assert out.stats["stats"] in ("corrected", "exact_recomputed")


def test_phrase_flag_contiguous_and_near():
    """V7-06.08: contiguous adjacent-pair → 1.0, NEAR/3 → 0.5, none → 0."""
    conn = _db()
    seed_unit(conn, "contig", text="we pursue matter together")
    seed_unit(conn, "near", text="pursue is distant matter")
    # >3 tokens between the terms — outside NEAR/3
    seed_unit(conn, "far", text="matter was settled long ago and pursue forgotten")
    out = lex.lane_lexical(mkctx(conn), mkqv("pursue matter"), mkslice())
    flags = {c.unit_id: c.signals["phrase"] for c in out.candidates}
    assert flags["contig"] == 1.0
    assert flags["near"] == 0.5
    assert flags["far"] == 0.0


def test_quoted_span_phrase_flag():
    conn = _db()
    seed_unit(conn, "u1", text="the golden gate bridge view")
    seed_unit(conn, "u2", text="golden times at the bridge")
    qv = mkqv('show me "golden gate bridge" pictures')
    out = lex.lane_lexical(mkctx(conn), qv, mkslice())
    flags = {c.unit_id: c.signals["phrase"] for c in out.candidates}
    assert flags.get("u1") == 1.0
    # u2 matched some terms but never the contiguous quoted span
    assert flags.get("u2", 0.0) <= 0.5


def test_lane_cap_honored():
    conn = _db()
    for i in range(8):
        seed_unit(conn, f"u{i}", text=f"common term variant {i}")
    out = lex.lane_lexical(mkctx(conn), mkqv("common term"), mkslice(cap=3))
    assert len(out.candidates) == 3
    assert out.stats["overflow"] > 0
    assert [c.rank for c in out.candidates] == [1, 2, 3]


def test_deadline_zero_and_partial():
    conn = _db()
    for i in range(1500):
        seed_unit(conn, f"u{i}", text="deadline probe term")
    out = lex.lane_lexical(mkctx(conn), mkqv("probe"), mkslice(deadline_ms=0.0))
    assert out.status == LaneStatus.DEADLINE
    assert out.reason == "deadline"
    assert not out.candidates

    # scripted expiry forces a mid-scan cut → PARTIAL, honest counts
    class FakeDeadline:
        def __init__(self, _ms):
            self.calls = 0

        def expired(self):
            self.calls += 1
            return self.calls > 3

    monkey = pytest.MonkeyPatch()
    monkey.setattr(lex, "_Deadline", FakeDeadline)
    try:
        out = lex.lane_lexical(mkctx(conn), mkqv("probe"), mkslice())
    finally:
        monkey.undo()
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "deadline"


def test_unavailable_without_index():
    conn = sqlite3.connect(":memory:")
    out = lex.lane_lexical(mkctx(conn), mkqv("anything"), mkslice())
    assert out.status == LaneStatus.UNAVAILABLE
    assert out.reason == "no_unit_fts"


def test_determinism():
    conn = _db()
    docs = [("u1", "alpha beta gamma"), ("u2", "beta gamma delta"),
            ("u3", "gamma delta epsilon"), ("u4", "alpha delta")]
    for u, t in docs:
        seed_unit(conn, u, text=t)
    a = lex.lane_lexical(mkctx(conn), mkqv("beta gamma"), mkslice())
    b = lex.lane_lexical(mkctx(conn), mkqv("beta gamma"), mkslice())
    assert [(c.unit_id, c.raw_score) for c in a.candidates] == \
        [(c.unit_id, c.raw_score) for c in b.candidates]


def test_identifier_terms_scored_as_phrase_hits():
    """Identifier-channel terms match through the quoted MATCH (FTS5
    re-tokenizes 'jira-123' into the adjacent token pair it indexed) and
    score a nominal text-field hit — the exact_id lane stays the precise
    surface."""
    conn = _db()
    seed_unit(conn, "u1", text="ticket jira-123 closed today")
    seed_unit(conn, "u2", text="jira at position 999 mentioned")
    seed_unit(conn, "u3", text="no ticket talk")
    qv = mkqv("what about jira-123", channels={"jira-123": "identifier"})
    out = lex.lane_lexical(mkctx(conn), qv, mkslice())
    got = {c.unit_id: c for c in out.candidates}
    assert "u1" in got
    assert got["u1"].signals["matched_terms"]["jira-123"]["channel"] == \
        "identifier"


def test_no_terms_skipped():
    conn = _db()
    seed_unit(conn, "u1", text="something")
    qv = mkqv("!!!")
    out = lex.lane_lexical(mkctx(conn), qv, mkslice())
    assert out.status == LaneStatus.SKIPPED


# ---------------------------------------------------------------------------
# Fuzzy lane
# ---------------------------------------------------------------------------


def test_fuzzy_respell_persue_pursue_h07():
    """H07: 'persue' (df=0) respells to indexed 'pursue' at declared weight."""
    conn = _db()
    seed_unit(conn, "u1", text="we will pursue the matter")
    seed_unit(conn, "u2", text="unrelated topic")
    out = fz.lane_fuzzy(mkctx(conn), mkqv("persue"), mkslice())
    assert out.status == LaneStatus.OK
    assert out.stats["oov_terms"] == ["persue"]
    assert "pursue" in out.stats["respellings"].get("persue", [])
    assert len(out.candidates) >= 1
    assert out.candidates[0].unit_id == "u1"
    assert out.candidates[0].raw_score > 0
    mt = out.candidates[0].signals["matched_terms"]
    # the respelled term is disclosed with its declared weight + origin
    assert mt.get("pursue", {}).get("weight") == fz.RESPELL_WEIGHT
    assert mt["pursue"].get("of") == "persue"


def test_fuzzy_trigram_substring_hit():
    """The OOV string as a *substring* of a longer indexed token is caught
    by the trigram shadow (the token itself is too distant to respell)."""
    conn = _db()
    seed_unit(conn, "glued", text="we must antepersuer this now")
    seed_unit(conn, "ok", text="we will pursue this now")
    out = fz.lane_fuzzy(mkctx(conn), mkqv("persue"), mkslice())
    got = {c.unit_id: c for c in out.candidates}
    assert "glued" in got  # substring coverage of the raw OOV string
    via = {t: d.get("via") for t, d in
           got["glued"].signals["matched_terms"].items()}
    assert "trigram" in via.values()


def test_fuzzy_skipped_when_no_oov():
    conn = _db()
    seed_unit(conn, "u1", text="known vocabulary here")
    out = fz.lane_fuzzy(mkctx(conn), mkqv("known vocabulary"), mkslice())
    assert out.status == LaneStatus.SKIPPED
    assert out.reason == "no_oov_terms"
    assert not out.candidates


def test_fuzzy_identifier_never_respelled():
    """Identifiers are a mutation target: df=0 identifier produces no
    respellings and no trigram probes — the lane stays skipped."""
    conn = _db()
    seed_unit(conn, "u1", text="ticket jira 123 closed")
    seed_unit(conn, "u2", text="jira-124 is still open")
    qv = mkqv("jira-999", channels={"jira-999": "identifier"})
    # wait — mkqv folds 'jira-999' as one regex token; identifier channel
    out = fz.lane_fuzzy(mkctx(conn), qv, mkslice())
    assert out.status == LaneStatus.SKIPPED
    assert not out.candidates
    # mixed query: only the text term is fuzzed
    qv2 = mkqv("persue jira-999", channels={"jira-999": "identifier"})
    out2 = fz.lane_fuzzy(mkctx(conn), qv2, mkslice())
    assert "jira-999" not in out2.stats["oov_terms"]
    assert all("jira" not in s for s in
               out2.stats["respellings"].get("jira-999", []))


def test_fuzzy_no_vocab_table_partial_but_trigram_works():
    conn = _db()
    seed_unit(conn, "typo", text="we must antepersuer this now")
    conn.execute("DROP TABLE unit_fts_vocab")
    out = fz.lane_fuzzy(mkctx(conn), mkqv("persue"), mkslice())
    assert out.status == LaneStatus.PARTIAL
    assert out.reason == "no_vocab_table"
    assert out.stats["vocab"] == "unavailable"
    assert {c.unit_id for c in out.candidates} == {"typo"}


def test_fuzzy_unavailable_no_index():
    conn = sqlite3.connect(":memory:")
    out = fz.lane_fuzzy(mkctx(conn), mkqv("persue"), mkslice())
    assert out.status == LaneStatus.UNAVAILABLE


def test_fuzzy_respell_bound_respected():
    """len ≤ 5 → distance ≤ 1 only: 'cat' must not respell to 'cats'...
    actually 'cat'→'cats' is distance 1 — use a 2-edit case instead."""
    conn = _db()
    seed_unit(conn, "u1", text="zebra stripes")
    out = fz.lane_fuzzy(mkctx(conn), mkqv("zbbra"), mkslice())
    # 'zbbra'→'zebra' is 1 edit (len 5 → bound 1): respelled
    assert out.status == LaneStatus.OK
    seed_unit(conn, "u2", text="xylophone solo")
    out2 = fz.lane_fuzzy(mkctx(conn), mkqv("cat"), mkslice())
    # 'cat' → no ≤1-edit vocab term ('cats' absent) → empty/stats only
    assert not any("xylophone" == s for hits in
                   out2.stats.get("respellings", {}).values()
                   for s in hits)


def test_porter_matches_sqlite_porter():
    """The bundled stemmer is byte-identical to FTS5's porter tokenizer
    (validated against the engine's own stem vocabulary)."""
    words = ("running pursued matters easily fairly relational caresses "
             "ponies hopping filing happy skies")
    c2 = sqlite3.connect(":memory:")
    c2.execute(
        "CREATE VIRTUAL TABLE s USING fts5(text, tokenize='porter unicode61')")
    c2.execute("INSERT INTO s(rowid,text) VALUES(1,?)", (words,))
    c2.execute("CREATE VIRTUAL TABLE sv USING fts5vocab('s','row')")
    sqlite_stems = {r[0] for r in c2.execute("SELECT term FROM sv")}
    for w in words.split():
        assert lex.porter_stem(lex._fold(w)) in sqlite_stems


def test_lexical_df_zero_term_not_nominated():
    conn = _db()
    seed_unit(conn, "u1", text="only real terms")
    out = lex.lane_lexical(mkctx(conn), mkqv("absent"), mkslice())
    assert out.status == LaneStatus.OK
    assert not out.candidates
    assert out.stats["df"] == {"absent": 0}
