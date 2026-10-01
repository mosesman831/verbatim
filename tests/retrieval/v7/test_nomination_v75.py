"""J12 / V75-04.02 — df-ordered nomination budget on the lexical lane.

The lexical lane must select *nominating* terms identifiers-first, then
content terms by ascending eligible-set df, capped at the policy budget
``nominate_terms_max`` (a Q1 formula-search constant resolved through
``load_policy`` → ``RetrievalPolicyV7`` → ``ctx.policy`` — the same
per-profile channel as ``lanes``/``lane_weights``).  Every parsed term
still collects postings and scores every nominated document — the budget
bounds nomination-set membership only, and ``stats["nomination_dropped"]``
reports each withheld term with its df and reason.

Same §30 mirror DDL as ``test_lexical.py`` (swap for
``schema_v7.ensure_v7_additive`` at integration); the seed helper
maintains ``lex_stats``/``lex_df`` exactly as stats_v7 will.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verbatim.core.types import ErrorCode, VerbatimError  # noqa: E402
from verbatim.core.types_v7 import (  # noqa: E402
    NOMINATE_TERMS_MAX_R0,
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
from verbatim.retrieval.v7 import policy as pol  # noqa: E402


# §30 mirror DDL — identical shape to test_lexical.py's.
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
STATS_VERSION = "bm25f/v1"


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_DDL)
    return conn


def seed_unit(conn, unit_id, *, text="", speaker="", entities="",
              session="", when="", scope="s", gen=1, source_id=None,
              revision=1, kind="turn"):
    """One unit + FTS shadows + maintained lex_stats/lex_df — the same
    transaction shape stats_v7 guarantees (V7-06.06)."""
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


def mkctx(conn, *, eligible=None, scope="s", gen=1, policy=None):
    return LaneContextV7(
        store=conn, scope_id=scope, generation=gen, eligible=eligible,
        query_time_us=0, profile="test", budget=BudgetClass.MID,
        policy=policy if policy is not None else RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="test",
            lanes=(LaneName.LEX, LaneName.FUZZY), lane_weights={}),
        manifest={})


def mkslice(cap=200, deadline_ms=60000.0):
    return LaneSlice(deadline_ms=deadline_ms, cap=cap)


# ---------------------------------------------------------------------------
# Fixture: 20 content terms with known, distinct eligible dfs.
#
#   zzrare      df=2   (u_rare + u_mixed)          <- the rarest term
#   termNN      df=N+3 (N=0..18, i.e. 3..21)       <- except term18: df=22
#                                                  (u_mixed also carries it)
#   u_mixed  "zzrare term18 pad"  — nominated via zzrare, scored by term18
#
# Query order is *descending* df with the rarest term LAST — the position
# the old ``_MAX_TERMS`` truncation would have punished first.
# ---------------------------------------------------------------------------

_TERMS = [f"term{i:02d}" for i in range(19)] + ["zzrare"]


def _df_fixture(conn):
    """Seed the 20-term df ladder; returns {term: expected_eligible_df}."""
    expected = {}
    for i in range(19):
        term = f"term{i:02d}"
        n = i + 3
        for k in range(n):
            seed_unit(conn, f"d{i}_{k}", text=f"{term} pad")
        expected[term] = n
    seed_unit(conn, "u_rare", text="zzrare unique")
    seed_unit(conn, "u_mixed", text="zzrare term18 pad")
    expected["zzrare"] = 2
    expected["term18"] += 1  # u_mixed
    return expected


def _query_text():
    # descending df, rarest last — worst case for positional truncation
    return " ".join(reversed(_TERMS[:-1])) + " zzrare"


# ---------------------------------------------------------------------------
# J12 — the acceptance scenario
# ---------------------------------------------------------------------------


def test_j12_df_ordered_budget_rarest_always_nominates():
    """A 20-content-term query under budget=8 nominates the eight
    lowest-df terms; the rarest (last in the query string) is always
    among them, and coverage lists every dropped term with df+reason."""
    conn = _db()
    expected_df = _df_fixture(conn)
    policy = pol.load_policy("test", {"nominate_terms_max": 8})
    out = lex.lane_lexical(mkctx(conn, policy=policy),
                         mkqv(_query_text()), mkslice())
    assert out.status == LaneStatus.OK

    # every term still measured (postings collected for all) — the df
    # report covers all 20, not just the nominating eight
    assert out.stats["df"] == expected_df
    assert out.stats["nomination_eligible_terms"] == 20
    assert out.stats["nominate_terms_max"] == 8
    assert out.stats["nominated_terms"] == 8

    # docs reachable only via a nominating term: kept = the eight
    # lowest-df terms (zzrare df=2, term00..term06 df=3..9)
    got = {c.unit_id for c in out.candidates}
    assert "u_rare" in got          # the rarest term's only-doc
    assert "u_mixed" in got         # zzrare nominated it; term18 scores it
    # V8-07.03: the starved nomination (< K_rescue eligible candidates)
    # fires the bounded rescue over the ≤2 rarest dropped terms —
    # term07 + term08's docs arrive marked ``signals.rescue``, never
    # as nominated evidence; deeper dropped terms stay undelivered.
    rescue = out.stats["coverage_lexical"]["rescue"]
    assert rescue["fired"] is True
    assert rescue["terms"] == ["term07", "term08"]
    rescued = {
        c.unit_id for c in out.candidates if c.signals.get("rescue")
    }
    assert rescued
    assert all(
        u.startswith(("d7_", "d8_")) for u in rescued
    ), rescued
    for i in range(9, 19):
        assert not any(
            u.startswith(f"d{i}_") for u in got
        ), f"term{i:02d} docs leaked"

    # coverage: exactly the 12 highest-df terms dropped, honestly
    dropped = out.stats["nomination_dropped"]
    assert len(dropped) == 12
    assert all(d["reason"] == "budget" for d in dropped)
    by_term = {d["term"]: d["df"] for d in dropped}
    for i in range(7, 19):
        term = f"term{i:02d}"
        assert by_term[term] == expected_df[term]

    # a dropped term still SCORES a nominated doc (term18 on u_mixed)
    mixed = next(c for c in out.candidates if c.unit_id == "u_mixed")
    assert "term18" in mixed.signals["matched_terms"]
    assert "zzrare" in mixed.signals["matched_terms"]


def test_j12_rarest_nominates_at_any_budget():
    """Parametrize budgets: with identifiers absent the lowest-df term
    is always first in selection order, so it nominates at budget ≥ 1."""
    conn = _db()
    _df_fixture(conn)
    for budget in (1, 3, 19):
        policy = pol.load_policy("test", {"nominate_terms_max": budget})
        out = lex.lane_lexical(mkctx(conn, policy=policy),
                             mkqv(_query_text()), mkslice())
        assert out.stats["nominated_terms"] == budget
        got = {c.unit_id for c in out.candidates}
        assert {"u_rare", "u_mixed"} <= got
        assert len(out.stats["nomination_dropped"]) == 20 - budget


def test_j12_selection_is_deterministic():
    """Tie-free fixture still: identical inputs → identical stats/cands."""
    conn = _db()
    _df_fixture(conn)
    policy = pol.load_policy("test", {"nominate_terms_max": 8})
    a = lex.lane_lexical(mkctx(conn, policy=policy),
                       mkqv(_query_text()), mkslice())
    b = lex.lane_lexical(mkctx(conn, policy=policy),
                       mkqv(_query_text()), mkslice())
    assert [(c.unit_id, c.raw_score) for c in a.candidates] == \
        [(c.unit_id, c.raw_score) for c in b.candidates]
    assert a.stats["nomination_dropped"] == b.stats["nomination_dropped"]


# ---------------------------------------------------------------------------
# Policy channel
# ---------------------------------------------------------------------------


def test_budget_flows_through_policy_channel():
    """``load_policy`` resolves ``nominate_terms_max`` top-level and via
    the ``profiles`` overlay; the lane applies the resolved value."""
    p = pol.load_policy("test", {"nominate_terms_max": 5})
    assert p.nominate_terms_max == 5
    # profile overlay wins over the top-level value
    doc = {"nominate_terms_max": 9,
           "profiles": {"p2": {"nominate_terms_max": 4}}}
    assert pol.load_policy("p2", doc).nominate_terms_max == 4
    assert pol.load_policy("test", doc).nominate_terms_max == 9
    # r0 default when the document is silent
    assert pol.load_policy("test").nominate_terms_max == \
        NOMINATE_TERMS_MAX_R0 == 32
    assert pol.load_policy("test").nominate_df_theta is None
    # coverage view prints the applied knobs
    view = pol.policy_view(pol.load_policy("test", {"nominate_terms_max": 7,
                                                  "nominate_df_theta": 0.25}))
    assert view["nominate_terms_max"] == 7
    assert view["nominate_df_theta"] == 0.25

    # and the lane honors the resolved policy object end-to-end
    conn = _db()
    _df_fixture(conn)
    out = lex.lane_lexical(mkctx(conn, policy=p), mkqv(_query_text()),
                         mkslice())
    assert out.stats["nominate_terms_max"] == 5
    assert out.stats["nominated_terms"] == 5


@pytest.mark.parametrize("bad", [0, -3, "8", 1.5, True, None])
def test_nominate_terms_max_rejects_bad_values(bad):
    with pytest.raises(VerbatimError) as exc:
        pol.load_policy("test", {"nominate_terms_max": bad})
    assert exc.value.code == ErrorCode.VALIDATION


@pytest.mark.parametrize("bad", [0.0, -0.2, 1.5, "0.25", True,
                                 float("inf"), float("nan")])
def test_nominate_df_theta_rejects_bad_values(bad):
    with pytest.raises(VerbatimError) as exc:
        pol.load_policy("test", {"nominate_df_theta": bad})
    assert exc.value.code == ErrorCode.VALIDATION


def test_nomination_knobs_validate_hand_built_policies():
    """A hand-constructed policy carrying a malformed knob fails loudly
    at the lane — never silently reconfigures nomination."""
    conn = _db()
    seed_unit(conn, "u1", text="alpha beta")
    bad = RetrievalPolicyV7(
        policy_id="p", profile="test", lanes=(LaneName.LEX,),
        lane_weights={}, nominate_terms_max=0)
    with pytest.raises(VerbatimError):
        lex.lane_lexical(mkctx(conn, policy=bad), mkqv("alpha"), mkslice())


# ---------------------------------------------------------------------------
# Regression — default budget, under the cap
# ---------------------------------------------------------------------------


def test_default_budget_under_cap_identical_nomination():
    """At the r0 default (32) a query under the cap nominates exactly the
    union of all nominating-term postings — identical to pre-V75-04.02."""
    conn = _db()
    docs = {
        "u1": "alpha beta gamma",
        "u2": "beta gamma delta",
        "u3": "gamma delta epsilon",
        "u4": "alpha delta",
        "u5": "unrelated content",
    }
    for u, t in docs.items():
        seed_unit(conn, u, text=t)
    # default policy object (no document) → r0 budget
    out = lex.lane_lexical(mkctx(conn), mkqv("alpha beta gamma"), mkslice())
    assert out.status == LaneStatus.OK
    assert out.stats["nominate_terms_max"] == NOMINATE_TERMS_MAX_R0
    assert out.stats["nominated_terms"] == 3
    assert out.stats["nomination_dropped"] == []
    # union of the three terms' postings: u1..u4
    assert {c.unit_id for c in out.candidates} == {"u1", "u2", "u3", "u4"}


def test_no_terms_still_skipped():
    conn = _db()
    seed_unit(conn, "u1", text="something here")
    out = lex.lane_lexical(mkctx(conn), mkqv("!!!"), mkslice())
    assert out.status == LaneStatus.SKIPPED


# ---------------------------------------------------------------------------
# Stopword handling (unchanged)
# ---------------------------------------------------------------------------


def test_stopword_only_overlap_never_nominates():
    """Content-bearing query: a doc sharing only stopwords stays out."""
    conn = _db()
    seed_unit(conn, "u1", text="alpha beta")
    seed_unit(conn, "u2", text="the and of")          # stopwords only
    out = lex.lane_lexical(mkctx(conn), mkqv("alpha the and"), mkslice())
    assert {c.unit_id for c in out.candidates} == {"u1"}
    # the stopwords are not in the nominating pool at all — they are
    # excluded by the content rule, not the budget, so nothing is
    # reported dropped
    assert out.stats["nomination_eligible_terms"] == 1
    assert out.stats["nomination_dropped"] == []
    # u1 contains no stopword from the query → only "alpha" matched
    assert set(out.candidates[0].signals["matched_terms"]) == {"alpha"}


def test_facetless_query_still_nominates_on_every_term():
    """No content terms at all → every term nominates (V7-11.01
    fallback), exactly as before the budget existed."""
    conn = _db()
    seed_unit(conn, "u1", text="alpha beta")
    seed_unit(conn, "u2", text="the and of")
    out = lex.lane_lexical(mkctx(conn), mkqv("the and of"), mkslice())
    assert {c.unit_id for c in out.candidates} == {"u2"}
    assert out.stats["nomination_eligible_terms"] == 3
    assert out.stats["nominated_terms"] == 3
    assert out.stats["nomination_dropped"] == []


# ---------------------------------------------------------------------------
# Identifier-first ordering + θ arm (plumbed, off by default)
# ---------------------------------------------------------------------------


def test_identifiers_nominate_first():
    """Budget=1: the identifier takes the only slot; content terms are
    reported dropped by the budget, and the identifier's doc lands."""
    conn = _db()
    for i in range(6):
        seed_unit(conn, f"v{i}", text="filler alpha beta")
    seed_unit(conn, "v_id", text="refers ticket jira-77 resolved")
    policy = pol.load_policy("test", {"nominate_terms_max": 1})
    qv = mkqv("alpha beta jira-77", channels={"jira-77": "identifier"})
    out = lex.lane_lexical(mkctx(conn, policy=policy), qv, mkslice())
    assert out.stats["nominated_terms"] == 1
    got = {c.unit_id for c in out.candidates}
    assert "v_id" in got
    # V8-07.03: the one-candidate nomination fires the bounded rescue —
    # the two dropped content terms scan their eligible postings, so
    # the filler docs arrive marked ``signals.rescue`` while the
    # identifier's nominated doc stays unmarked.
    rescue = out.stats["coverage_lexical"]["rescue"]
    assert rescue["fired"] is True
    assert set(rescue["terms"]) == {"alpha", "beta"}
    rescued = {
        c.unit_id for c in out.candidates if c.signals.get("rescue")
    }
    assert {f"v{i}" for i in range(6)} <= rescued
    assert "v_id" not in rescued
    dropped = {d["term"]: d["reason"] for d in
               out.stats["nomination_dropped"]}
    assert dropped == {"alpha": "budget", "beta": "budget"}


def test_df_theta_arm_disarmed_by_default_and_excludes_when_armed():
    """``df > θ·N_E`` exclusion: off by default; armed, the over-common
    term drops with reason=df_threshold and consumes no budget slot."""
    conn = _db()
    # N_E = 11; common df=11, mid df=5, rareterm df=1 (u_rare)
    for i in range(10):
        seed_unit(conn, f"u{i}",
                  text="common filler" + (" mid" if i < 5 else ""))
    seed_unit(conn, "u_rare", text="rareterm common")

    # default: θ disarmed — nothing is excluded on df grounds
    out = lex.lane_lexical(mkctx(conn), mkqv("common mid rareterm"),
                         mkslice())
    assert "nominate_df_theta" not in out.stats
    assert all(d["reason"] != "df_threshold"
               for d in out.stats["nomination_dropped"])

    # armed: θ=0.5, N_E=11 → cut 5.5 — common(11) excluded, mid(5) kept
    policy = pol.load_policy("test", {"nominate_df_theta": 0.5})
    out = lex.lane_lexical(mkctx(conn, policy=policy),
                         mkqv("common mid rareterm"), mkslice())
    assert out.stats["nominate_df_theta"] == 0.5
    dropped = {d["term"]: d for d in out.stats["nomination_dropped"]}
    assert dropped["common"]["reason"] == "df_threshold"
    assert dropped["common"]["df"] == 11
    # the exclusion frees no budget slot question — both remaining terms
    # nominate even at budget=1... no: budget still caps; with default
    # budget both nominate
    assert out.stats["nominated_terms"] == 2
    got = {c.unit_id for c in out.candidates}
    assert "u_rare" in got                       # rareterm nominated
    assert all(u in got for u in
               ("u0", "u1", "u2", "u3", "u4"))   # mid nominated
    # V8-07.03: the gated term's docs may still arrive — only via the
    # bounded rescue, marked ``signals.rescue``; the gate's drop record
    # (reason=df_threshold) stands and common never *nominates*.
    u5 = next(c for c in out.candidates if c.unit_id == "u5")
    assert u5.signals.get("rescue") is True
    rescue = out.stats["coverage_lexical"]["rescue"]
    assert rescue["fired"] is True
    assert "common" in rescue["terms"]


def test_unknown_df_sorts_after_known_df():
    """A term whose df could not be measured (MATCH errored) orders after
    every known-df term — it never outranks a measured rare term."""
    conn = _db()
    seed_unit(conn, "u1", text="zzrare unique")
    seed_unit(conn, "u2", text="term00 pad")
    policy = pol.load_policy("test", {"nominate_terms_max": 1})

    orig = lex._match_rowids

    def flaky(c, table, match):
        if table == lex.FTS_TABLE and '"term00"' in match:
            return None  # simulate an exact-channel failure
        return orig(c, table, match)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(lex, "_match_rowids", flaky)
    try:
        out = lex.lane_lexical(mkctx(conn, policy=policy),
                             mkqv("term00 zzrare"), mkslice())
    finally:
        monkey.undo()
    # zzrare's df is known (1) → it nominates; term00 (df unknown) sorts
    # after it and drops to the budget
    assert out.stats["nominated_terms"] == 1
    got = {c.unit_id for c in out.candidates}
    assert "u1" in got
    # V8-07.03: starved nomination fires the bounded rescue over the
    # dropped term — u2 arrives marked ``signals.rescue``, never as a
    # nominated candidate.
    u2 = next(c for c in out.candidates if c.unit_id == "u2")
    assert u2.signals.get("rescue") is True
    assert out.stats["coverage_lexical"]["rescue"]["terms"] == ["term00"]
    dropped = {d["term"]: d for d in out.stats["nomination_dropped"]}
    assert dropped["term00"]["reason"] == "budget"
    assert dropped["term00"]["df"] is None
    assert out.stats["match_errors"] >= 1
