"""V8.5 §05 ship-set tests (SPEC_V8_5).

- V85-05.01: ``load_policy`` defaults to ``LANES_V85`` =
  ``(lex, fuzzy, dense, time)``; ``ent``/``graph``/``typed``/``obs``
  stay registered and reachable through a declared ``lanes`` list or a
  profile overlay.
- V85-05.02/05.03: the lexical co-occurrence nomination filter
  (``lexical.nom_cooc`` / ``lexical.cooc_min`` / ``lexical.rare_df``) —
  a unit must match ≥ ``cooc_min`` nominated terms; identifiers and
  rare terms (measured eligible df < ``rare_df``) are exempt; an empty
  filtered set falls back to the union and says so in
  ``coverage_lexical.cooc``.
- V85-05.04: the slim rerank feature set — ``ent_idf`` weight 0 and the
  seven dead features absent from the vector and ``score_detail``.
- The §23 params register (``policy.PARAM_DEFAULTS``) declares the
  shipped priors so ``PolicyPatch``-style ``params`` documents toggle
  them through the real ``load_policy`` seam.

Same §30 mirror DDL as ``test_nomination_v75.py`` (swap for
``schema_v7.ensure_v7_additive`` at integration).
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verbatim.core.types import ErrorCode, VerbatimError  # noqa: E402
from verbatim.core.types_v7 import (  # noqa: E402
    BudgetClass,
    FusedCandidate,
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
from verbatim.retrieval.v7 import lexical as lex  # noqa: E402
from verbatim.retrieval.v7 import policy as pol  # noqa: E402
from verbatim.retrieval.v7.rerank_features import (  # noqa: E402
    FEATURE_WEIGHTS_V1,
    score_candidates,
)

# ---------------------------------------------------------------------------
# §30 mirror DDL — identical shape to test_nomination_v75.py's.
# ---------------------------------------------------------------------------

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
              session="", when="", scope="s", gen=1):
    cur = conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " parent_unit_id, session_id, seq, speaker_canon, perspective,"
        " recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (unit_id, unit_id, 1, scope, "turn", None,
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


def mkctx(conn, *, policy=None, manifest=None):
    return LaneContextV7(
        store=conn, scope_id="s", generation=1, eligible=None,
        query_time_us=0, profile="test", budget=BudgetClass.MID,
        policy=policy if policy is not None else pol.load_policy("test"),
        manifest=manifest or {})


def mkslice(cap=200, deadline_ms=60000.0):
    return LaneSlice(deadline_ms=deadline_ms, cap=cap)


def _fused(unit, rrf, lane_ranks, signals=None):
    return FusedCandidate(
        unit_id=unit,
        source_id=f"src-{unit}",
        revision=1,
        rrf=rrf,
        lane_ranks=dict(lane_ranks),
        signals=signals or {},
    )


# ---------------------------------------------------------------------------
# V85-05.01 — default lane tuple
# ---------------------------------------------------------------------------


class TestDefaultLanes:
    def test_default_lanes_are_the_ship_set(self):
        p = pol.load_policy("test")
        assert p.lanes == (
            LaneName.LEX, LaneName.FUZZY, LaneName.DENSE, LaneName.TIME)
        assert pol.LANES_V85 == p.lanes

    def test_lanes_v1_keeps_the_full_eight(self):
        """The §32.3 column order is untouched — LANES_V1 still names all
        eight lanes the weight/cost tables are written in."""
        assert set(pol.LANES_V1) == {
            LaneName.LEX, LaneName.FUZZY, LaneName.DENSE, LaneName.ENT,
            LaneName.TIME, LaneName.GRAPH, LaneName.TYPED, LaneName.OBS,
        }

    def test_nondefault_lanes_reachable_via_lanes_decl(self):
        doc = {"lanes": ["lex", "ent", "graph", "typed", "obs"]}
        p = pol.load_policy("test", doc)
        assert p.lanes == (
            LaneName.LEX, LaneName.ENT, LaneName.GRAPH,
            LaneName.TYPED, LaneName.OBS)
        # and the materialized weight rows cover every declared lane
        row = p.lane_weights[IntentClass.LOOKUP]
        assert set(row) == set(p.lanes)

    def test_nondefault_lanes_reachable_via_profile(self):
        doc = {"profiles": {"wide": {"lanes": ["lex", "time", "graph"]}}}
        p = pol.load_policy("wide", doc)
        assert p.lanes == (LaneName.LEX, LaneName.TIME, LaneName.GRAPH)
        # the unselected profile still gets the ship set
        assert pol.load_policy("other", doc).lanes == pol.LANES_V85

    def test_extension_lane_scope_still_opt_in(self):
        """``scope`` is a registered extension lane — declared by name
        (its ``LaneName`` mints on demand, so compare values)."""
        p = pol.load_policy("test", {"lanes": ["lex", "scope"]})
        assert [lane.value for lane in p.lanes] == ["lex", "scope"]


# ---------------------------------------------------------------------------
# §23 params register declarations
# ---------------------------------------------------------------------------


class TestParamsRegister:
    def test_register_declares_the_v85_shipset(self):
        assert pol.PARAM_DEFAULTS["lexical.nom_cooc"] is True
        assert pol.PARAM_DEFAULTS["lexical.cooc_min"] == 2
        assert pol.PARAM_DEFAULTS["lexical.rare_df"] == 4
        assert pol.PARAM_DEFAULTS["temporal.as_of_scope"] == "window"
        # deadline.py already runs the S2a/S2b split — the declared
        # prior is the current behavior.
        assert pol.PARAM_DEFAULTS["scheduler.two_phase"] is True

    def test_register_matches_lexical_read_site_priors(self):
        """The register and the lane's §23 priors are one declaration —
        pinned equal so they cannot drift silently."""
        assert pol.PARAM_DEFAULTS["lexical.nom_cooc"] == \
            lex._NOM_COOC_DEFAULT
        assert pol.PARAM_DEFAULTS["lexical.cooc_min"] == \
            lex._COOC_MIN_DEFAULT
        assert pol.PARAM_DEFAULTS["lexical.rare_df"] == \
            lex._RARE_DF_DEFAULT

    def test_policy_param_uses_register_when_no_default_passed(self):
        bare = RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="test",
            lanes=pol.LANES_V85, lane_weights={})
        # register-backed names resolve without an explicit default
        assert pol.policy_param(bare, "temporal.as_of_scope") == "window"
        assert pol.policy_param(bare, "scheduler.two_phase") is True
        assert pol.policy_param(bare, "lexical.cooc_min") == 2
        # undeclared names still yield None
        assert pol.policy_param(bare, "scheduler.R_post") is None
        # an explicit default — including None — always wins
        assert pol.policy_param(bare, "temporal.as_of_scope", None) is None
        assert pol.policy_param(bare, "temporal.as_of_scope", "x") == "x"

    def test_params_document_overrides_register(self):
        p = pol.load_policy(
            "test", {"params": {"scheduler.two_phase": False,
                                "lexical.cooc_min": 3}})
        assert pol.policy_param(p, "scheduler.two_phase") is False
        assert pol.policy_param(p, "lexical.cooc_min") == 3
        # nested one-level form flattens to the same dotted key
        p2 = pol.load_policy(
            "test", {"params": {"temporal": {"as_of_scope": "global"}}})
        assert pol.policy_param(p2, "temporal.as_of_scope") == "global"
        # policy_view discloses the applied override, not the prior
        assert pol.policy_view(p)["params"]["lexical.cooc_min"] == 3


# ---------------------------------------------------------------------------
# V85-05.02/05.03 — co-occurrence nomination on the real lane
# ---------------------------------------------------------------------------


class TestCoocNomination:
    def _two_term_fixture(self, conn):
        """alpha df=5, beta df=5 (both ≥ rare_df — not rare); only
        ``u_both`` carries both terms."""
        seed_unit(conn, "u_both", text="alpha beta")
        for i in range(1, 5):
            seed_unit(conn, f"u_a{i}", text=f"alpha pad{i}")
            seed_unit(conn, f"u_b{i}", text=f"beta pad{i}")

    def test_one_term_unit_not_nominated(self):
        """V85-05.02: a unit matching only one nominated term is dropped
        when a ≥``cooc_min``-term unit exists."""
        conn = _db()
        self._two_term_fixture(conn)
        out = lex.lane_lexical(mkctx(conn), mkqv("alpha beta"),
                               mkslice())
        assert out.status == LaneStatus.OK
        got = {c.unit_id for c in out.candidates}
        assert got == {"u_both"}
        cooc = out.stats["coverage_lexical"]["cooc"]
        assert cooc["enabled"] is True
        assert cooc["min"] == 2 and cooc["rare_df"] == 4
        assert cooc["union"] == 9 and cooc["filtered"] == 1
        assert "fallback" not in cooc
        # the filter narrows the scored pool honestly
        assert out.stats["nominated"] == 1

    def test_empty_filtered_falls_back_to_union(self):
        """V85-05.02: disjoint non-rare terms (df=4 each, not < 4) → the
        filtered set is empty → the union nominates, honestly reported."""
        conn = _db()
        for i in range(1, 5):
            seed_unit(conn, f"u_g{i}", text=f"gamma pad{i}")
            seed_unit(conn, f"u_d{i}", text=f"delta pad{i}")
        out = lex.lane_lexical(mkctx(conn), mkqv("gamma delta"),
                               mkslice())
        got = {c.unit_id for c in out.candidates}
        assert got == (
            {f"u_g{i}" for i in range(1, 5)}
            | {f"u_d{i}" for i in range(1, 5)})
        cooc = out.stats["coverage_lexical"]["cooc"]
        assert cooc["filtered"] == 0
        assert cooc["fallback"] == "union"
        assert cooc["union"] == 8

    def test_rare_term_exempt_from_cooc(self):
        """V85-05.03: a kept term with measured df < ``rare_df``
        nominates its postings without satisfying the co-occurrence
        requirement; a common term's units still need ≥2."""
        conn = _db()
        seed_unit(conn, "u_r1", text="rare w1")
        seed_unit(conn, "u_r2", text="rare w2")
        for i in range(1, 6):
            seed_unit(conn, f"u_c{i}", text=f"common pad{i}")
        out = lex.lane_lexical(mkctx(conn), mkqv("rare common"),
                               mkslice())
        got = {c.unit_id for c in out.candidates}
        assert got == {"u_r1", "u_r2"}
        cooc = out.stats["coverage_lexical"]["cooc"]
        assert cooc["filtered"] == 2
        assert {"term": "rare", "reason": "rare", "df": 2} in \
            cooc["exempt_terms"]

    def test_identifier_exempt_from_cooc(self):
        """Identifiers nominate by identity (the df_theta discipline) —
        an identifier-only unit is never starved by co-occurrence."""
        conn = _db()
        seed_unit(conn, "u_id", text="ref ticket zz900")
        self._two_term_fixture(conn)
        qv = mkqv("alpha beta zz900", channels={"zz900": "identifier"})
        out = lex.lane_lexical(mkctx(conn), qv, mkslice())
        got = {c.unit_id for c in out.candidates}
        assert {"u_both", "u_id"} <= got
        cooc = out.stats["coverage_lexical"]["cooc"]
        assert {"term": "zz900", "reason": "identifier"} in \
            cooc["exempt_terms"]

    def test_nom_cooc_off_returns_union(self):
        """The ``lexical.nom_cooc = false`` arm restores the plain union
        — through the real ``params`` channel a PolicyPatch drives."""
        conn = _db()
        self._two_term_fixture(conn)
        p = pol.load_policy(
            "test", {"params": {"lexical.nom_cooc": False}})
        out = lex.lane_lexical(mkctx(conn, policy=p),
                               mkqv("alpha beta"), mkslice())
        got = {c.unit_id for c in out.candidates}
        assert len(got) == 9
        cooc = out.stats["coverage_lexical"]["cooc"]
        assert cooc["enabled"] is False
        assert cooc["filtered"] == cooc["union"] == 9

    def test_cooc_min_override_via_params(self):
        """``lexical.cooc_min = 3`` tightens the filter through the same
        ``params`` seam (here ``fable`` stays rare-exempt either way)."""
        conn = _db()
        seed_unit(conn, "u_all", text="tango quartz fable")
        for i in range(1, 5):
            seed_unit(conn, f"u_p{i}", text="tango quartz")
        qv = mkqv("tango quartz fable")
        # cooc_min=2 (default): the pair-docs survive at count 2
        out2 = lex.lane_lexical(mkctx(conn), qv, mkslice())
        assert {c.unit_id for c in out2.candidates} == \
            {"u_all", "u_p1", "u_p2", "u_p3", "u_p4"}
        # cooc_min=3: only the three-term unit survives
        p3 = pol.load_policy(
            "test", {"params": {"lexical.cooc_min": 3}})
        out3 = lex.lane_lexical(mkctx(conn, policy=p3), qv, mkslice())
        assert {c.unit_id for c in out3.candidates} == {"u_all"}
        cooc = out3.stats["coverage_lexical"]["cooc"]
        assert cooc["min"] == 3 and cooc["filtered"] == 1
        assert {"term": "fable", "reason": "rare", "df": 1} in \
            cooc["exempt_terms"]

    def test_rare_df_zero_disarms_exemption(self):
        """``lexical.rare_df = 0`` disarms the rare-term rescue — a df-2
        term's units then need the co-occurrence like everyone else."""
        conn = _db()
        seed_unit(conn, "u_r1", text="rare w1")
        seed_unit(conn, "u_r2", text="rare w2")
        for i in range(1, 6):
            seed_unit(conn, f"u_c{i}", text=f"common pad{i}")
        p = pol.load_policy(
            "test", {"params": {"lexical.rare_df": 0}})
        out = lex.lane_lexical(mkctx(conn, policy=p),
                               mkqv("rare common"), mkslice())
        cooc = out.stats["coverage_lexical"]["cooc"]
        # nothing rare-exempt, nothing co-occurring → union fallback
        assert cooc["filtered"] == 0
        assert cooc["fallback"] == "union"
        assert "exempt_terms" not in cooc
        assert len({c.unit_id for c in out.candidates}) == 7

    def test_malformed_arms_fail_loudly(self):
        conn = _db()
        self._two_term_fixture(conn)
        for bad in ({"lexical.cooc_min": 0},
                    {"lexical.cooc_min": "two"},
                    {"lexical.rare_df": -1},
                    {"lexical.nom_cooc": "yes"}):
            p = pol.load_policy("test", {"params": bad})
            with pytest.raises(VerbatimError) as ei:
                lex.lane_lexical(mkctx(conn, policy=p),
                                 mkqv("alpha beta"), mkslice())
            assert ei.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# V85-05.04 — slim rerank feature set
# ---------------------------------------------------------------------------

_DEAD = ("cov_idf_ctx", "ident_exact", "life_state", "corrob",
         "perspective_fit", "event_pred", "lane_agree")


class TestSlimRerank:
    def test_weight_table_is_the_slim_set(self):
        assert dict(FEATURE_WEIGHTS_V1) == {
            "rrf_norm": 1.0,
            "cov_idf": 0.9,
            "ent_idf": 0.0,
            "phrase": 0.3,
            "speaker_match": 0.4,
            "t_prox": 0.4,
        }

    def test_ent_idf_weight_is_zero(self):
        assert FEATURE_WEIGHTS_V1["ent_idf"] == 0.0

    def test_dead_features_absent_from_score_detail(self):
        """Even with every dead-feature input signal present —
        identifiers, lane ranks, corroboration, lifecycle, event
        predicates — the shipped vector and detail omit them."""
        q = QueryViewV7(
            query="when did alpha ship",
            norm=NormAnalysis(
                analyzer_id="norm/v2",
                terms=(NormTerm(term="alpha", channel="text",
                                byte_start=8, byte_end=13),),
                identifiers=(NormTerm(term="T-900", channel="identifier",
                                      byte_start=0, byte_end=5),),
                text="when did alpha ship"),
            intent=IntentResult(primary=IntentClass.CURRENT_VALUE,
                                classes=(IntentClass.CURRENT_VALUE,)),
            entity_canons=("alpha",),
            speaker_canon="mel",
        )
        cand = _fused("u1", 0.05, {"lex": 1, "dense": 2}, signals={
            "matched_terms": ("alpha",),
            "matched_terms_ctx": ("alpha", "beta"),
            "matched_canons": ("alpha",),
            "ident_exact": 1.0,
            "identifiers": ["T-900"],
            "event_pred": 1.0,
            "corroboration": 0.9,
            "lifecycle": "current",
            "perspective": "user_stated",
            "speaker_canon": "mel",
            "phrase": 1.0,
        })
        out = score_candidates(q, [cand])
        feats = out[0].detail["features"]
        for name in _DEAD:
            assert name not in feats, name
            assert name not in out[0].detail["weights"], name
            assert name not in out.stats["weights"], name
        assert not set(_DEAD) & set(out.stats["features_seen"])
        # the kept set still emits
        assert feats["rrf_norm"] == 1.0
        assert feats["cov_idf"] == 1.0
        assert feats["phrase"] == 1.0
        assert feats["speaker_match"] == 1.0
        assert feats["t_prox"] == 0.5   # no window → neutral
        assert feats["ent_idf"] == 1.0  # computed, but weighted 0
        assert out[0].detail["weights"]["ent_idf"] == 0.0
        # hand-computed: ent_idf × 0 contributes nothing
        expected = (1.0 * 1.0 + 0.9 * 1.0 + 0.0 * 1.0 + 0.3 * 1.0
                    + 0.4 * 1.0 + 0.4 * 0.5)
        assert math.isclose(out[0].detail["feature_score"], expected,
                            rel_tol=1e-12)
        assert out[0].score == out[0].detail["feature_score"]

    def test_ent_idf_arm_rearmable_via_weights(self):
        """A declared ``feature_weights`` arm (the PolicyPatch seam)
        re-arms ``ent_idf`` — the feature is kept, only its shipped
        coefficient is zero.  The harness merges overrides over
        ``FEATURE_WEIGHTS_V1`` and passes the merged table."""
        q = QueryViewV7(
            query="alpha",
            norm=NormAnalysis(
                analyzer_id="norm/v2",
                terms=(NormTerm(term="alpha", channel="text",
                                byte_start=0, byte_end=5),),
                identifiers=(),
                text="alpha"),
            intent=IntentResult(primary=IntentClass.LOOKUP,
                                classes=(IntentClass.LOOKUP,)),
            entity_canons=("alpha",),
        )
        cand = _fused("u1", 0.05, {"lex": 1},
                      signals={"matched_terms": ("alpha",),
                               "matched_canons": ("alpha",)})
        arm_weights = {**dict(FEATURE_WEIGHTS_V1), "ent_idf": 0.6}
        out = score_candidates(q, [cand], weights=arm_weights)
        assert out.stats["weights"]["ent_idf"] == 0.6
        assert out[0].detail["weights"]["ent_idf"] == 0.6
        # and it now contributes — 0.6 · 1.0 lands in the score
        # (rrf 1.0 + cov 0.9 + ent 0.6 + speaker 0.5 + t_prox 0.5)
        assert math.isclose(
            out[0].detail["feature_score"],
            1.0 * 1.0 + 0.9 * 1.0 + 0.6 * 1.0 + 0.4 * 0.5 + 0.4 * 0.5,
            rel_tol=1e-12)
