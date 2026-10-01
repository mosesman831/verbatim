"""V8 facets / multi-hop verification tests — SPEC_V8 §10, §21.7, §23,
§24, §25 scenarios K59–K62.

All machinery is real: ``querying/query_view.build_query_view`` and
``querying/intent_v2.decompose_structural`` run the production
norm/v2 + intent/v2 analysis; ``retrieval/v7/entity.lane_entity`` runs
against the real §30 schema (``ensure_v7_additive`` — units /
entity_mentions / entity_canon / entity_aliases_v7 / lex_stats);
``retrieval/v7/fusion.reserve_facet_slots`` and ``rrf_fuse`` run the
§21.7 reserved-share algorithm; ``retrieval/v7/pipeline._run_lane`` is
the production facet fan-out/merge site (D8-16).  No mocks.

- K59: coordinated ≥2-canon queries decompose structurally —
  "How did Melanie and Caroline each spend the summer?" yields 2
  canon-bound facets marked ``V8-10.01.structural``; a single-canon
  question yields 0; facets are one level deep and capped by
  ``pool.max_facets``.
- K60: when the whole query fills ``capv`` each facet keeps a reserved
  share; rollover is reported (``produced``/``kept``/``rolled_over``);
  facet runs execute on lex+ent only by default, dense via the
  ``facets.dense_per_facet`` arm.
- K61: with ≥2 canons resolved, units mentioning all of them emit
  first with ``signals["joint"] = True``.
- K62: a prolific canon contributes at most ``ent.per_canon_quota``
  (prior 50) non-joint emissions; the ``pack.group_maxpool`` arm has a
  decision record and reports its applied state honestly.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from verbatim.core.types import VerbatimError
from verbatim.core.types_v7 import (
    POOLS,
    BudgetClass,
    CandidateV7,
    IntentClass,
    IntentResult,
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    NormAnalysis,
    NormTerm,
    PackItemV7,
    QueryViewV7,
    RetrievalPolicyV7,
)
from verbatim.querying.intent_v2 import (
    COORD_LEXICON_ID,
    decompose_structural,
)
from verbatim.querying.query_view import (
    FACET_SOURCE_STRUCTURAL,
    build_query_view,
    facet_source,
)
from verbatim.retrieval.v7 import entity as ent_mod
from verbatim.retrieval.v7 import fusion as fusion_mod
from verbatim.retrieval.v7 import pipeline as pipeline_mod
from verbatim.retrieval.v7.entity import lane_entity
from verbatim.retrieval.v7.fusion import reserve_facet_slots, rrf_fuse
from verbatim.retrieval.v7.lanes_base import LANE_REGISTRY, register_lane
from verbatim.storage.schema_v7 import ensure_v7_additive

SCOPE = "scope-facets"
GEN = 1
T0 = 1_700_000_000_000_000

REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_REGISTER = REPO_ROOT / "research" / "v8_final_pack" / "arm_register_v8.md"


# ---------------------------------------------------------------------------
# fixtures / helpers — the real §30 schema, seeded like test_entity.py
# ---------------------------------------------------------------------------


def mk_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_v7_additive(conn)
    return conn


def add_unit(
    conn,
    uid,
    *,
    scope=SCOPE,
    gen=GEN,
    source=None,
    rev=1,
    speaker=None,
    session=None,
    seq=None,
):
    conn.execute(
        "INSERT OR REPLACE INTO units (unit_id, source_id, revision,"
        " scope_id, kind, parent_unit_id, session_id, seq, speaker_canon,"
        " perspective, recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            uid,
            source or f"src-{uid}",
            rev,
            scope,
            "turn",
            None,
            session,
            seq,
            speaker,
            "user_stated",
            T0,
            None,
            None,
            "unknown",
            "unknown",
            None,
            None,
            gen,
        ),
    )


def add_mention(
    conn,
    uid,
    c,
    *,
    surface=None,
    bs=0,
    be=4,
    role="mention",
    scope=SCOPE,
    gen=GEN,
):
    conn.execute(
        "INSERT INTO entity_mentions (scope_id, canon, unit_id,"
        " generation, surface, byte_start, byte_end, role)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (scope, c, uid, gen, surface if surface is not None else c, bs, be, role),
    )


def set_df(conn, c, df, *, scope=SCOPE, gen=GEN, display=None):
    conn.execute(
        "INSERT OR REPLACE INTO entity_canon"
        " (scope_id, canon, generation, display, df_units)"
        " VALUES (?,?,?,?,?)",
        (scope, c, gen, display or c, df),
    )


def set_lex_stats(conn, n, *, scope=SCOPE, gen=GEN, field="text"):
    conn.execute(
        "INSERT OR REPLACE INTO lex_stats"
        " (scope_id, generation, field, stats_version, n_units, total_len)"
        " VALUES (?,?,?,?,?,?)",
        (scope, gen, field, "bm25f/v1", n, n * 10),
    )


def mk_ctx(
    conn,
    *,
    eligible=None,
    gen=GEN,
    scope=SCOPE,
    manifest=None,
    policy_params=None,
    lanes=(LaneName.ENT,),
):
    if eligible is None:
        eligible = lambda row: True  # noqa: E731 — allow-all fixture
    policy = RetrievalPolicyV7(
        policy_id="retrieval_policy/v7",
        profile="test",
        lanes=tuple(lanes),
        lane_weights={},
    )
    if policy_params is not None:
        object.__setattr__(policy, "params", dict(policy_params))
    return LaneContextV7(
        store=conn,
        scope_id=scope,
        generation=gen,
        eligible=eligible,
        query_time_us=T0,
        profile="test",
        budget=BudgetClass.MID,
        policy=policy,
        manifest=dict(manifest or {}),
    )


def mk_qv(canons=(), query="q", *, facets=()):
    terms = tuple(
        NormTerm(term=t, channel="text", byte_start=0, byte_end=len(t))
        for t in query.split()
    )
    return QueryViewV7(
        query=query,
        norm=NormAnalysis(
            analyzer_id="norm/v2", terms=terms, identifiers=(), text=query
        ),
        intent=IntentResult(
            primary=IntentClass.LOOKUP, classes=(IntentClass.LOOKUP,)
        ),
        entity_canons=tuple(canons),
        facets=tuple(facets),
        query_time_us=T0,
    )


def sl(cap=50, ms=10_000.0):
    return LaneSlice(deadline_ms=ms, cap=cap)


def by_id(out):
    return {c.unit_id: c for c in out.candidates}


@pytest.fixture(autouse=True)
def clean_registry():
    saved = dict(LANE_REGISTRY)
    LANE_REGISTRY.clear()
    yield
    LANE_REGISTRY.clear()
    LANE_REGISTRY.update(saved)


def _facet_terms(facet: QueryViewV7) -> tuple:
    return tuple(t.term for t in facet.norm.terms if t.channel == "text")


# ---------------------------------------------------------------------------
# K59 — V8-10.01 structural facet decomposition (build_query_view /
# decompose_structural; D8-15: intent-gated decomposition missed every
# coordinated-subject question outside multi_hop/comparison)
# ---------------------------------------------------------------------------


class TestK59StructuralDecomposition:
    def test_k59_coordinated_canons_yield_two_structural_facets(self):
        """The spec's headline case: "How did Melanie and Caroline each
        spend the summer?" — two resolved canons joined by the owned
        coordination lexicon produce exactly two canon-bound facets."""
        view = build_query_view(
            "How did Melanie and Caroline each spend the summer?",
            now_us=T0,
            known_canons=("melanie", "caroline"),
        )
        assert len(view.facets) == 2
        assert facet_source(view) == "structural"
        seeds = [f.entity_canons[0] for f in view.facets]
        assert seeds == ["melanie", "caroline"]  # coordination order
        for f in view.facets:
            assert FACET_SOURCE_STRUCTURAL in f.intent.rule_trace

    def test_k59_single_canon_query_yields_zero_facets(self):
        view = build_query_view(
            "What did Caroline say about painting?",
            now_us=T0,
            known_canons=("melanie", "caroline"),
        )
        assert view.facets == ()
        assert facet_source(view) is None

    def test_k59_facet_is_restricted_to_one_canon(self):
        """Each facet keeps its own canon; every OTHER joined canon and
        the coordinator sites inside the coordination zone are removed
        from its nominating terms."""
        view = build_query_view(
            "How did Melanie and Caroline each spend the summer?",
            now_us=T0,
            known_canons=("melanie", "caroline"),
        )
        mel, car = view.facets
        assert "melanie" in mel.entity_canons
        assert "caroline" not in mel.entity_canons
        assert "caroline" not in _facet_terms(mel)
        assert "caroline" in car.entity_canons
        assert "melanie" not in car.entity_canons
        assert "melanie" not in _facet_terms(car)
        # coordinator sites inside the zone are dropped — no dangling
        # "and"/"each" survives in either facet's term stream.
        for f in view.facets:
            toks = _facet_terms(f)
            assert "and" not in toks
            assert "each" not in toks
            # the shared predicate context stays in every facet
            assert "spend" in toks and "summer" in toks

    def test_k59_decomposition_is_one_level_deep(self):
        view = build_query_view(
            "How did Melanie and Caroline each spend the summer?",
            now_us=T0,
            known_canons=("melanie", "caroline"),
        )
        assert view.facets
        for f in view.facets:
            assert f.facets == ()  # allow_facets=False on sub-views

    def test_k59_max_facets_cap_binds(self):
        """Three coordinated canons with max_facets=2 → two facets, in
        coordination order (the pool.max_facets cap the pipeline
        re-applies at run time)."""
        view = build_query_view(
            "How did Melanie and Caroline and Alice spend the summer?",
            now_us=T0,
            known_canons=("melanie", "caroline", "alice"),
            max_facets=2,
        )
        assert len(view.facets) == 2
        seeds = [f.entity_canons[0] for f in view.facets]
        assert seeds == ["melanie", "caroline"]

    def test_k59_structural_decompose_is_intent_independent(self):
        """D8-15's actual defect: decomposition used to be gated on the
        intent class.  Two resolved canons reachable only through the
        SPEAKER channel never reach the classifier — the primary stays
        non-decomposable (lookup) — yet the structural pass still emits
        two canon-bound facets."""
        view = build_query_view(
            "What did Melanie and Caroline each say?",
            now_us=T0,
            known_canons=(),
            known_speakers=("melanie", "caroline"),
        )
        assert view.intent.primary not in (
            IntentClass.MULTI_HOP,
            IntentClass.COMPARISON,
        )
        assert len(view.facets) == 2
        assert facet_source(view) == "structural"

    def test_k59_coordination_lexicon_variants(self):
        """The owned coord_lex/v1 list: joins (and/or/vs/as well as)
        bind the spans on both sides; markers (both/each/either/
        neither/between) bind the touching cluster.  ``decompose``
        itself stays (norm,)-shaped — these run the structural pass
        directly on a real norm/v2 analysis."""
        from verbatim.text.norm_v2 import analyze

        canons = ("melanie", "caroline")
        for text, n_seed in (
            ("Melanie and Caroline went", 2),
            ("Melanie or Caroline went", 2),
            ("Melanie vs Caroline", 2),
            ("Melanie as well as Caroline went", 2),
            ("both Melanie and Caroline went", 2),
            ("between Melanie and Caroline", 2),
        ):
            pairs = decompose_structural(analyze(text), canons, surface=text)
            assert len(pairs) == n_seed, text
            assert {seed for _norm, seed in pairs} == set(canons)
        # uncoordinated adjacent canons never bind
        assert decompose_structural(
            analyze("Melanie Caroline went"), canons,
            surface="Melanie Caroline went",
        ) == ()
        # a coordinator inside one canon is not a site ("rock and roll")
        assert decompose_structural(
            analyze("Rock and Roll show"), ("rock and roll",),
            surface="Rock and Roll show",
        ) == ()
        # one canon alone can never coordinate
        assert decompose_structural(
            analyze("Melanie and her sister went"), ("melanie",),
            surface="Melanie and her sister went",
        ) == ()

    def test_k59_unresolved_names_do_not_decompose(self):
        """Canons outside the scope vocabulary are never seeds — the
        coordination gate is on *resolved* canons (V8-10.01)."""
        view = build_query_view(
            "How did Melanie and Caroline each spend the summer?",
            now_us=T0,
            known_canons=("melanie",),  # caroline unresolved
        )
        assert view.facets == ()

    def test_k59_speaker_channel_canons_seed_facets(self):
        """Speaker-resolved canons feed coordination detection and facet
        seeding without ever joining ``entity_canons`` on the parent
        view — the facet pins ``speaker_canon`` for the speaker-seeded
        participant."""
        view = build_query_view(
            "What did Melanie and Caroline each say?",
            now_us=T0,
            known_canons=(),
            known_speakers=("melanie", "caroline"),
        )
        assert len(view.facets) == 2
        assert facet_source(view) == "structural"
        # parent keeps its honest class — speaker canons never join
        # entity_canons upstream (R14 protection).
        assert set(view.entity_canons) == set()
        seeds = []
        for f in view.facets:
            assert FACET_SOURCE_STRUCTURAL in f.intent.rule_trace
            # the seed is pinned on the facet (speaker seed → speaker_canon)
            seed = f.speaker_canon
            assert seed in ("melanie", "caroline")
            seeds.append(seed)
        assert seeds == ["melanie", "caroline"]

    def test_k59_coord_lexicon_id_is_versioned(self):
        """The owned coordination lexicon carries a versioned id for
        coverage/explain attribution."""
        assert COORD_LEXICON_ID == "coord_lex/v1"


# ---------------------------------------------------------------------------
# K60 — V8-10.02 reserved facet slots + V8-10.06 facet lanes
# ---------------------------------------------------------------------------


def _fc(unit, lane, rank, signals=None):
    return CandidateV7(
        unit_id=unit,
        source_id=f"src-{unit}",
        revision=1,
        lane=lane,
        rank=rank,
        raw_score=1.0,
        signals=dict(signals or {}),
    )


class TestK60ReservedFacetSlots:
    def test_k60_reserve_facet_slots_whole_and_equal_shares(self):
        """§21.7 verbatim: share_q = floor(capv·whole_share); each facet
        gets floor((capv − share_q)/n_facets).  Whole keeps its share;
        each facet keeps its own — even though the facet items were
        appended AFTER a full whole list."""
        items = (
            [_fc(f"w{i}", "lex", i + 1) for i in range(6)]
            + [_fc(f"x{i}", "lex", i + 1, {"facet": 0}) for i in range(3)]
            + [_fc(f"y{i}", "lex", i + 1, {"facet": 1}) for i in range(3)]
        )
        kept, stats = reserve_facet_slots(items, 8, n_facets=2)
        ids = [c.unit_id for c in kept]
        # share_q = floor(8·0.5) = 4; share_f = floor(4/2) = 2
        assert stats["share_q"] == 4
        assert stats["share_f"] == 2
        assert ids[:4] == ["w0", "w1", "w2", "w3"]
        assert set(ids[4:6]) == {"x0", "x1"}
        assert set(ids[6:8]) == {"y0", "y1"}
        assert len(ids) == 8
        assert stats["whole"] == {"produced": 6, "kept": 4, "rolled_over": 0}
        assert stats["per_facet"]["0"] == {
            "produced": 3, "kept": 2, "rolled_over": 0}
        assert stats["per_facet"]["1"] == {
            "produced": 3, "kept": 2, "rolled_over": 0}
        assert stats["kept_total"] == 8

    def test_k60_reserve_facet_slots_rollover_in_facet_order(self):
        """A facet that under-produces rolls its unused slots over: the
        deterministic round-robin fills remaining capacity, whole query
        first, then facets in facet order — and the fill is REPORTED as
        rolled_over on the facet that consumed it."""
        items = (
            [_fc(f"w{i}", "lex", i + 1) for i in range(3)]
            + [_fc(f"x{i}", "lex", i + 1, {"facet": 0}) for i in range(4)]
            + [_fc("y0", "lex", 1, {"facet": 1})]  # under-produced facet
        )
        kept, stats = reserve_facet_slots(items, 8, n_facets=2)
        ids = [c.unit_id for c in kept]
        # share_q=4 → all 3 whole kept (under-produced);
        # share_f=floor(4/2)=2 → x0,x1 + y0; leftover 3 fills
        # round-robin: whole leftover ∅ → facet0 x2, facet1 ∅ → x3.
        assert ids == ["w0", "w1", "w2", "x0", "x1", "y0", "x2", "x3"]
        assert stats["whole"]["rolled_over"] == 0
        assert stats["per_facet"]["0"] == {
            "produced": 4, "kept": 4, "rolled_over": 2}
        assert stats["per_facet"]["1"] == {
            "produced": 1, "kept": 1, "rolled_over": 0}
        assert stats["kept_total"] == 8

    def test_k60_reserve_facet_slots_dedupes_across_runs(self):
        """The same unit surfacing in the whole run and a facet run is
        ONE unit — it consumes the whole share first, never a facet's
        reserved slot."""
        items = [_fc("w0", "lex", 1), _fc("w1", "lex", 2)]
        items += [
            _fc("w1", "lex", 1, {"facet": 0}),  # dupe of a kept whole unit
            _fc("x0", "lex", 2, {"facet": 0}),
            _fc("x1", "lex", 3, {"facet": 0}),
        ]
        kept, stats = reserve_facet_slots(items, 4, n_facets=1)
        ids = [c.unit_id for c in kept]
        # share_q=2 → w0,w1; share_f=2 → w1 skipped (already kept),
        # x0,x1 kept.
        assert ids == ["w0", "w1", "x0", "x1"]
        assert stats["per_facet"]["0"]["kept"] == 2

    def test_k60_reserve_facet_slots_validates_arms(self):
        items = [_fc("w0", "lex", 1)]
        with pytest.raises(ValueError):
            reserve_facet_slots(items, -1)
        with pytest.raises(ValueError):
            reserve_facet_slots(items, 4, whole_share=0.0)
        with pytest.raises(ValueError):
            reserve_facet_slots(items, 4, whole_share=1.5)
        with pytest.raises(ValueError):
            reserve_facet_slots(items, 4, n_facets=-1)

    def test_k60_rrf_fuse_limit_preserves_facet_units(self):
        """End to end at the fused cap: the lane output carries
        whole-query candidates first and facet-tagged candidates at the
        tail (the ``_run_lane`` merge shape), so every facet unit ranks
        below the cut on RRF — yet the fused ``limit`` keeps each
        facet's reserved share."""
        out_lex = LaneOutput(
            lane="lex",
            status=LaneStatus.OK,
            candidates=(
                [_fc(f"w{i}", "lex", i + 1) for i in range(8)]
                + [
                    _fc(f"x{i}", "lex", 9 + i, {"facet": 0})
                    for i in range(4)
                ]
            ),
        )
        fused = rrf_fuse([out_lex], limit=6)
        ids = [c.unit_id for c in fused]
        # share_q = floor(6·0.5)=3 whole; share_f = 3 for the one facet —
        # x0..x2 survive the cut despite RRF ranks 9–12.
        assert set(ids) == {"w0", "w1", "w2", "x0", "x1", "x2"}
        slots = fused.stats["facet_slots"]
        assert slots["whole"]["kept"] == 3
        assert slots["per_facet"]["0"]["kept"] == 3
        assert fused.stats["truncated"] is True

    def test_k60_rrf_fuse_facet_share_arm_off_truncates_plain(self):
        """``facets.whole_share`` = False (arm off) → the reservation is
        disabled; the fused cap is a plain RRF truncation and the
        tail-ranked facet units die — the D8-16 failure shape."""
        out_lex = LaneOutput(
            lane="lex",
            status=LaneStatus.OK,
            candidates=(
                [_fc(f"w{i}", "lex", i + 1) for i in range(8)]
                + [
                    _fc(f"x{i}", "lex", 9 + i, {"facet": 0})
                    for i in range(4)
                ]
            ),
        )
        fused = rrf_fuse(
            [out_lex], limit=6, facet_whole_share=False
        )
        ids = [c.unit_id for c in fused]
        assert set(ids) == {f"w{i}" for i in range(6)}
        assert "facet_slots" not in fused.stats

    def _seed_joint_corpus(self, conn):
        """One joint unit, a small rare-canon set, and a run of
        prolific-canon singles — enough that a small lane cap overflows
        the merge."""
        add_unit(conn, "u_joint")
        add_mention(conn, "u_joint", "alpha")
        add_mention(conn, "u_joint", "beta", bs=20, be=24)
        for i in range(3):
            add_unit(conn, f"b{i}")
            add_mention(conn, f"b{i}", "beta")
        for i in range(10):
            add_unit(conn, f"a{i:02d}")
            add_mention(conn, f"a{i:02d}", "alpha")
        set_df(conn, "alpha", 11)
        set_df(conn, "beta", 4)
        set_lex_stats(conn, 14)

    def test_k60_lane_merge_reserves_facet_share(self):
        """D8-16's exact site — ``pipeline._run_lane``: when the
        whole-query run fills the lane cap, facet-run candidates still
        keep a reserved share instead of dying under append-then-
        truncate."""
        register_lane(LaneName.ENT, lane_entity)
        conn = mk_conn()
        self._seed_joint_corpus(conn)
        ctx = mk_ctx(conn)
        whole = mk_qv(("alpha", "beta"), "alpha beta")
        facet0 = mk_qv(("alpha",), "alpha")
        out = pipeline_mod._run_lane(
            ctx,
            whole,
            LaneName.ENT,
            sl(cap=6),
            [facet0],
            frozenset({LaneName.ENT}),
            POOLS[BudgetClass.MID],
            time.monotonic,
            lambda: 10_000.0,
        )
        assert out.status is LaneStatus.OK
        tagged = [c for c in out.candidates if "facet" in c.signals]
        # The facet run produced real candidates and ≥ one survived the
        # capped merge with its reserved share.
        assert out.stats["facets"][0]["produced"] > 0
        assert tagged, "facet candidates died under a full whole run"
        facet_ids = {c.unit_id for c in tagged}
        whole_ids = {
            c.unit_id for c in out.candidates if "facet" not in c.signals
        }
        # reserved share: whole keeps share_q = floor(6·0.5) = 3
        assert len(whole_ids) <= 3
        assert facet_ids - whole_ids  # a facet-only unit made the cut

    def test_k60_lane_merge_reports_produced_kept_rolled_over(self):
        """Coverage reports per-facet produced/kept/rolled_over
        (V8-10.02) — the lane's facet stats carry the slot accounting,
        not just the produced count."""
        register_lane(LaneName.ENT, lane_entity)
        conn = mk_conn()
        self._seed_joint_corpus(conn)
        ctx = mk_ctx(conn)
        whole = mk_qv(("alpha", "beta"), "alpha beta")
        facet0 = mk_qv(("alpha",), "alpha")
        out = pipeline_mod._run_lane(
            ctx,
            whole,
            LaneName.ENT,
            sl(cap=6),
            [facet0],
            frozenset({LaneName.ENT}),
            POOLS[BudgetClass.MID],
            time.monotonic,
            lambda: 10_000.0,
        )
        entry = out.stats["facets"][0]
        assert entry["status"] == "ok"
        assert entry["produced"] > 0
        assert "kept" in entry and "rolled_over" in entry
        assert entry["kept"] == sum(
            1 for c in out.candidates if c.signals.get("facet") == 0
        )

    def test_k60_lane_merge_without_overflow_is_unchanged(self):
        """No cap pressure → no reordering: the merge keeps whole-run
        order then facet-appended order, exactly as before."""
        register_lane(LaneName.ENT, lane_entity)
        conn = mk_conn()
        add_unit(conn, "u1")
        add_mention(conn, "u1", "alpha")
        add_unit(conn, "u2")
        add_mention(conn, "u2", "beta")
        set_df(conn, "alpha", 1)
        set_df(conn, "beta", 1)
        ctx = mk_ctx(conn)
        whole = mk_qv(("alpha", "beta"), "alpha beta")
        facet0 = mk_qv(("alpha",), "alpha")
        out = pipeline_mod._run_lane(
            ctx,
            whole,
            LaneName.ENT,
            sl(cap=50),
            [facet0],
            frozenset({LaneName.ENT}),
            POOLS[BudgetClass.MID],
            time.monotonic,
            lambda: 10_000.0,
        )
        assert len(out.candidates) == 3  # 2 whole + 1 facet (u1 again)
        assert [c.rank for c in out.candidates] == [1, 2, 3]
        assert out.candidates[2].signals["facet"] == 0

    def test_k60_facet_lanes_lex_ent_only_by_default(self):
        """V8-10.06: the facet fan-out runs on lex+ent — the dense lane
        answers the whole query only (a facet pass must not multiply
        heavy lanes)."""
        assert pipeline_mod._FACET_LANES_V8 == frozenset(
            {LaneName.LEX, LaneName.ENT}
        )
        register_lane(LaneName.ENT, lane_entity)
        conn = mk_conn()
        self._seed_joint_corpus(conn)
        ctx = mk_ctx(conn)
        whole = mk_qv(("alpha", "beta"), "alpha beta")
        facet0 = mk_qv(("alpha",), "alpha")

        ent_out = pipeline_mod._run_lane(
            ctx,
            whole,
            LaneName.ENT,
            sl(),
            [facet0],
            pipeline_mod._FACET_LANES_V8,
            POOLS[BudgetClass.MID],
            time.monotonic,
            lambda: 10_000.0,
        )
        assert "facets" in ent_out.stats
        assert ent_out.stats["facets"][0]["status"] == "ok"

        # A lane outside the facet set runs the whole query only — no
        # facet fan-out, no facet stats, no facet-tagged candidates.
        dense_out = pipeline_mod._run_lane(
            ctx,
            whole,
            LaneName.DENSE,
            sl(),
            [facet0],
            pipeline_mod._FACET_LANES_V8,
            POOLS[BudgetClass.MID],
            time.monotonic,
            lambda: 10_000.0,
        )
        assert "facets" not in dense_out.stats
        assert all(
            "facet" not in c.signals for c in dense_out.candidates
        )

    def test_k60_dense_per_facet_arm_adds_dense_to_fanout(self):
        """The ``facets.dense_per_facet`` arm adds the dense lane to the
        fan-out set (§23).  The wiring in ``run_search`` reads the
        policy ``params`` map — exercise it through the same
        ``_policy_param`` channel the pipeline uses."""
        register_lane(LaneName.ENT, lane_entity)
        conn = mk_conn()
        self._seed_joint_corpus(conn)
        ctx = mk_ctx(conn, policy_params={"facets.dense_per_facet": True})
        # the run_search arm resolution: default set + dense when armed
        facet_lanes = set(pipeline_mod._FACET_LANES_V8)
        if pipeline_mod._policy_param(ctx.policy, "facets.dense_per_facet"):
            facet_lanes.add(LaneName.DENSE)
        facet_lanes = frozenset(facet_lanes)
        assert LaneName.DENSE in facet_lanes

        whole = mk_qv(("alpha", "beta"), "alpha beta")
        facet0 = mk_qv(("alpha",), "alpha")
        out = pipeline_mod._run_lane(
            ctx,
            whole,
            LaneName.DENSE,
            sl(),
            [facet0],
            facet_lanes,
            POOLS[BudgetClass.MID],
            time.monotonic,
            lambda: 10_000.0,
        )
        # dense is unregistered in this fixture → the lane honestly
        # reports unavailable, but the fan-out RAN: per-facet calls were
        # attempted and their statuses recorded.
        assert "facets" in out.stats
        assert out.stats["facets"][0]["status"] == "unavailable"


# ---------------------------------------------------------------------------
# K61 — V8-10.03 conjunctive entity retrieval (the joint-lane semantics:
# units mentioning ALL resolved canons emit first, signals.joint = true)
# ---------------------------------------------------------------------------


class TestK61ConjunctiveEntity:
    def test_k61_joint_unit_emits_first_and_marked(self):
        """Two resolved canons → the unit mentioning BOTH is emitted
        first with ``signals["joint"]=True`` — structural ordering, not
        score order (a rare-canon single can outscore it)."""
        conn = mk_conn()
        for i in range(7):
            add_unit(conn, f"f{i}")
        add_unit(conn, "u_joint")
        add_unit(conn, "u_a")
        add_unit(conn, "u_b")
        add_mention(conn, "u_joint", "alice")
        add_mention(conn, "u_joint", "bob", bs=20, be=23)
        add_mention(conn, "u_a", "alice")
        add_mention(conn, "u_a", "ally", bs=20, be=24)  # alias of alice
        add_mention(conn, "u_b", "bob")
        for i in range(6):
            add_mention(conn, f"f{i}", "bob")
        set_df(conn, "alice", 2)
        set_df(conn, "ally", 1)
        set_df(conn, "bob", 8)
        conn.execute(
            "INSERT OR REPLACE INTO entity_aliases_v7"
            " (scope_id, canon, alias_canon, generation, rule_id,"
            "  evidence_count, method, state)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (SCOPE, "alice", "ally", GEN, "A5", 1, "caller", "active"),
        )
        out = lane_entity(
            mk_ctx(conn), mk_qv(("alice", "bob")), sl()
        )
        ids = by_id(out)
        assert out.candidates[0].unit_id == "u_joint"
        assert ids["u_joint"].signals["joint"] is True
        assert "joint" not in ids["u_a"].signals
        assert "joint" not in ids["u_b"].signals
        # u_a genuinely outscores u_joint (two rare matched canons beat
        # rare + capped-dominant) — joint ordering is structural, not a
        # score artifact.  A canon + its own alias still covers ONE
        # query canon, so u_a is not joint.
        assert ids["u_a"].raw_score > ids["u_joint"].raw_score
        assert ids["u_a"].signals["query_canons_covered"] == 1
        assert out.stats["joint"] == 1

    def test_k61_partial_coverage_is_not_joint(self):
        """Covering 2 of 3 resolved canons is a boosted single — only
        all-canons-present marks joint."""
        conn = mk_conn()
        for u, cs in (
            ("u_abc", ("a", "b", "c")),
            ("u_ab", ("a", "b")),
            ("u_a", ("a",)),
        ):
            add_unit(conn, u)
            for j, c in enumerate(cs):
                add_mention(conn, u, c, bs=10 * j, be=10 * j + 4)
        set_df(conn, "a", 3)
        set_df(conn, "b", 2)
        set_df(conn, "c", 1)
        out = lane_entity(mk_ctx(conn), mk_qv(("a", "b", "c")), sl())
        ids = by_id(out)
        assert out.candidates[0].unit_id == "u_abc"
        assert ids["u_abc"].signals["joint"] is True
        assert ids["u_ab"].signals["query_canons_covered"] == 2
        assert "joint" not in ids["u_ab"].signals

    def test_k61_joint_respects_eligibility_and_fence(self):
        """The joint set comes out of the same fenced, eligible pool —
        a held or post-fence joint unit is never emitted."""
        conn = mk_conn()
        for u in ("j1", "j2", "a1"):
            add_unit(conn, u)
        for u in ("j1", "j2"):
            add_mention(conn, u, "a")
            add_mention(conn, u, "b", bs=20, be=24)
        add_mention(conn, "a1", "a")
        out = lane_entity(
            mk_ctx(conn, eligible={"j2", "a1"}), mk_qv(("a", "b")), sl()
        )
        ids = [c.unit_id for c in out.candidates]
        assert ids[0] == "j2"
        assert by_id(out)["j2"].signals["joint"] is True
        assert "j1" not in ids
        assert out.stats["ineligible"] == 1

        # generation fence: the completing mention beyond the fence
        # leaves the unit non-joint
        conn2 = mk_conn()
        add_unit(conn2, "u_x")
        add_unit(conn2, "u_y")
        add_mention(conn2, "u_x", "a", gen=1)
        add_mention(conn2, "u_x", "b", gen=9, bs=20, be=24)
        add_mention(conn2, "u_y", "a", gen=1)
        add_mention(conn2, "u_y", "b", gen=1, bs=20, be=24)
        out3 = lane_entity(
            mk_ctx(conn2, gen=3), mk_qv(("a", "b")), sl()
        )
        ids3 = by_id(out3)
        assert out3.candidates[0].unit_id == "u_y"
        assert ids3["u_y"].signals["joint"] is True
        assert "joint" not in ids3["u_x"].signals


# ---------------------------------------------------------------------------
# K62 — V8-10.04 per-canon quota + V8-10.05 group max-pool arm
# ---------------------------------------------------------------------------


class TestK62PerCanonQuota:
    def test_k62_prolific_canon_bounded_by_quota(self):
        """A canon with df = 1000 contributes at most its quota (prior
        50) of non-joint emissions — it cannot flood the lane cap and
        starve the rare canon."""
        conn = mk_conn()
        set_lex_stats(conn, 2000)
        add_unit(conn, "u_j")
        add_mention(conn, "u_j", "prolific")
        add_mention(conn, "u_j", "rare", bs=10, be=14)
        for i in range(60):
            add_unit(conn, f"p{i:03d}")
            add_mention(conn, f"p{i:03d}", "prolific")
        for i in range(3):
            add_unit(conn, f"r{i}")
            add_mention(conn, f"r{i}", "rare")
        set_df(conn, "prolific", 1000)  # the scenario's df
        set_df(conn, "rare", 4)
        out = lane_entity(
            mk_ctx(conn), mk_qv(("prolific", "rare")), sl(cap=200)
        )
        ids = by_id(out)
        assert out.candidates[0].unit_id == "u_j"
        assert ids["u_j"].signals["joint"] is True
        prolific = [u for u in ids if u.startswith("p")]
        rare = [u for u in ids if u.startswith("r")]
        assert len(prolific) == 50  # capped at the §23 prior
        assert len(rare) == 3  # rare canon unflooded
        assert out.stats["per_canon_quota"] == 50
        assert out.stats["quota_dropped"] == 10
        assert out.stats["per_canon"]["prolific"] == {
            "emitted": 50,
            "dropped": 10,
        }
        assert out.stats["per_canon"]["rare"] == {
            "emitted": 3,
            "dropped": 0,
        }

    def test_k62_quota_arm_overrides_and_validates(self):
        """``ent.per_canon_quota`` resolves off policy params then the
        manifest; ``0`` disables single-canon emission (joints still
        emit); a mistyped arm raises VALIDATION loudly."""
        conn = mk_conn()
        add_unit(conn, "j0")
        add_mention(conn, "j0", "a")
        add_mention(conn, "j0", "b", bs=9, be=12)
        for i in range(4):
            add_unit(conn, f"a{i}")
            add_mention(conn, f"a{i}", "a")
        for i in range(4):
            add_unit(conn, f"b{i}")
            add_mention(conn, f"b{i}", "b")
        set_df(conn, "a", 5)
        set_df(conn, "b", 5)

        # policy params channel wins over the manifest channel
        ctx = mk_ctx(
            conn,
            manifest={ent_mod.PER_CANON_QUOTA_ARM: 3},
            policy_params={ent_mod.PER_CANON_QUOTA_ARM: 1},
        )
        out = lane_entity(ctx, mk_qv(("a", "b")), sl())
        assert out.stats["per_canon_quota"] == 1
        singles = [
            c for c in out.candidates if not c.signals.get("joint")
        ]
        assert len(singles) == 2  # 1 per canon
        assert by_id(out)["j0"].signals["joint"] is True

        # manifest channel alone
        ctx2 = mk_ctx(conn, manifest={ent_mod.PER_CANON_QUOTA_ARM: 0})
        out2 = lane_entity(ctx2, mk_qv(("a", "b")), sl())
        assert [c.unit_id for c in out2.candidates] == ["j0"]
        assert out2.stats["per_canon_quota"] == 0

        for bad in ("lots", -1, 1.5, True):
            ctxb = mk_ctx(conn, manifest={ent_mod.PER_CANON_QUOTA_ARM: bad})
            with pytest.raises(VerbatimError):
                lane_entity(ctxb, mk_qv(("a", "b")), sl())


class TestK62GroupMaxPool:
    def test_k62_group_maxpool_arm_clusters_sessions(self):
        """V8-10.05 armed: ``assemble_pack(group_maxpool=True)`` clusters
        atomic units sharing a session into one pack unit positioned by
        its best member — the real ``_session_maxpool`` path — and the
        result records the applied arm state honestly."""
        from verbatim.retrieval.v7 import pack as pack_mod

        def item(uid, session=None):
            return PackItemV7(
                ref=f"u:{uid}",
                unit_id=uid,
                quote=f"text of {uid}".encode(),
                session={"id": session} if session else None,
            )

        scored = [
            item("s1a", "sess-1"),
            item("x0"),               # session-less — own pack unit
            item("s1b", "sess-1"),
            item("s1c", "sess-1"),
            item("s2a", "sess-2"),
        ]
        res_on = pack_mod.assemble_pack(
            scored, mk_qv(), group_maxpool=True
        )
        res_off = pack_mod.assemble_pack(
            scored, mk_qv(), group_maxpool=False
        )
        assert res_on.group_maxpool is True
        assert res_off.group_maxpool is False
        # clusters order by best member position — sess-1's best is
        # pos 0, session-less x0 is pos 1, sess-2's best is pos 4 →
        # sess-1 members expand in rank order, then x0, then sess-2.
        ids_on = [it.unit_id for it in res_on.items]
        assert ids_on == ["s1a", "s1b", "s1c", "x0", "s2a"]
        # off: input order = rank order, no clustering
        ids_off = [it.unit_id for it in res_off.items]
        assert ids_off == ["s1a", "x0", "s1b", "s1c", "s2a"]

    def test_k62_group_maxpool_arm_reads_ctx_flag(self):
        """The §23 arm channel: ``pack.group_maxpool`` resolves off the
        lane-context manifest flag; absent → default off (the arm was
        measured and rejected — see decision record)."""
        from verbatim.retrieval.v7 import pack as pack_mod

        conn = mk_conn()
        ctx = mk_ctx(conn, manifest={"pack.group_maxpool": True})
        item = PackItemV7(
            ref="u:a", unit_id="a", quote=b"hello", session={"id": "s"}
        )
        res = pack_mod.assemble_pack([item], mk_qv(), ctx=ctx)
        assert res.group_maxpool is True
        res2 = pack_mod.assemble_pack([item], mk_qv())
        assert res2.group_maxpool is False

    def test_k62_group_maxpool_arm_has_decision_record(self):
        """Scenario wording: "the group max-pool arm has a decision
        record".  The V8 arm register is the project's decision-record
        surface — the ``pack.group_maxpool`` row must exist and carry a
        measured verdict (it measured catastrophic on multi-hop
        all@10 and was REJECTED → the arm ships default-off)."""
        assert ARM_REGISTER.is_file(), (
            f"V8 arm register missing: {ARM_REGISTER}"
        )
        text = ARM_REGISTER.read_text(encoding="utf-8")
        row = next(
            (
                ln
                for ln in text.splitlines()
                if "pack.group_maxpool" in ln and "|" in ln
            ),
            None,
        )
        assert row is not None, (
            "no decision record for pack.group_maxpool in the arm register"
        )
        assert "REJECT" in row, (
            f"decision record has no verdict: {row.strip()}"
        )
        # consistent with the shipped default: the arm is off unless
        # explicitly armed
        from verbatim.retrieval.v7 import pack as pack_mod

        item = PackItemV7(ref="u:a", unit_id="a", quote=b"x")
        res = pack_mod.assemble_pack([item], mk_qv())
        assert res.group_maxpool is False
