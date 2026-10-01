"""V8 §23 declared-arm coverage (V8-23.02) — every declared flag must be
read and change behavior, or degrade honestly. Six arms under test:

- ``dense.B_max`` — the dense-compaction trigger
  (``retrieval/v7/dense_compact.resolve_b_max`` → ``handle_dense_compact``).
- ``dense.embed_batch`` — the coalesced source-embed bound
  (``512 rows / 250 ms queue age`` prior), resolved by
  ``dense_compact.embed_batch_bound`` and applied inside
  ``jobs/source_jobs._stage_embed_siblings`` / ``_prepare_unit_vectors``.
- ``dense.tier`` — the profile→encoder-tier pin (V8-08.05), resolved by
  ``dense_compact.resolve_dense_tier`` and gated on the write path by
  ``source_jobs`` (an unprovisioned pin defers honestly). The read-side
  encoder selection lives in ``retrieval/v7/dense.py`` — an ownership
  handoff outside this change's file scope.
- ``fusion.dense_form`` — ``rrf`` (default) / ``combsum`` research form
  (V8-08.06), resolved inside ``fusion.rrf_fuse`` via the explicit
  ``dense_form``/``dense_form_alpha`` kwargs or the ``policy`` carrier
  (``GatedPolicyV7.params``). The pipeline's ``fusion_kwargs`` loop does
  not yet forward ``ctx.policy`` — a one-tuple handoff in
  ``pipeline.py`` (outside this change's file scope); the arm itself is
  fully read, validated, and applied at the fusion boundary.
- ``context.ctx_field_weight`` — BM25F weight of the optional indexed
  ``ctx`` field (V8-06.03), resolved in ``lexical._lexical_arms`` and
  applied in ``_score_candidates``. The deriver that would populate a
  ``ctx`` column is outside scope; an armed arm on a schema without the
  column is reported (``stats["ctx_field_weight"]["indexed"] is
  False``), never fabricated.
- ``rerank.speaker_match_weight`` — the S4 ``speaker_match`` coefficient
  (V8-11.02), resolved in ``rerank_features.score_candidates`` off
  ``ctx.policy``; explicit caller ``weights`` keep precedence.
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
import threading
import types as _types
from dataclasses import replace

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.time import now_us
from verbatim.core.types import (
    ErrorCode,
    JobKind,
    Provenance,
    Scope,
    SourceKind,
    VerbatimError,
)
from verbatim.core.types_v7 import (
    BudgetClass,
    CandidateV7,
    FusedCandidate,
    LaneContextV7,
    LaneName,
    LaneStatus,
    LaneSlice,
    LaneOutput,
    NormAnalysis,
    NormTerm,
    QueryViewV7,
    RetrievalPolicyV7,
    IntentClass,
    IntentResult,
)
from verbatim.embeddings import matrix as mx
from verbatim.embeddings.hashing import HashingEncoder
from verbatim.ingest import Ingester, SourceEnvelope
from verbatim.jobs import source_jobs as sj
from verbatim.readiness import ingest_receipt_id
from verbatim.retrieval.v7 import dense_compact as dc
from verbatim.retrieval.v7.fusion import (
    DEFAULT_DENSE_FORM_ALPHA,
    rrf_fuse,
    score_detail,
)
from verbatim.retrieval.v7.lexical import (
    CTX_FIELD_W_DEFAULT,
    lane_lexical,
)
from verbatim.retrieval.v7.policy import GatedPolicyV7
from verbatim.retrieval.v7.rerank_features import (
    FEATURE_WEIGHTS_V1,
    SPEAKER_MATCH_WEIGHT_DEFAULT,
    score_candidates,
)
from verbatim.storage.schema import DDL_V1
from verbatim.storage.schema_v7 import ensure_v7_additive
from verbatim.storage.store import Store


SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")
SID = "scope-a"
ENC = "test:enc:v1"


def _policy(**params) -> GatedPolicyV7:
    return GatedPolicyV7(
        policy_id="retrieval_policy/v7",
        profile="test",
        lanes=(LaneName.LEX, LaneName.DENSE),
        lane_weights={},
        params=params,
    )


# ---------------------------------------------------------------------------
# dense.B_max (V8-08.01) — resolve_b_max + handle_dense_compact
# ---------------------------------------------------------------------------


class _TxStore:
    """tx()/read() shim over one explicit-transaction in-memory conn."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()
        self._event_us = 1_000_000

    @contextlib.contextmanager
    def read(self):
        yield self._conn

    @contextlib.contextmanager
    def tx(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def next_event_us(self) -> int:
        self._event_us += 1
        return self._event_us


_ORACLE_DDL = """
CREATE TABLE unit_vectors (
    unit_key TEXT NOT NULL,
    encoder_id TEXT NOT NULL,
    dims INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (unit_key, encoder_id)
);
"""


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(DDL_V1)
    ensure_v7_additive(c)
    c.executescript(_ORACLE_DDL)
    yield c
    c.close()


@pytest.fixture()
def txstore(conn):
    return _TxStore(conn)


def _vec(seed: int, dims: int = 16) -> list[float]:
    import random

    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dims)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _fat_blocks(conn, n_blocks: int, rows: int = 300, *,
                scope: str = SID, gen: int = 1) -> None:
    """Healthy-density blocks (mean ≥ MIN_MEAN_ROWS) — only a ``B_max``
    breach can trigger compaction, isolating the arm under test."""
    for bno in range(n_blocks):
        mx.write_block(
            conn, ENC, scope, gen, bno,
            [(f"u{bno}-{i:03d}", _vec(bno * 1000 + i)) for i in range(rows)],
        )


def _bare_ingester(store) -> _types.SimpleNamespace:
    return _types.SimpleNamespace(store=store)


class TestDenseBMax:
    """``dense.B_max`` — reader + trigger behavior."""

    def test_default_prior(self):
        assert dc.resolve_b_max({}, _bare_ingester(None)) == 64
        assert dc.resolve_b_max(None, _bare_ingester(None)) == 64

    def test_refs_dotted_and_legacy(self):
        ing = _bare_ingester(None)
        assert dc.resolve_b_max({"dense.B_max": 3}, ing) == 3
        assert dc.resolve_b_max({"dense_b_max": 4}, ing) == 4
        assert dc.resolve_b_max({"b_max": 5}, ing) == 5

    def test_zero_is_a_real_value(self):
        """``0`` must not collapse into the prior — a declared zero arms
        "compact whenever any reducible block exists"."""
        ing = _bare_ingester(None)
        assert dc.resolve_b_max({"b_max": 0}, ing) == 0
        assert dc.resolve_b_max({"dense.B_max": 0}, ing) == 0

    def test_params_carrier_inside_refs(self):
        ing = _bare_ingester(None)
        refs = {"params": {"dense.B_max": 7}}
        assert dc.resolve_b_max(refs, ing) == 7

    def test_policy_carrier_on_ingester(self):
        ing = _types.SimpleNamespace(
            store=None,
            retrieval_policy=_policy(**{"dense.B_max": 9}),
        )
        assert dc.resolve_b_max({}, ing) == 9

    def test_cfg_jobs_carrier(self):
        c = VerbatimConfig()
        cfg = replace(
            c, jobs=replace(c.jobs, dense_compact_b_max=33)
        )
        ing = _types.SimpleNamespace(store=None, cfg=cfg)
        assert dc.resolve_b_max({}, ing) == 33
        # …but a job-level declaration still outranks the deployment.
        assert dc.resolve_b_max({"dense.B_max": 2}, ing) == 2

    def test_precedence_refs_over_policy(self):
        ing = _types.SimpleNamespace(
            store=None,
            retrieval_policy=_policy(**{"dense.B_max": 9}),
        )
        assert dc.resolve_b_max({"dense.B_max": 2}, ing) == 2

    @pytest.mark.parametrize("bad", [-1, "x", 2.5, True, [3]])
    def test_malformed_raises_validation(self, bad):
        with pytest.raises(VerbatimError) as ei:
            dc.resolve_b_max({"dense.B_max": bad}, _bare_ingester(None))
        assert ei.value.code == ErrorCode.VALIDATION

    def test_compaction_trigger_uses_arm(self, conn, txstore):
        """3 fat blocks: default 64 → untouched; ``dense.B_max: 2`` in
        job refs → the handler compacts (the flag changes behavior)."""
        _fat_blocks(conn, 3)
        ing = _bare_ingester(txstore)
        dc.handle_dense_compact(
            {"job_id": "j-a", "scope_id": SID,
             "input_refs": {"scope_id": SID, "dense.B_max": 2}},
            "w", ing,
        )
        assert len(list(mx.iter_blocks(conn, ENC, SID, 1))) == 1
        assert dc.compaction_epoch(conn, ENC, SID, 1) == 1

    def test_compaction_default_leaves_healthy_space(self, conn, txstore):
        _fat_blocks(conn, 3)
        ing = _bare_ingester(txstore)
        dc.handle_dense_compact(
            {"job_id": "j-b", "scope_id": SID,
             "input_refs": {"scope_id": SID}},
            "w", ing,
        )
        assert len(list(mx.iter_blocks(conn, ENC, SID, 1))) == 3
        assert dc.compaction_epoch(conn, ENC, SID, 1) == 0

    def test_zero_compacts_any_reducible_space(self, conn, txstore):
        _fat_blocks(conn, 2)
        ing = _bare_ingester(txstore)
        dc.handle_dense_compact(
            {"job_id": "j-c", "scope_id": SID,
             "input_refs": {"scope_id": SID, "dense.B_max": 0}},
            "w", ing,
        )
        assert len(list(mx.iter_blocks(conn, ENC, SID, 1))) == 1


# ---------------------------------------------------------------------------
# dense.embed_batch (V8-08.02) — embed_batch_bound resolution
# ---------------------------------------------------------------------------


class TestEmbedBatchBound:
    def test_default_prior(self):
        b = dc.embed_batch_bound({}, _bare_ingester(None))
        assert (b.rows, b.max_age_ms) == (512, 250.0)

    def test_cfg_defaults_feed_unresolved(self):
        ing = _types.SimpleNamespace(store=None, cfg=VerbatimConfig())
        b = dc.embed_batch_bound({}, ing)
        assert (b.rows, b.max_age_ms) == (512, 250.0)

    def test_scalar_is_row_bound(self):
        b = dc.embed_batch_bound(
            {"dense.embed_batch": 32}, _bare_ingester(None))
        assert b.rows == 32 and b.max_age_ms == 250.0

    def test_pair_form(self):
        b = dc.embed_batch_bound(
            {"dense.embed_batch": [64, 100.0]}, _bare_ingester(None))
        assert b.rows == 64 and b.max_age_ms == 100.0

    def test_mapping_form(self):
        b = dc.embed_batch_bound(
            {"dense.embed_batch": {"rows": 100, "max_age_ms": 50}},
            _bare_ingester(None),
        )
        assert b.rows == 100 and b.max_age_ms == 50.0

    def test_off_disarms(self):
        for off in (False, "off", "none", "disabled"):
            b = dc.embed_batch_bound(
                {"dense.embed_batch": off}, _bare_ingester(None))
            assert (b.rows, b.max_age_ms) == (0, 0.0), off

    def test_split_keys_override_components(self):
        refs = {
            "dense.embed_batch.rows": 96,
            "dense.embed_batch.max_age_ms": 75.0,
        }
        b = dc.embed_batch_bound(refs, _bare_ingester(None))
        assert (b.rows, b.max_age_ms) == (96, 75.0)

    def test_policy_carrier(self):
        ing = _types.SimpleNamespace(
            store=None,
            retrieval_policy=_policy(
                **{"dense.embed_batch": {"rows": 48, "max_age_ms": 90}}
            ),
        )
        b = dc.embed_batch_bound({}, ing)
        assert (b.rows, b.max_age_ms) == (48, 90.0)

    def test_cfg_carrier_components(self):
        c = VerbatimConfig()
        cfg = replace(
            c,
            jobs=replace(
                c.jobs,
                dense_embed_batch_rows=128,
                dense_embed_batch_max_age_ms=500.0,
            ),
        )
        b = dc.embed_batch_bound({}, _types.SimpleNamespace(
            store=None, cfg=cfg))
        assert (b.rows, b.max_age_ms) == (128, 500.0)

    @pytest.mark.parametrize(
        "bad", ["bogus", [1], [1, 2, 3], {"rows": "x"}, -4, {"rows": -1}]
    )
    def test_malformed_raises_validation(self, bad):
        with pytest.raises(VerbatimError) as ei:
            dc.embed_batch_bound(
                {"dense.embed_batch": bad}, _bare_ingester(None))
        assert ei.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# dense.tier (V8-08.05) — resolution + satisfaction
# ---------------------------------------------------------------------------


class TestDenseTier:
    def test_absent_and_auto_are_unconstrained(self):
        ing = _bare_ingester(None)
        assert dc.resolve_dense_tier({}, ing) is None
        for word in ("auto", "default", "inherit"):
            assert dc.resolve_dense_tier({"dense.tier": word}, ing) is None

    def test_profile_aliases(self):
        ing = _bare_ingester(None)
        assert dc.resolve_dense_tier(
            {"dense.tier": "local_memory"}, ing) == "hashing"
        assert dc.resolve_dense_tier(
            {"dense.tier": "local_memory_quality"}, ing) == "potion-base-8m"
        # Literal candidate spelling normalizes case.
        assert dc.resolve_dense_tier(
            {"dense.tier": "Potion-Base-8M"}, ing) == "potion-base-8m"

    def test_params_carrier(self):
        ing = _types.SimpleNamespace(
            store=None,
            retrieval_policy=_policy(**{"dense.tier": "local_memory"}),
        )
        assert dc.resolve_dense_tier({}, ing) == "hashing"

    @pytest.mark.parametrize("bad", [42, "", "   ", ["hashing"], True])
    def test_malformed_raises_validation(self, bad):
        with pytest.raises(VerbatimError) as ei:
            dc.resolve_dense_tier({"dense.tier": bad}, _bare_ingester(None))
        assert ei.value.code == ErrorCode.VALIDATION

    def test_satisfaction(self):
        # Backend-family match.
        assert dc.dense_tier_satisfied(
            "hashing", "hashing:subword-ngram:v1", "hashing")
        # Encoder-id segment match (an artifact-family pin).
        assert dc.dense_tier_satisfied(
            "potion-base-8m", "potion-base-8m:r3", "artifact")
        # Honest mismatch — never a silent downgrade.
        assert not dc.dense_tier_satisfied(
            "potion-base-8m", "hashing:subword-ngram:v1", "hashing")
        assert not dc.dense_tier_satisfied("hashing", None, None)


# ---------------------------------------------------------------------------
# fusion.dense_form (V8-08.06) — rrf default + combsum research arm
# ---------------------------------------------------------------------------


def _cand(unit, lane, rank, *, raw=1.0, source=None, rev=1, signals=None):
    return CandidateV7(
        unit_id=unit,
        source_id=source or f"src-{unit}",
        revision=rev,
        lane=lane,
        rank=rank,
        raw_score=raw,
        signals=signals or {},
    )


def _lane(name, cands):
    return LaneOutput(lane=name, status=LaneStatus.OK,
                      candidates=list(cands))


class TestFusionDenseForm:
    def test_default_rrf_unchanged(self):
        outs = [
            _lane("lex", [_cand("a", "lex", 1, raw=10.0),
                          _cand("b", "lex", 2, raw=5.0)]),
            _lane("dense", [_cand("b", "dense", 1, raw=0.5),
                            _cand("c", "dense", 2, raw=0.4)]),
        ]
        out = rrf_fuse(outs, {})
        assert [c.unit_id for c in out] == ["b", "a", "c"]
        assert math.isclose(out[0].rrf, 1 / 61 + 1 / 62, rel_tol=1e-15)
        assert out.stats["dense_form"]["form"] == "rrf"
        assert out.stats["dense_form"]["applied"] == 0

    def test_combsum_score_space(self):
        """lex → bm25_norm (lane-max normalized), dense → α·cos; other
        lanes stay rank-space."""
        outs = [
            _lane("lex", [_cand("a", "lex", 1, raw=10.0),
                          _cand("b", "lex", 2, raw=5.0)]),
            _lane("dense", [_cand("a", "dense", 1, raw=0.5),
                            _cand("c", "dense", 2, raw=0.25)]),
            _lane("graph", [_cand("a", "graph", 1, raw=7.0)]),
        ]
        fused = rrf_fuse(outs, {}, dense_form="combsum")
        by_id = {c.unit_id: c for c in fused}
        alpha = DEFAULT_DENSE_FORM_ALPHA
        # a: bm25_norm=10/10=1.0 + α·0.5 + graph rank term 1/61
        assert math.isclose(
            by_id["a"].rrf, 1.0 + alpha * 0.5 + 1 / 61, rel_tol=1e-12)
        # b: bm25_norm=5/10=0.5 only (no dense/graph row)
        assert math.isclose(by_id["b"].rrf, 0.5, rel_tol=1e-12)
        # c: α·0.25 only
        assert math.isclose(by_id["c"].rrf, alpha * 0.25, rel_tol=1e-12)
        st = fused.stats["dense_form"]
        assert st["form"] == "combsum" and st["lex_max"] == 10.0
        assert st["applied"] == 4  # a(lex+dense), b(lex), c(dense)

    def test_policy_carrier_reads_params(self):
        """The arm resolves off ``policy.params`` — the §23 register."""
        outs = [
            _lane("lex", [_cand("a", "lex", 1, raw=10.0)]),
            _lane("dense", [_cand("a", "dense", 1, raw=0.5)]),
        ]
        fused = rrf_fuse(
            outs, {}, policy=_policy(**{"fusion.dense_form": "combsum"}))
        assert fused.stats["dense_form"]["form"] == "combsum"
        assert math.isclose(
            fused[0].rrf, 1.0 + DEFAULT_DENSE_FORM_ALPHA * 0.5,
            rel_tol=1e-12)

    def test_explicit_kwarg_overrides_policy(self):
        outs = [
            _lane("lex", [_cand("a", "lex", 1, raw=10.0)]),
            _lane("dense", [_cand("a", "dense", 1, raw=0.5)]),
        ]
        fused = rrf_fuse(
            outs, {}, dense_form="rrf",
            policy=_policy(**{"fusion.dense_form": "combsum"}))
        assert fused.stats["dense_form"]["form"] == "rrf"
        assert math.isclose(fused[0].rrf, 1 / 61 + 1 / 61, rel_tol=1e-15)

    def test_alpha_arm(self):
        outs = [
            _lane("dense", [_cand("a", "dense", 1, raw=0.5)]),
        ]
        fused = rrf_fuse(
            outs, {}, dense_form="combsum", dense_form_alpha=2.0)
        assert math.isclose(fused[0].rrf, 1.0, rel_tol=1e-12)
        assert fused[0].signals["dense_form_alpha"] == 2.0
        fused2 = rrf_fuse(
            outs, {},
            policy=_policy(**{
                "fusion.dense_form": "combsum",
                "fusion.dense_form_alpha": 0.5,
            }))
        assert math.isclose(fused2[0].rrf, 0.25, rel_tol=1e-12)

    def test_lane_weights_never_multiply_raw_scores(self):
        """V75-03.03: under combsum a lane weight still applies only to
        rank-space terms — the score-space terms are unweighted."""
        outs = [
            _lane("lex", [_cand("a", "lex", 1, raw=10.0)]),
            _lane("dense", [_cand("a", "dense", 1, raw=0.5)]),
        ]
        fused = rrf_fuse(
            outs, {"lex": 3.0, "dense": 3.0}, dense_form="combsum")
        # Weights are NOT multipliers here: same score as unweighted.
        assert math.isclose(
            fused[0].rrf, 1.0 + DEFAULT_DENSE_FORM_ALPHA * 0.5,
            rel_tol=1e-12)

    def test_provenance_signals(self):
        outs = [
            _lane("lex", [_cand("a", "lex", 1, raw=8.0)]),
            _lane("dense", [_cand("a", "dense", 1, raw=0.5)]),
        ]
        fused = rrf_fuse(outs, {}, dense_form="combsum",
                         dense_form_alpha=1.5)
        sig = fused[0].signals
        assert sig["dense_form"] == "combsum"
        assert sig["dense_form_alpha"] == 1.5
        assert sig["dense_form_lex_max"] == 8.0
        assert sig["bm25_norm"] == 1.0
        assert sig["dense_cos"] == 0.5

    def test_score_detail_reconstructs(self):
        outs = [
            _lane("lex", [_cand("a", "lex", 1, raw=10.0),
                          _cand("b", "lex", 2, raw=4.0)]),
            _lane("dense", [_cand("b", "dense", 1, raw=0.5)]),
            _lane("graph", [_cand("b", "graph", 1, raw=2.0)]),
        ]
        fused = rrf_fuse(outs, {}, dense_form="combsum")
        b = next(c for c in fused if c.unit_id == "b")
        detail = score_detail(b, {})
        total = sum(detail["contributions"].values())
        for bonus in ("context_bonus", "facet_bonus"):
            v = detail.get(bonus) or 0.0
            total += v
        assert math.isclose(total, b.rrf, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(
            detail["contributions"]["lex"], 0.4, rel_tol=1e-12)
        assert math.isclose(
            detail["contributions"]["dense"],
            DEFAULT_DENSE_FORM_ALPHA * 0.5, rel_tol=1e-12)
        assert math.isclose(
            detail["contributions"]["graph"], 1 / 61, rel_tol=1e-12)
        assert detail["dense_form"]["form"] == "combsum"

    @pytest.mark.parametrize("bad", ["bogus", 42, {"form": "nope"}])
    def test_malformed_form_raises(self, bad):
        with pytest.raises(ValueError):
            rrf_fuse([], {}, dense_form=bad)

    @pytest.mark.parametrize("bad_alpha", ["x", -1.0, 0.0, float("nan")])
    def test_malformed_alpha_raises(self, bad_alpha):
        with pytest.raises(ValueError):
            rrf_fuse([], {}, dense_form="combsum",
                     dense_form_alpha=bad_alpha)

    def test_combsum_zero_lex_max_is_honest(self):
        """A lex lane with no finite raws contributes a true zero —
        never a fabricated normalizer."""
        outs = [
            _lane("lex", [_cand("a", "lex", 1, raw=None)]),
            _lane("dense", [_cand("a", "dense", 1, raw=0.5)]),
        ]
        fused = rrf_fuse(outs, {}, dense_form="combsum")
        assert fused.stats["dense_form"]["lex_max"] == 0.0
        assert math.isclose(
            fused[0].rrf, 0.0 + DEFAULT_DENSE_FORM_ALPHA * 0.5,
            rel_tol=1e-12)


# ---------------------------------------------------------------------------
# context.ctx_field_weight (V8-06.03) — optional indexed ctx field
# ---------------------------------------------------------------------------


_CTX_DDL = """
CREATE TABLE units(
  unit_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, revision INTEGER NOT NULL,
  scope_id TEXT NOT NULL, kind TEXT NOT NULL, parent_unit_id TEXT,
  session_id TEXT, seq INTEGER, speaker_canon TEXT, perspective TEXT,
  recorded_at_us INTEGER, occurred_start_us INTEGER, occurred_end_us INTEGER,
  occurred_precision TEXT, occurred_source TEXT, byte_start INTEGER,
  byte_end INTEGER, generation INTEGER NOT NULL);
CREATE VIRTUAL TABLE unit_fts USING fts5(
  text, speaker, entities, session, "when", ctx,
  tokenize='unicode61 remove_diacritics 2');
"""

_PLAIN_DDL = """
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
"""


def _lex_ctx(conn, *, manifest=None, params=None):
    policy = (
        _policy(**params)
        if params is not None
        else RetrievalPolicyV7(
            policy_id="retrieval_policy/v7", profile="test",
            lanes=(LaneName.LEX,), lane_weights={})
    )
    return LaneContextV7(
        store=conn, scope_id="s", generation=1, eligible=None,
        query_time_us=0, profile="test", budget=BudgetClass.MID,
        policy=policy, manifest=dict(manifest or {}))


def _lex_qv(query):
    terms = tuple(
        NormTerm(term=t, channel="text", byte_start=0, byte_end=len(t))
        for t in query.lower().split()
    )
    return QueryViewV7(
        query=query,
        norm=NormAnalysis(analyzer_id="norm/v2", terms=terms,
                          identifiers=(), text=query.lower()),
        intent=IntentResult(primary=IntentClass.LOOKUP,
                            classes=(IntentClass.LOOKUP,)),
    )


def _lex_slice(cap=50):
    return LaneSlice(deadline_ms=60_000.0, cap=cap)


def _seed_ctx_unit(conn, unit_id, *, text, ctx="", scope="s", gen=1):
    """One unit row + an FTS row whose ``ctx`` column carries neighbor
    text — the V8-06.03 indexed-field shape (``ctx`` DDL present)."""
    cur = conn.execute(
        "INSERT INTO units(unit_id, source_id, revision, scope_id, kind,"
        " parent_unit_id, session_id, seq, speaker_canon, perspective,"
        " recorded_at_us, occurred_start_us, occurred_end_us,"
        " occurred_precision, occurred_source, byte_start, byte_end,"
        " generation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (unit_id, unit_id, 1, scope, "turn", None, "sess", 0, "",
         "user_stated", 0, 0, 0, "instant", "explicit", 0,
         len(text.encode()), gen),
    )
    rid = int(cur.lastrowid)
    conn.execute(
        'INSERT INTO unit_fts(rowid,text,speaker,entities,session,'
        '"when",ctx) VALUES(?,?,?,?,?,?,?)',
        (rid, text, "", "", "", "", ctx),
    )


class TestCtxFieldWeight:
    def test_default_off_ctx_ignored(self):
        """No arm → the ``ctx`` column never enters the scored field
        set: a term appearing ONLY in ctx earns zero fielded score."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(_CTX_DDL)
        _seed_ctx_unit(conn, "u1", text="alpha beta", ctx="zebra")
        out = lane_lexical(_lex_ctx(conn), _lex_qv("zebra"), _lex_slice())
        assert out.status == LaneStatus.OK
        assert len(out.candidates) == 1
        assert out.candidates[0].raw_score == 0.0
        gate = out.stats["ctx_field_weight"]
        assert gate["armed"] is None and gate["indexed"] is False

    def test_armed_ctx_contributes(self):
        """``True`` → the 0.5 prior; the ctx-only term now scores."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(_CTX_DDL)
        _seed_ctx_unit(conn, "u1", text="alpha beta", ctx="zebra")
        ctx = _lex_ctx(conn, manifest={"context.ctx_field_weight": True})
        out = lane_lexical(ctx, _lex_qv("zebra"), _lex_slice())
        assert out.candidates[0].raw_score > 0.0
        gate = out.stats["ctx_field_weight"]
        assert gate["armed"] == CTX_FIELD_W_DEFAULT
        assert gate["indexed"] is True

    def test_numeric_arm_scales_contribution(self):
        """A numeric override changes the score — larger weight, larger
        BM25F contribution for the same fixture."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(_CTX_DDL)
        _seed_ctx_unit(conn, "u1", text="alpha beta", ctx="zebra")
        half = lane_lexical(
            _lex_ctx(conn, manifest={"context.ctx_field_weight": 0.5}),
            _lex_qv("zebra"), _lex_slice())
        full = lane_lexical(
            _lex_ctx(conn, manifest={"context.ctx_field_weight": 1.0}),
            _lex_qv("zebra"), _lex_slice())
        assert full.candidates[0].raw_score > half.candidates[0].raw_score

    def test_zero_is_honest_not_absent(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(_CTX_DDL)
        _seed_ctx_unit(conn, "u1", text="alpha beta", ctx="zebra")
        ctx = _lex_ctx(conn, manifest={"context.ctx_field_weight": 0.0})
        out = lane_lexical(ctx, _lex_qv("zebra"), _lex_slice())
        # Armed (the field is fetched + counted) but weighted 0 —
        # the arm is read, not dropped.
        assert out.stats["ctx_field_weight"]["armed"] == 0.0
        assert out.stats["ctx_field_weight"]["indexed"] is True
        assert out.candidates[0].raw_score == 0.0

    def test_armed_but_unindexed_reports_honestly(self):
        """Schema without ``ctx``: the arm resolves, the column gate
        reports ``indexed=False``, and nothing is fabricated."""
        conn = sqlite3.connect(":memory:")
        conn.executescript(_PLAIN_DDL)
        cur = conn.execute(
            "INSERT INTO units(unit_id, source_id, revision, scope_id,"
            " kind, parent_unit_id, session_id, seq, speaker_canon,"
            " perspective, recorded_at_us, occurred_start_us,"
            " occurred_end_us, occurred_precision, occurred_source,"
            " byte_start, byte_end, generation)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("u1", "u1", 1, "s", "turn", None, "sess", 0, "",
             "user_stated", 0, 0, 0, "instant", "explicit", 0, 10, 1),
        )
        conn.execute(
            'INSERT INTO unit_fts(rowid,text,speaker,entities,session,'
            '"when") VALUES(?,?,?,?,?,?)',
            (int(cur.lastrowid), "alpha zebra", "", "", "", ""),
        )
        ctx = _lex_ctx(conn, manifest={"context.ctx_field_weight": True})
        out = lane_lexical(ctx, _lex_qv("zebra"), _lex_slice())
        assert out.status == LaneStatus.OK
        gate = out.stats["ctx_field_weight"]
        assert gate["armed"] == CTX_FIELD_W_DEFAULT
        assert gate["indexed"] is False

    @pytest.mark.parametrize("bad", ["bogus", -0.5, float("nan"), [0.5]])
    def test_malformed_raises_validation(self, bad):
        conn = sqlite3.connect(":memory:")
        conn.executescript(_CTX_DDL)
        ctx = _lex_ctx(conn, manifest={"context.ctx_field_weight": bad})
        with pytest.raises(VerbatimError) as ei:
            lane_lexical(ctx, _lex_qv("zebra"), _lex_slice())
        assert ei.value.code == ErrorCode.VALIDATION

    def test_policy_params_carrier(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(_CTX_DDL)
        _seed_ctx_unit(conn, "u1", text="alpha beta", ctx="zebra")
        ctx = _lex_ctx(
            conn, params={"context.ctx_field_weight": True})
        out = lane_lexical(ctx, _lex_qv("zebra"), _lex_slice())
        assert out.candidates[0].raw_score > 0.0
        assert out.stats["ctx_field_weight"]["indexed"] is True


# ---------------------------------------------------------------------------
# rerank.speaker_match_weight (V8-11.02) — S4 coefficient arm
# ---------------------------------------------------------------------------


def _sm_fused():
    """Caroline's turn + Jon's turn — the speaker feature fires on the
    first when the query's speaker canon is ``caroline``."""
    return [
        FusedCandidate(
            unit_id="u_c", source_id="src", revision=1,
            rrf=0.016, lane_ranks={"lex": 1},
            signals={"speaker_canon": "caroline"}),
        FusedCandidate(
            unit_id="u_j", source_id="src", revision=1,
            rrf=0.016, lane_ranks={"lex": 2},
            signals={"speaker_canon": "jon"}),
    ]


def _sm_qv():
    return QueryViewV7(
        query="what did caroline say",
        norm=NormAnalysis(analyzer_id="norm/v2", terms=(),
                          identifiers=(), text="what did caroline say"),
        intent=IntentResult(primary=IntentClass.LOOKUP,
                            classes=(IntentClass.LOOKUP,)),
        speaker_canon="caroline",
    )


def _sm_ctx(**params):
    return LaneContextV7(
        store=None, scope_id="s", generation=1, eligible=None,
        query_time_us=0, profile="test", budget=BudgetClass.MID,
        policy=_policy(**params), manifest={})


class TestSpeakerMatchWeight:
    def test_default_prior(self):
        scored = score_candidates(_sm_qv(), _sm_fused(), ctx=_sm_ctx())
        meta = scored.stats["speaker_match_weight"]
        assert meta["value"] == SPEAKER_MATCH_WEIGHT_DEFAULT == 0.4
        assert meta["declared"] is None and meta["source"] == "default"

    def test_numeric_override_changes_score(self):
        base = score_candidates(_sm_qv(), _sm_fused(), ctx=_sm_ctx())
        armed = score_candidates(
            _sm_qv(), _sm_fused(),
            ctx=_sm_ctx(**{"rerank.speaker_match_weight": 0.9}))
        meta = armed.stats["speaker_match_weight"]
        assert meta["value"] == 0.9 and meta["source"] == "policy_arm"
        base_c = next(s for s in base if s.unit_id == "u_c")
        arm_c = next(s for s in armed if s.unit_id == "u_c")
        # speaker_match==1.0 on u_c — the delta is exactly the weight gap.
        assert math.isclose(
            arm_c.score - base_c.score, 0.9 - 0.4, rel_tol=1e-9)
        # Jon's unit carries speaker_match=0.0 — its score is unchanged.
        base_j = next(s for s in base if s.unit_id == "u_j")
        arm_j = next(s for s in armed if s.unit_id == "u_j")
        assert math.isclose(arm_j.score, base_j.score, rel_tol=1e-9)

    def test_named_forms(self):
        for name, want in (("current", 0.4), ("half", 0.2), ("zero", 0.0),
                           ("off", 0.0), ("none", 0.0)):
            scored = score_candidates(
                _sm_qv(), _sm_fused(),
                ctx=_sm_ctx(**{"rerank.speaker_match_weight": name}))
            assert scored.stats["speaker_match_weight"]["value"] == want

    def test_zero_disables_contribution(self):
        scored = score_candidates(
            _sm_qv(), _sm_fused(),
            ctx=_sm_ctx(**{"rerank.speaker_match_weight": "zero"}))
        c = next(s for s in scored if s.unit_id == "u_c")
        # The feature still computes (1.0) but contributes nothing.
        assert c.detail["features"]["speaker_match"] == 1.0
        meta = scored.stats["speaker_match_weight"]
        assert meta["value"] == 0.0 and meta["source"] == "policy_arm"

    def test_explicit_weights_outrank_arm(self):
        weights = dict(FEATURE_WEIGHTS_V1)
        weights["speaker_match"] = 0.7
        scored = score_candidates(
            _sm_qv(), _sm_fused(), weights=weights,
            ctx=_sm_ctx(**{"rerank.speaker_match_weight": 0.1}))
        meta = scored.stats["speaker_match_weight"]
        assert meta["value"] == 0.7 and meta["source"] == "weights"
        assert meta["declared"] == 0.1  # declared, honestly recorded

    def test_manifest_carrier(self):
        ctx = LaneContextV7(
            store=None, scope_id="s", generation=1, eligible=None,
            query_time_us=0, profile="test", budget=BudgetClass.MID,
            policy=RetrievalPolicyV7(
                policy_id="retrieval_policy/v7", profile="test",
                lanes=(LaneName.LEX,), lane_weights={}),
            manifest={"rerank.speaker_match_weight": 0.25})
        scored = score_candidates(_sm_qv(), _sm_fused(), ctx=ctx)
        assert scored.stats["speaker_match_weight"]["value"] == 0.25

    @pytest.mark.parametrize("bad", ["bogus", -0.1, float("nan"), True,
                                     [0.2]])
    def test_malformed_raises_validation(self, bad):
        with pytest.raises(VerbatimError) as ei:
            score_candidates(
                _sm_qv(), _sm_fused(),
                ctx=_sm_ctx(**{"rerank.speaker_match_weight": bad}))
        assert ei.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# dense.embed_batch + dense.tier — behavioral, through the real job path
# ---------------------------------------------------------------------------


@pytest.fixture()
def vcfg() -> VerbatimConfig:
    c = VerbatimConfig()
    return replace(c, capture=replace(c.capture, enabled=True))


@pytest.fixture()
def vstore(tmp_path):
    s = Store.create(str(tmp_path / "v8flags.db"))
    yield s
    s.close()


@pytest.fixture()
def vencoder(vcfg) -> HashingEncoder:
    return HashingEncoder(vcfg.embedding)


@pytest.fixture()
def vingester(vstore, vcfg, vencoder):
    return Ingester(vstore, vcfg, encoder=vencoder)


def _capture_src(ingester, text: str, *, metadata=None) -> str:
    env = SourceEnvelope(
        origin="test",
        source_kind=SourceKind.USER_MESSAGE,
        scope=SCOPE,
        speaker_id="alice",
        payload=text.encode("utf-8"),
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
        metadata=metadata or {},
    )
    r = ingester.ingest(env)
    sid = r.accepted[0]
    with ingester.store.tx() as conn:
        sj.enqueue_source_jobs(
            conn, ingester.store, receipt_id=ingest_receipt_id(sid, 1))
    return sid


def _drain_kind(ingester, kind, handler, limit=8):
    leased = ingester.jobs.lease(None, [kind], owner="w1", limit=limit)
    for j in leased:
        handler(j, "w1", ingester)
    return leased


def _job_state(store, job_id):
    with store.read() as conn:
        row = conn.execute(
            "SELECT state FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return row[0] if row else None


def _pending_embed_jobs(store):
    with store.read() as conn:
        return [
            r[0]
            for r in conn.execute(
                "SELECT job_id FROM jobs WHERE kind='source_embed'"
                " AND state IN ('queued','retry_wait')"
            ).fetchall()
        ]


def _embed_events(store):
    with store.read() as conn:
        return conn.execute(
            "SELECT payload_json FROM events WHERE kind IN"
            " ('source_embedded','source_embed_skipped')"
        ).fetchall()


def _freeze_queue_clock(ingester) -> None:
    """Pin ``ingester.jobs._now`` at the newest enqueue stamp so every
    sibling's measured queue age is 0 — the ``dense.embed_batch`` age
    bound stays inert and tests isolate the row bound. (The real clock
    makes >250 ms-old siblings legitimately close a batch.)"""
    with ingester.store.read() as conn:
        rows = conn.execute(
            "SELECT input_refs_json FROM jobs WHERE kind='source_embed'"
        ).fetchall()
    stamps = [
        (json.loads(r[0]) or {}).get("_enqueued_us") for r in rows
    ]
    stamps = [s for s in stamps if isinstance(s, (int, float))]
    if stamps:
        t0 = int(max(stamps))
        ingester.jobs._now = lambda: t0


class TestEmbedBatchJobs:
    """``dense.embed_batch`` behavior through the real leased path."""

    def _three_pending(self, ingester):
        for i in range(3):
            _capture_src(ingester, f"coalesced source number {i}")
        _drain_kind(ingester, JobKind.SOURCE_PROJECT,
                    sj.handle_source_project)
        pending = _pending_embed_jobs(ingester.store)
        assert len(pending) == 3
        return pending

    def test_default_coalesces_all_siblings(self, vingester):
        self._three_pending(vingester)
        _freeze_queue_clock(vingester)   # isolate: no age-bound flush
        leased = vingester.jobs.lease(
            None, [JobKind.SOURCE_EMBED], owner="w1", limit=1)
        sj.handle_source_embed(leased[0], "w1", vingester)
        # All three embed jobs settled in the shared commit.
        assert _pending_embed_jobs(vingester.store) == []
        assert len(_embed_events(vingester.store)) == 3

    def test_rows_bound_closes_batch(self, vingester):
        """``dense.embed_batch`` rows=1 — one sibling rides the commit;
        the tail stays queued for the next drain (FIFO, never lost)."""
        self._three_pending(vingester)
        _freeze_queue_clock(vingester)
        leased = vingester.jobs.lease(
            None, [JobKind.SOURCE_EMBED], owner="w1", limit=1)
        leased[0]["input_refs"]["dense.embed_batch"] = {"rows": 1}
        sj.handle_source_embed(leased[0], "w1", vingester)
        # Primary + exactly one sibling committed; the tail is queued.
        assert len(_embed_events(vingester.store)) == 2
        assert len(_pending_embed_jobs(vingester.store)) == 1

    def test_age_bound_closes_batch(self, vingester):
        """``max_age_ms: 0`` — the first admitted sibling already exceeds
        its queue-age bound → the batch closes after it (a deadline
        flush, never a skip)."""
        self._three_pending(vingester)
        leased = vingester.jobs.lease(
            None, [JobKind.SOURCE_EMBED], owner="w1", limit=1)
        leased[0]["input_refs"]["dense.embed_batch"] = {"max_age_ms": 0}
        sj.handle_source_embed(leased[0], "w1", vingester)
        assert len(_embed_events(vingester.store)) == 2
        assert len(_pending_embed_jobs(vingester.store)) == 1

    def test_encode_calls_bounded_by_rows(self, vingester, vencoder):
        """The row bound also caps one ``encoder.encode`` call — a
        multi-unit revision encodes in chunks, never unbounded."""
        contents = [f"turn {i} says something distinct" for i in range(4)]
        _capture_src(
            vingester,
            "\n".join(contents),
            metadata={"messages": [
                {"role": "user", "speaker": "alice", "content": c,
                 "session_id": "s9"}
                for c in contents]},
        )
        _drain_kind(vingester, JobKind.SOURCE_PROJECT,
                    sj.handle_source_project)
        with vingester.store.read() as conn:
            n_units = conn.execute(
                "SELECT COUNT(*) FROM units WHERE kind='turn'"
            ).fetchone()[0]
        assert n_units == 4

        sizes = []
        real_encode = vencoder.encode

        def _spy(texts):
            sizes.append(len(list(texts)))
            return real_encode(texts)

        vencoder.encode = _spy
        try:
            leased = vingester.jobs.lease(
                None, [JobKind.SOURCE_EMBED], owner="w1", limit=1)
            leased[0]["input_refs"]["dense.embed_batch"] = {"rows": 2}
            sj.handle_source_embed(leased[0], "w1", vingester)
        finally:
            vencoder.encode = real_encode
        assert sizes, "encoder never invoked"
        assert max(sizes) <= 2
        assert len(sizes) > 1  # chunked, not one unbounded call

    def test_malformed_embed_batch_fails_loudly(self, vingester):
        _capture_src(vingester, "a source")
        _drain_kind(vingester, JobKind.SOURCE_PROJECT,
                    sj.handle_source_project)
        leased = vingester.jobs.lease(
            None, [JobKind.SOURCE_EMBED], owner="w1", limit=1)
        leased[0]["input_refs"]["dense.embed_batch"] = "bogus"
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_embed(leased[0], "w1", vingester)
        assert ei.value.code == ErrorCode.VALIDATION


class TestDenseTierJobs:
    """``dense.tier`` write-side gate through the real leased path."""

    def _one_pending(self, ingester, text="tiered source"):
        _capture_src(ingester, text)
        _drain_kind(ingester, JobKind.SOURCE_PROJECT,
                    sj.handle_source_project)
        leased = ingester.jobs.lease(
            None, [JobKind.SOURCE_EMBED], owner="w1", limit=1)
        assert len(leased) == 1
        return leased[0]

    def test_unprovisioned_tier_defers_honestly(self, vingester):
        """``local_memory_quality`` pins ``potion-base-8m`` — the
        hashing encoder can't satisfy it: recorded deferral, no vectors,
        never a silent downgrade."""
        job = self._one_pending(vingester)
        job["input_refs"]["dense.tier"] = "local_memory_quality"
        sj.handle_source_embed(job, "w1", vingester)
        with vingester.store.read() as conn:
            n_vecs = conn.execute(
                "SELECT COUNT(*) FROM source_vectors").fetchone()[0]
            n_blocks = conn.execute(
                "SELECT COUNT(*) FROM unit_vectors_block"
            ).fetchone()[0]
            ev = conn.execute(
                "SELECT payload_json FROM events"
                " WHERE kind='source_embed_skipped'"
            ).fetchone()
        assert n_vecs == 0 and n_blocks == 0
        payload = json.loads(ev[0])
        assert payload["reason"] == "dense_tier_unprovisioned"

    def test_satisfied_tier_encodes(self, vingester):
        """``local_memory`` → hashing — the provisioned encoder
        satisfies the pin and the job runs normally."""
        job = self._one_pending(vingester)
        job["input_refs"]["dense.tier"] = "local_memory"
        sj.handle_source_embed(job, "w1", vingester)
        with vingester.store.read() as conn:
            n_vecs = conn.execute(
                "SELECT COUNT(*) FROM source_vectors").fetchone()[0]
        assert n_vecs == 1

    def test_literal_tier_name(self, vingester):
        job = self._one_pending(vingester)
        job["input_refs"]["dense.tier"] = "hashing"
        sj.handle_source_embed(job, "w1", vingester)
        with vingester.store.read() as conn:
            n_vecs = conn.execute(
                "SELECT COUNT(*) FROM source_vectors").fetchone()[0]
        assert n_vecs == 1

    def test_malformed_tier_fails_loudly(self, vingester):
        job = self._one_pending(vingester)
        job["input_refs"]["dense.tier"] = 42
        with pytest.raises(VerbatimError) as ei:
            sj.handle_source_embed(job, "w1", vingester)
        assert ei.value.code == ErrorCode.VALIDATION
