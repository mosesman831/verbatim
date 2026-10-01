# V7 contracts — frozen worker interfaces (SPEC_V7 R1, wave A)

Binding per V7-27.01/36.01. Two workers never own the same file. Workers code
against the signatures below and `verbatim/core/types_v7.py`; they MUST NOT
edit this file, `types_v7.py`, `SPEC_V7.md`, shared `__init__.py` files
inside existing packages (new package dirs need their own `__init__.py`,
which the owning worker creates), `conftest.py`, `pyproject.toml`, or
`verbatim/memory/facade.py` / `verbatim/ingest.py` / `verbatim/jobs/source_jobs.py`
(write-path seams are integrated by the main session in a later wave).

## Hard invariants (every worker)

- Eligibility before rank; a lane never widens the caller's eligible set.
- Degradation is honest: `skipped/partial/unavailable/deadline` + reason;
  never fabricate results; never turn a result `ready` from a degraded path.
- Every deliverable-as-text artifact carries byte pins or is labeled
  `unsupported_extraction`.
- No network, no model download, no LLM call anywhere in wave A. Optional
  packages (numpy, onnxruntime, tokenizers) are imported lazily behind
  `try/except ImportError` and their absence reports `unavailable`.
- Pure stdlib + existing repo deps only. No new dependencies.
- Determinism: identical inputs -> identical outputs; fixed seeds in tests.
- Do not run the full test suite, envelopes, or Track R (quiet-tree rule
  V7-27.03). Run only your own test file(s):
  `python -m pytest <your test file> -x -q`.
- All §32 constants are `provisional/v7-r0`; record the tag on artifacts.

## Frozen APIs

```python
# verbatim/text/norm_v2.py   (§32.1, V7-05.10/11)
ANALYZER_ID = "norm/v2"
def analyze(text: str) -> NormAnalysis           # terms + identifier channel + byte offsets
def fold(surface: str) -> str                    # NFKC + casefold + diacritic strip (matching projection)

# verbatim/querying/intent_v2.py  (§32.14, V7-05.12/13)
def classify(norm: NormAnalysis, entity_canons: tuple[str, ...],
             identifiers: tuple[NormTerm, ...]) -> IntentResult
def decompose(norm: NormAnalysis, intent: IntentResult) -> tuple[NormAnalysis, ...]  # ≤ max_facets

# verbatim/storage/schema_v7.py  (§30, V7-30.01–04)
# NOTE (R1a, audit finding): `events` and `entity_aliases` are V1 table names —
# V7 tables are named `events_v7` and `entity_aliases_v7`. Every V7 table
# carries scope_id + generation; the generation fence is `<=` snapshot.
SCHEMA_V7_TAG = "schema_v7/v1"
V7_TABLES: tuple[str, ...]                       # every §30 table name (renames applied)
DDL_V7: str                                      # additive DDL incl. indexes
def ensure_v7_additive(conn) -> None             # lazy first-writer creation (V6 semantics)
def v7_tables_present(conn) -> bool

# verbatim/storage/stats_v7.py  (V7-06.06)
def update_stats(conn, scope_id: str, generation: int,
                 unit_rows: Iterable[dict]) -> None   # incremental df + length
def corpus_stats(conn, scope_id: str, generation: int) -> dict  # {field: {n,total_len}}, df lookup helper

# verbatim/enrichment/entities_v2.py  (V7-08.01–06, §32.7)
def canon(surface: str) -> str                   # fold + strip possessive
def extract_mentions(norm: NormAnalysis, unit_id: str,
                     speaker: str | None) -> list[EntityMention]
def propose_aliases(mentions: Iterable[EntityMention],
                    known_canons: Iterable[str]) -> list[AliasRow]   # rules A1–A6
def expand_query(canons: Iterable[str], aliases: Iterable[AliasRow],
                 limit: int = 8) -> list[str]

# verbatim/enrichment/temporal_v2.py  (V7-09.03–05, §32.9 T01–T30)
RESOLVER_ID = "temporal/v2"
def resolve(text: str, anchor_us: int, *, locale: str = "en-US",
            hemisphere: str = "north") -> list[ResolvedTime]
def resolve_query_window(norm: NormAnalysis, query_time_us: int) -> Optional[IntervalUs]

# verbatim/temporal/algebra.py  (V7-09.10)
def contains(outer: IntervalUs, inner: IntervalUs) -> bool
def overlaps(a: IntervalUs, b: IntervalUs) -> bool
def before(a: IntervalUs, b: IntervalUs) -> bool
def duration(a: IntervalUs, b: IntervalUs, unit: str) -> float   # calendar-correct
def shift(iv: IntervalUs, n: int, unit: str) -> IntervalUs
def age_at_date(birth: IntervalUs, at_us: int) -> Optional[int]

# verbatim/enrichment/events.py  (V7-09.08, §32.10 `event/v1`)
def extract_events(norm: NormAnalysis, unit_id: str, speaker_canon: str | None,
                   occurred: IntervalUs, sieve: Callable | None = None) -> list[EventTuple]

# verbatim/enrichment/coref_sieve.py  (V7-13.20, `coref_sieve/v1`)
def resolve_antecedent(mention: str, unit_index: int,
                       session_turns: list[dict], lookback: int = 6) -> Optional[str]
   # returns canon or None (abstain); rules: recency, number/gender where reliable,
   # co-occurrence; never picks when two candidates plausible

# verbatim/enrichment/prefs_state.py  (§32.11/12, V7-13.08, V7-16.03)
def extract_preferences(norm: NormAnalysis, unit_id: str,
                        speaker_canon: str) -> list[PreferenceFact]
def extract_state_facts(norm: NormAnalysis, unit_id: str,
                        speaker_canon: str, occurred: IntervalUs) -> list[StateFact]
def state_compatible(key: str, a: str, b: str) -> bool   # per-family compatibility fn

# verbatim/retrieval/v7/fusion.py  (V7-10.01–05)
def rrf_fuse(outputs: list[LaneOutput], weights: dict[LaneName, float],
             k: int = 60) -> list[FusedCandidate]
   # no ties: distinct raw signal vectors never collapse (V7-10.02);
   # constant signals (≥90% same value) contribute 0 (V7-10.03)

# verbatim/retrieval/v7/rerank_features.py  (§32.4, V7-10.06/07)
FEATURE_WEIGHTS_V1: dict[str, float]
def score_candidates(query: QueryViewV7, fused: list[FusedCandidate],
                     ctx) -> list[ScoredCandidate]   # "ranking/v7"

# verbatim/retrieval/v7/boosts.py  (§32.5, V7-10.10/11)
def apply_boosts(scored: list[ScoredCandidate], query: QueryViewV7,
                 now_us: int) -> list[ScoredCandidate]  # swing within [-25%,+30%]

# verbatim/querying/verdict_v2.py  (V7-11)
def classify_groups(items, query: QueryViewV7, ctx) -> list[GroupVerdict]
def result_verdict(groups: list[GroupVerdict], query: QueryViewV7,
                   calibration) -> tuple[ResultStatus, MissingDescriptor | None]
   # triggers (a)-(d) only; no literal-coverage floor (D7-04 closed)

# verbatim/retrieval/v7/policy.py  (V7-05.02/07)
def load_policy(profile: str, policy_json: dict | None = None) -> RetrievalPolicyV7
def pool_for(budget: BudgetClass) -> PoolProfile
LANE_WEIGHTS_V1: dict  # §32.3 table

# verbatim/retrieval/v7/deadline.py  (V7-05.05)
def allocate(remaining_ms: float, lanes: Iterable[LaneName],
             policy: RetrievalPolicyV7) -> dict[LaneName, LaneSlice]

# verbatim/retrieval/v7/pipeline.py  (§04.2, V7-04.01–04, V7-05)
LANE_REGISTRY: dict[LaneName, Callable]   # populated at import by lane modules
def run_search(ctx: LaneContextV7, query: QueryViewV7,
               deadline_ms: float) -> "PipelineResult"
@dataclass class PipelineResult: lanes: dict[str, LaneOutput];
    fused: list[FusedCandidate]; scored: list[ScoredCandidate];
    verdict: ResultStatus; missing: MissingDescriptor | None;
    coverage: CoverageV7; stage: StageRecord; explain: dict | None

# verbatim/retrieval/v7/lexical.py  (V7-06, §32.2 bm25f/v1)
def lane_lexical(ctx: LaneContextV7, qv: QueryViewV7, slice: LaneSlice) -> LaneOutput
def bm25f_score(...)   # hand-computable; tests vs pure-Python reference

# verbatim/retrieval/v7/fuzzy.py  (V7-06.04)
def lane_fuzzy(ctx, qv, slice) -> LaneOutput   # trigram + respell df=0 terms

# verbatim/retrieval/v7/graph.py + verbatim/jobs/graph_jobs.py  (V7-08.07–12, §32.6)
def lane_graph(ctx, qv, slice, seeds: list[str]) -> LaneOutput  # bounded graph expansion (§32.6 r0; Q6 selects)
def build_edges(conn, scope_id: str, generation: int, unit_ids) -> int  # job handler

# verbatim/retrieval/v7/temporal.py  (V7-09.06–09)
def lane_temporal(ctx, qv, slice) -> LaneOutput  # event-index first for temporal intents

# verbatim/retrieval/v7/dense.py + verbatim/embeddings/matrix.py  (V7-07.04/05)
def lane_dense(ctx, qv, slice) -> LaneOutput   # contiguous-matrix scan; honest
def write_block(conn, encoder_id, scope_id, generation, block_no, rows) -> None
def unit_vector_encoder_id(encoder_id) -> str  # unit-vector space "<id>:hdr:v1" (V7-07.09)
def block_row_keys(conn, encoder_id, scope_id, generation) -> set[str]  # rowmap keys only
def scan_block(...)  # array('f') blocked loop == per-row oracle ordering

# verbatim/retrieval/v7/pack.py / render.py / computed.py  (V7-12, §32.15)
def assemble_pack(scored, query, ctx, max_tokens, limit, neighbor_n) -> PackResult
def render_reader_view(pack, query_time_us) -> str    # pack_render/v1 format
def computed_items(query: QueryViewV7, items) -> list[ComputedItem]
def estimate_tokens(text: str) -> int                 # tok/v1

# verbatim/projections/units_v7.py  (V7-30.01, V7-13.02–05)
def derive_units(source_row, revision_row, add_args: dict) -> list[dict]
   # deterministic: sources+revisions+add args -> units rows, byte-identical

# eval/v7/ modules
#   ledger.py: parse SPEC_V7.md requirement ids -> ledger_v7.json (--check)
#   manifest.py: run-manifest builder + digest (§22 manifest fields)
#   dataset_registry.py: registered datasets {id, license, path, loader, gate}
#   twins_*.py: deterministic corpus generators (seeded; no benchmark text)
#   track_r.py: retrieval-only runner; metrics any@k/all@k/ndcg@10 per §33
#   arms.py: verbatim arm + flat_bm25 reference + fts5 baseline
#   attribution.py: per-query attribution classes (lane_miss|rank_shift|packed_out|abstain|unsupported)
#   stage_profile.py: §32.17 record + p50/p95/p99 envelope aggregation
#   gates.py: gate evaluator skeleton — reads artifacts + ledger only (V7-26.02)
```

## Ownership map (wave A)

Package `__init__.py` markers (`verbatim/text`, `verbatim/temporal`,
`verbatim/retrieval/v7`, `verbatim/extraction`, `verbatim/reflect`,
`eval/v7`, `tests/text`, `tests/temporal`, `tests/retrieval/v7`) are
pre-created and owned by the main session — workers create only the named
modules and test files below.


| Worker | Owned files (all new) | Requirements |
| --- | --- | --- |
| w-aud-read | report only | D7-25+ retrieval read path |
| w-aud-write | report only | D7-25+ write path |
| w-aud-schema | report only | schema/storage gap map |
| w-aud-eval | report only | eval reuse map |
| w-aud-enrich | report only | enrichment gap map |
| w-aud-trust | report only | verdict/pack/security/mutation |
| w-types | (landed: main session wrote types_v7.py) | contracts |
| w-ledger | `eval/v7/ledger.py` `eval/v7/ledger_v7.json` `eval/v7/manifest.py` `eval/v7/gates.py` `tests/eval/test_v7_ledger.py` | V7-22/26 ledger, V7-26.02 |
| w-datasets | `eval/v7/dataset_registry.py` `eval/v7/corpora.py` `tests/eval/test_v7_datasets.py` | V7-22 dataset registry |
| w-twin-dialogue | `eval/v7/twins_locomo_like.py` `tests/eval/test_v7_twins_locomo.py` | V7-22 owned twins |
| w-twin-lme | `eval/v7/twins_lme_like.py` `tests/eval/test_v7_twins_lme.py` | V7-22 owned twins |
| w-twin-extra | `eval/v7/twins_actions.py` `eval/v7/twins_scale.py` `eval/v7/twins_prefs.py` `tests/eval/test_v7_twins_extra.py` | V7-22 owned twins |
| w-trackr | `eval/v7/track_r.py` `eval/v7/arms.py` `eval/v7/attribution.py` `eval/v7/metrics.py` `tests/eval/test_v7_trackr.py` | V7-22.12, §33 |
| w-profiler | `eval/v7/stage_profile.py` `eval/v7/mutations.yaml` `tests/eval/test_v7_profiler.py` | §32.17, V7-35.01 |
| w-norm | `verbatim/text/norm_v2.py` `tests/text/test_norm_v2.py` | V7-05.10/11, §32.1 |
| w-intent | `verbatim/querying/intent_v2.py` `tests/querying/test_intent_v2.py` | V7-05.12/13, §32.14 |
| w-schema | `verbatim/storage/schema_v7.py` `verbatim/storage/stats_v7.py` `tests/storage/test_schema_v7.py` | §30, V7-06.06 |
| w-entities | `verbatim/enrichment/entities_v2.py` `tests/enrichment/test_entities_v2.py` | V7-08.01–06, §32.7 |
| w-temporal | `verbatim/enrichment/temporal_v2.py` `tests/enrichment/test_temporal_v2.py` | V7-09.03–05, §32.9 |
| w-algebra | `verbatim/temporal/algebra.py` `tests/temporal/test_algebra.py` | V7-09.10 |
| w-events | `verbatim/enrichment/events.py` `tests/enrichment/test_events.py` | V7-09.08, §32.10 |
| w-coref | `verbatim/enrichment/coref_sieve.py` `tests/enrichment/test_coref.py` | V7-13.20 |
| w-prefs | `verbatim/enrichment/prefs_state.py` `tests/enrichment/test_prefs_state.py` | §32.11/12, V7-13.08 |
| w-fusion | `verbatim/retrieval/v7/fusion.py` `verbatim/retrieval/v7/rerank_features.py` `verbatim/retrieval/v7/boosts.py` `tests/retrieval/v7/test_fusion.py` | V7-10.01–11 |
| w-verdict | `verbatim/querying/verdict_v2.py` `tests/querying/test_verdict_v2.py` | V7-11 |
| w-policy | `verbatim/retrieval/v7/policy.py` `verbatim/retrieval/v7/deadline.py` `tests/retrieval/v7/test_policy.py` | V7-05.02/05/07 |
| w-pipeline | `verbatim/retrieval/v7/pipeline.py` `verbatim/retrieval/v7/lanes_base.py` `tests/retrieval/v7/test_pipeline.py` | §04.2, V7-04/05 |
| w-lexical | `verbatim/retrieval/v7/lexical.py` `verbatim/retrieval/v7/fuzzy.py` `tests/retrieval/v7/test_lexical.py` | V7-06 |
| w-graph | `verbatim/retrieval/v7/graph.py` `verbatim/jobs/graph_jobs.py` `tests/retrieval/v7/test_graph.py` | V7-08.07–12, §32.6 |
| w-tlane | `verbatim/retrieval/v7/temporal.py` `tests/retrieval/v7/test_tlane.py` | V7-09.06–09 |
| w-dense | `verbatim/retrieval/v7/dense.py` `verbatim/embeddings/matrix.py` `tests/retrieval/v7/test_dense.py` | V7-07.04/05 |
| w-pack | `verbatim/retrieval/v7/pack.py` `verbatim/retrieval/v7/render.py` `verbatim/retrieval/v7/computed.py` `tests/retrieval/v7/test_pack.py` | V7-12 |
| w-units | `verbatim/projections/units_v7.py` `tests/projections/test_units_v7.py` | V7-30.01, V7-13.02–05 |

Wave briefs archived under `eval/v7/waves/`. Integration (facade seams,
lane registration, Track R on the real engine) is main-session work per
V7-27.04.
