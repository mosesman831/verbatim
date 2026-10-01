# d1 — Hindsight (vectorize-io/hindsight): TEMPR retrieval as implemented in code

Repo cloned to `/tmp/hindsight-research/hindsight` (public, MIT License — `LICENSE`, © 2025 Vectorize AI, Inc.). Engine lives in `hindsight-api-slim/hindsight_api/engine/`. Companion paper: arXiv:2512.12818 (downloaded, 84.5k chars). Docs site sources in `hindsight-docs/docs/developer/` corroborate the code. Findings below separate **code** (verified at file:line) from **paper claims** (marked PAPER).

## 1. Provenance map

| Item | Code location | Value |
|---|---|---|
| Lane set | `engine/memory_engine.py:8878-8880` | `[semantic, bm25, graph] + temporal iff date constraint` |
| Fusion | `engine/search/fusion.py:29-109` | `RRF score(m) = Σ_lanes 1.0/(k + rank_lane(m))`, `k=60`, 1-indexed, **unweighted** |
| Fusion alternative | `engine/search/fusion.py:112-176` | `interleave_fusion` round-robin — used ONLY for consolidation dedup recall (`memory_engine.py:8881`, `consolidator.py:3177-3183`) |
| Pre-fusion cap | `engine/search/fusion.py:8`; `config.py:1319` | `cap_per_source(results, cap)`; `DEFAULT_RECALL_MAX_CANDIDATES_PER_SOURCE = 0` (off) |
| Reranker model | `config.py:1268` | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local default; TEI/flashrank/Cohere etc. supported) |
| Reranker pool cap | `config.py:1291` + `memory_engine.py:8922-8936` | **`DEFAULT_RERANKER_MAX_CANDIDATES = 300`** — the "~32" in the assignment is actually the *batch size* `DEFAULT_RERANKER_LOCAL_BATCH_SIZE = 32` (`config.py:1276`). Pool cap is 300. Per-budget overrides (`_LOW/_MID/_HIGH`) default to 0 → fall back to 300 (`config.py:1292-1295`, `memory_engine.py:1486-1501`). |
| CE input pair | `engine/search/reranking.py:373-384` | `[query, "[Date: {Month D, YYYY} ({YYYY-MM-DD})] {context}: {text}"]` |
| CE normalization | `reranking.py:402-422` | sigmoid for local logits; pass-through for [0,1]-calibrated providers; NaN→0 |
| Boost alphas | `reranking.py:19-21` | `RECENCY=0.2, TEMPORAL=0.2, PROOF_COUNT=0.1` |
| Combined score | `reranking.py:284-287` | `final = CE_norm × (1+0.2(rec−0.5)) × (1+0.2(temp−0.5)) × (1+0.1(proof−0.5))` |
| Passthrough CE | `reranking.py:245` | when reranker off: `CE_norm = 1.0 − 0.9·rank/(n−1)` |
| 365-day decay | `reranking.py:35-62`, `config.py:1043`, `:1324` | linear `max(0.1, min(1.0, 1−days/365))` default; exponential `0.5**(days/90)`; "none"→0.5 |
| Coarse dates | `reranking.py:84-155` | month/year-spanning dates scored from period END, capped ≤0.5 (`_spans_calendar_period`, ±86400s tolerance) |
| Graph arm | `engine/search/link_expansion_retrieval.py` | **single-hop** additive expansion, NOT PPR |
| Temporal arm | `engine/search/retrieval.py:463-792` | window-overlap pool → coverage selection → bounded BFS spreading |
| Entity resolution | `engine/entity_resolver.py` | trgm probe → hard gates → weighted score → merge ≥0.6 |
| Observations | `engine/consolidation/prompts.py:39-172`, `consolidator.py` | **LLM creates/updates/deletes** (JSON) + deterministic verbatim-duplicate drop + cosine≥0.97 dedup adjudication |
| Query analysis | `engine/query_analyzer.py`, `temporal_extraction.py:38-47` | default `DateparserQueryAnalyzer` (deterministic dateparser, `PREFER_DATES_FROM=past`), offloaded to a 1-worker ThreadPoolExecutor; `TransformerQueryAnalyzer` (flan-t5-small) exists, NOT default |
| Storage | alembic `5a366d414dce` + `d5e6f7a8b9c0` | PostgreSQL + pgvector (Oracle 23ai alt dialect); tables `banks, documents, chunks, entities, entity_cooccurrences, memory_units, unit_entities, memory_links, async_operations, knowledge_*` |
| Embedding | `config.py:1207,1264` | `BAAI/bge-small-en-v1.5`, 384-dim (ONNX default `intfloat/multilingual-e5-small`) |
| BM25 | `engine/sql/postgresql.py:306-417` | native = `ts_rank_cd` (cover-density, NO IDF — **not Okapi BM25**); alternatives vchord/pg_textsearch/pg_search/pgroonga |
| BM25 term cap | `search/bm25_term_selection.py:82-118`; `config.py` `bm25_max_query_terms=16` | keep 16 lowest-df tokens via `pg_stats.most_common_elems` |
| Budgets | `memory_engine.py:1453-1483`; `config.py:1908-1917` | fixed low/mid/high = 100/300/1000; adaptive = `clamp(max_tokens×{0.025,0.075,0.25}, 20, 2000)` |
| Token pack | `engine/fact_budget.py:43-87` | skip-and-continue under `max_tokens` (default 4096); floor = top-1 fact if nothing fits |
| Truncate | `memory_engine.py:9181` | `rerank_limit = thinking_budget × 2` before token pack |

## 2. The real lane set and per-arm mechanics

### Semantic arm (`retrieval.py:123-400`, `sql/postgresql.py:306-417`)
Per fact_type: `ORDER BY embedding <=> $emb LIMIT thinking_budget` over pgvector HNSW, `AND (1 - embedding <=> $emb) >= semantic_min_similarity` where `DEFAULT_SEMANTIC_MIN_SIMILARITY = 0.3` (`config.py:1296`). HNSW tuning `_ANN_TUNING_HIGH_RECALL`: `hnsw.ef_search=200`, `iterative_scan=strict_order` (`_vector_index.py:135-142`). UNION ALL across fact_types in ONE query — 1 roundtrip for all arms except temporal.

### BM25 arm
Native: `ts_rank_cd(search_vector, to_tsquery(lang, q))` + `search_vector @@ to_tsquery` gate; `search_vector` is a GENERATED tsvector over `text || ' ' || context` (`alembic/5a366d414dce:330`). Query tokenizer: lowercase, strip punctuation, split (`retrieval.py:34`). The `@@` gate + `ORDER BY ts_rank_cd DESC LIMIT budget` ranks EVERY matched row — a 16-term OR over a large bank caused a +60s production hang; fix = keep the 16 tokens with lowest corpus df read from `pg_stats.most_common_elems` (`bm25_term_selection.py`). `bm25_min_score = 0.0` (`config.py:1303`) — non-native backends that return ranked scores get an explicit floor; native has none.
PAPER claims "BM25" — code's default is cover-density `ts_rank_cd`, which has no IDF. True Okapi only via pluggable backends (vchord/pg_search).

### Temporal arm (`retrieval.py:463-792`) — runs ONLY when query has a date window
- Entry pool: ANN top `_TEMPORAL_POOL_SIZE=60` per fact_type with window-overlap WHERE: `occurred_start <= end AND occurred_end >= start OR mentioned_at/occurred_* in window`, `sim >= temporal_semantic_min_similarity = 0.1` (`retrieval.py:404`, `config.py:1298`).
- `_select_with_temporal_coverage(pool, ..., limit=_TEMPORAL_ENTRY_POINTS=10, n_buckets=8)` (lines 414-460): split window into 8 equal time-buckets, round-robin pick best-similarity item from each populated bucket tier — guarantees the answer set covers the whole window instead of clustering on the densest stretch.
- Proximity score: `1 − min(days_from_mid/(window_days/2), 1)`; missing date → 0.5 entry / 0.3 neighbor.
- BFS spreading (lines 663-785): batch_size=20, per_source_limit=10, max_iterations=5, link types `('temporal','causes','caused_by','enables','prevents')`, weight ≥0.1, boost `2.0` (causes/caused_by) / `1.5` (enables,prevents) / `1.0` (temporal); `propagated = parent_temporal × weight × boost × 0.7`; `combined = max(proximity, propagated)`; frontier continues iff propagated >0.2. Bounded by budget. **Propagated scores can exceed 1.0** (0.9×1.0×2.0×0.7 = 1.26) — fine inside one arm since fusion only uses rank.

### Graph arm (`link_expansion_retrieval.py`) — single-hop, NOT multi-hop PPR
- Seeds: up to `GRAPH_SEED_LIMIT=20` semantic hits at `similarity >= graph_seed_min_similarity = 0.3` (`link_expansion_retrieval.py:46`, `config.py:1297`); seeds reused from the semantic arm when `sem_min <= 0.3` (`retrieval.py:203-208`) — zero extra ANN call.
- Three expansion sub-signals in ONE CTE (`ops_postgresql.py:941-1037`), merged **additively** (`link_expansion_retrieval.py:240-264`), `graph_score = entity + semantic + causal ∈ [0,3]`:
  - **entity**: `COUNT(DISTINCT shared entity_id)` via `unit_entities` → `tanh(count × 0.5)` (∈[0,1]; tanh(1)=0.46, 2→0.76, 5→0.99)
  - **semantic**: `MAX(memory_links.weight)` for link_type='semantic', BOTH directions
  - **causal**: `MAX(ml.weight)` for `('causes','caused_by','enables','prevents')`, `DISTINCT ON` both directions — docstring claims "weight + 1.0" but code uses raw `ml.weight` (stale doc, `link_expansion_retrieval.py:16`)
- Guards: `link_expansion_per_entity_limit=200`, `link_expansion_timeout=10.0s` → fallback to semantic+causal only (`config.py:1580-1581`).
- Observation expansion: seeds' `source_memory_ids` → their entities → connected sources → observations sharing ≥1 source, score = `COUNT(DISTINCT shared sources)` (`ops_postgresql.py:1095-1155`).

PAPER §retrieval describes multi-hop BFS spreading activation `A(f,t+1)=max[A(f_i,t)·w·δ·μ(ℓ)]` for the graph arm — **code's graph arm is single-hop**; the multi-hop spreading with decay δ=0.7 and type-boosts μ lives only in the temporal arm. The docs' "multi-hop" marketing language refers to link-following generally.

## 3. Fusion — exact formula and worked example

```
score(m) = Σ_{lane ∈ arms} 1.0 / (60 + rank_lane(m))   # 1-indexed; absent from lane → no term
```
`fusion.py:29-85`, `k=60`, unweighted, all four arms equal (docs confirm: "all four strategies are weighted equally").

Worked example (4 lanes, verified in `worked2.py`):
- A: rank1 in semantic + rank1 in bm25 → 2/61 = **0.0328**
- B: rank3 in all 4 lanes → 4/63 = **0.0635** — B wins.
- With Verbatim weights {1.0,0.75,0.5,-}: A=0.0328, B=0.0516 → B/A=1.573 (B still wins, margin narrows).

**The critical side-finding for Verbatim:** `search/recall_boost.py:27-47` documents their own measurement that **score-space RRF weighting degenerates to lexicographic ordering** when a lane's weight exceeds the rank-term's dynamic range (`1/61 → 1/360`, factor ��5.9): measured recall@20 dropped 0.97→0.40 at w=7. Their fix — bias lanes in **rank space** (`boosted_rrf_score`, lines 104-128: `1/(k + rank/divisor)` with low/medium/high divisors {2,4,8}) or as **flat additive post-rerank** ({0.05,0.2,0.5}, lines 131-147). Verbatim's {1.0,0.75,0.5} weights are far below the collapse cliff, but the safer idiom is: keep RRF unweighted, express lane preference via rank-divisor or additive terms, not score multipliers.

`interleave_fusion` (round-robin) exists solely because consolidation's dedup recall needs every arm's top hit to reach the LLM — RRF buried the "semantic #1 twin" below budget (`consolidator.py:3177-3183`). Not for general recall.

## 4. Reranker and combined scoring

Default `reranking = "cross_encoder"`; bank config `enable_reranking=false` downgrades to `"rrf"` passthrough (`memory_engine.py:1504-1513`, `enable_reranking` default True `config.py:1901`). Consolidation dedup uses `"interleave"`.

Combined score (`reranking.py:158-288`):
```
proof_norm = clamp(0.5 + ln(proof_count)/10)            # ~272
final = CE_norm × (1 + 0.2·(recency − 0.5))
              × (1 + 0.2·(temporal − 0.5))
              × (1 + 0.1·(proof_norm − 0.5))
```
Bounded ≈ ×[0.615, 1.016] at CE=0.8 → **max +27% / −23%** (measured `worked2.py`). Rationale in code+docs: multiplicative keeps boosts proportional to base relevance — additive lets a barely-relevant recent item leapfrog a strong old one.

365-day decay (`reranking.py:35-62`): linear `max(0.1, min(1.0, 1 − days/365))` measured from `query_timestamp` (default now). Worked: 0d→1.0, 30d→0.918, 182d→0.501, ≥365d→0.1 floor. Alternatives: exponential `0.5**(days/90)`; "none"→constant 0.5. **Coarse-date subtlety** (`_spans_calendar_period`, 84-104): a fact stamped as a whole month/year gets scored from the period's END and capped ≤0.5 — so "2023"-dated facts can't ride recency to the top. Missing dates → 0.5 (neutral).

Post-rank `min_scores` floors (`memory_engine.py:9075-9084`): AND-ed `reranker`/`final` thresholds, opt-in, deliberately NO default — comment notes CE absolute scores aren't calibrated (a right answer can score ~0.001 while ranking correctly).

## 5. Entity resolution (`entity_resolver.py`)

Pipeline: intra-batch dedup (trigram ≥ `intrabatch_merge_similarity=0.5`, `_INTRABATCH_MAX_NAMES=250`) → SQL pg_trgm probe at `entity_trgm_similarity_threshold=0.15` (`config.py:1811`, index `entities_canonical_name_lower_trgm_nonlabel_idx`) → hard gates `merge_min_similarity=0.3` trgm + `_tokens_are_compatible` (every word of the shorter name must match the longer via exact/prefix/`SequenceMatcher ≥ _MIN_TOKEN_SIMILARITY=0.6`; single-word exempt; `entity_resolver.py:125-158`) → weighted score:

```
score = 0.5·SequenceMatcher(name_a, name_b)
      + 0.3·(Σ_{matched nearby entities} 1/√degree(e) / |nearby|)   # _cooccurrence_weight, hub damping
      + 0.2·max(0, 1 − days_since_cooccurrence/7)
merge iff round(score, 6) >= 0.6                       # lines 1314-1352
```
`_trigram_set` pads `"  {word} "` and computes Jaccard — verified byte-for-byte vs pg_trgm (81-118). `entity_resolution_max_candidates=200` (`config.py:1818`). Worked: `Dr Patel`~`Dr. Patel` → 0.5×0.941 + 0.3×0.062 + 0.2×1.0 = 0.689 → merge; `Anita`~`Anita Sharma` → 0.493 → keep both; `Alice`~`Alice Johnson` → 0.278 → keep. The token-compat gate is what stops "Anita" ⊂ "Anita Sharma" from merging on substring alone.

PAPER gives `score = α·name + β·cooccur + γ·temporal` with unspecified coefficients — code fills them: **0.5/0.3/0.2, merge ≥0.6**.

## 6. Retain-time links (`retain/link_utils.py`)

- **temporal**: `weight = max(0.3, 1 − Δhours/24)`, window `time_window_hours=24`, `MAX_TEMPORAL_LINKS_PER_UNIT=20`, bidirectional (lines 63,158,459-545).
- **semantic**: ANN kNN `top_k=50`, `similarity ≥ semantic_link_min_similarity=0.7` (573-579, `config.py:1299`); `MAX_SEMANTIC_LINKS_PER_UNIT=50` (`memories/pg/graph.py:53`).
- **causal**: weight 1.0, canonical type `caused_by` at retain, LLM-extracted **max 2 per fact**, target must point backward in input order (`_write_causal_links_batch` ~948-1000, `fact_extraction.py:1334`). Legacy `causes/enables/prevents` only via import.
- `memory_links` schema: `(from_unit_id, to_unit_id, link_type, entity_id, weight∈[0,1])` + unique `(from,to,type,COALESCE(entity_id,0-uuid))`; indexes `(from_unit_id,link_type,weight)`, `(to_unit_id,link_type,weight)`, entity.

## 7. Observations — LLM, not deterministic

Consolidation prompt returns JSON `{creates, updates, deletes}` each with reason (`consolidation/prompts.py:39-172`): UPDATE preferred over CREATE on the same facet; DELETE only when superseded. Separate `_DedupDecision` LLM adjudicates merge/keep for pairs at cosine ≥ `consolidation_dedup_threshold=0.97`, `_DEDUP_TOP_K=5` (`consolidator.py:160-326`, `config.py:1727`). `_duplicate_create_target` drops verbatim-text duplicate CREATEs **deterministically** (137-155). `consolidation_batch_size=50`, recall for dedup uses `reranking="interleave"` + `include_source_facts=True`.

At recall: `prefer_observations` drops raw world/experience facts superseded by top observations (within `thinking_budget×2` window) via `source_memory_ids`/`observation_sources` (`memory_engine.py:9124-9178`).

**⚠ Verbatim rule conflict:** hindsight's LLM CAN update/delete prior units — Verbatim's byte-pin rule forbids update/delete; LLM may only ADD facts with byte-matched quotes. The deterministic parts (verbatim-duplicate drop, supersession dedup at recall, `proof_count = len(source_fact_ids)`) are transplantable; the mutating loop is not.

## 8. Query analysis and budgets

Default `DateparserQueryAnalyzer` (NOT flan-t5): deterministic dateparser + `_query_can_score` cheap gate + `_date_match_score` filtering, `PREFER_DATES_FROM=past`, offloaded to a 1-worker ThreadPoolExecutor — inline call measured 1318ms stall → 2.8ms with the worker (`temporal_extraction.py:38-47`). `TransformerQueryAnalyzer` (flan-t5-small, rules-first + T5 fallback, ~30-80ms CPU) exists opt-in at `query_analyzer.py:469-720`. PAPER's "rules + T5 hybrid parser" describes the non-default option.

Budgets: fixed `low/mid/high` = 100/300/1000 = per-arm SQL LIMIT (`memory_engine.py:1453-1483`); adaptive `clamp(max_tokens×{0.025,0.075,0.25}, 20, 2000)`. `max_tokens` default 4096. Pack: `truncate_to = thinking_budget×2`, then `select_facts_within_budget` skip-and-continue; **floor = top-1 fact even if over budget** — a matched query never returns empty (`fact_budget.py:65-85`).

## 9. Storage layout

PostgreSQL + pgvector (Oracle 23ai alternative dialect). Per-(bank,fact_type) **partial** HNSW cosine indexes `idx_mu_emb_*` (migration `d5e6f7a8b9c0`), GIN `search_vector`, GIN `source_memory_ids` (PG uses `uuid[]` column on memory_units; Oracle gets `observation_sources` junction), `idx_memory_links_from_type_weight/_to_type_weight`, `idx_unit_entities_entity_unit`, pg_trgm GIN `LOWER(canonical_name)` non-label partial, date indexes. `memory_units` columns incl. `event_date, occurred_start/end, mentioned_at, fact_type ∈ {world,bank,opinion,observation}, confidence_score (required iff opinion), access_count`.

Estimated resident bytes per memory (PG): row ~0.5KB + embedding 1.5KB + HNSW ~2-3KB + tsvector/GIN ~0.3KB + links (≤50 sem + ≤20 temp + causal + entity) ~3-6KB + unit_entities/cooccurrences ~0.5KB ≈ **8-12 KB/memory** → ~1 GB at 100k. SQLite-Verbatim without HNSW (hash embeddings don't need ANN at this scale): ~3-5 KB/memory → ~0.5 GB at 100k.

## 10. Per-query complexity → ms on the 4-core reference

SQL roundtrips per recall (PG): 1 (sem+BM25 UNION) + 1-6 temporal (entry + ≤5 BFS batches) + ≤4 parallel graph CTEs + 1 hydrate + ~1-3 enrichment = **~5-10 sequential roundtrips**; then CE ≤300 pairs.

ms model for a Verbatim-shaped port (SQLite, 4-core CPU, mid budget 300/arm):

| Stage | 10k memories | 100k memories | Arithmetic |
|---|---|---|---|
| Query analysis (dateparser) | 2-5 | 2-5 | CPU-bound, size-independent |
| Query embed (bge-small or hash) | 10-15 / ~1 | 10-15 / ~1 | 1 short text |
| Semantic arm | 1-5 | 5-40 | HNSW: ~log N×ef=200 (~1-3ms). Brute hash-cosine: N×384 flops → 100k×384≈38M flops ≈ 10-40ms numpy |
| BM25 arm | 2-8 | 5-20 | match-set bounded by ≤16 lowest-df terms × budget LIMIT |
| Graph arm (1-hop) | 3-10 | 5-15 | ≤20 seeds × ≤200/entity + causal max — bounded regardless of N |
| Temporal arm (when active) | 5-15 | 10-30 | pool 60 + ≤5 BFS iters × 20-batch ≤ ~1k row ops |
| Fusion + hydrate + boosts + pack | 1-3 | 2-5 | ≤4×300 candidates dict ops + 1 fetch |
| **Subtotal, no CE** | **~25-60ms** | **~35-125ms** | matches their tracer's `store_recall` ~33-135ms comments |
| CE rerank, pool 300 (MiniLM-L6 local) | ~1.5-4s | ~1.5-4s | ~5-15ms/pair ×300 (batch 32 → ~10 forward passes) |
| CE rerank, pool 32 | ~0.3-0.8s | ~0.3-0.8s | 1 batch ≈ 32×10ms — still >150ms target → quality-profile only |

The lesson: hindsight's 300-pool is only affordable because they support remote/TEI rerankers; on bare 4-core CPU the CE dominates everything by 1-2 orders of magnitude. For Verbatim's ≤150ms CE target, pool ≈16-32 is the ceiling, or use a much smaller CE (TinyBERT/flashrank ~1-3ms/pair → 32 ≈ 30-100ms).

## 11. Paper vs. code — the deltas that matter

| Paper claim | Code reality |
|---|---|
| Graph arm = multi-hop BFS spreading activation (eq.12, δ·μ(ℓ)) | Graph arm is **single-hop** additive expansion; multi-hop spreading (δ=0.7, μ∈{2.0,1.5,1.0}, frontier>0.2, ≤5 iters) exists ONLY inside the temporal arm |
| BM25 channel | Native backend is `ts_rank_cd` cover-density — no IDF; true BM25 needs vchord/pg_textsearch/pg_search |
| Hybrid rules+T5 temporal parser | Default is pure dateparser; T5 analyzer exists but opt-in |
| Causal expansion "weight + 1.0" | Docstring at `link_expansion_retrieval.py:16` is stale; code uses raw `ml.weight` |
| α,β,γ for entity merge unspecified | 0.5 seq / 0.3 hub-damped cooc / 0.2 7-day recency; merge ≥0.6 |
| LLM never mutates (implied append-only spirit) | Consolidation LLM **can update and delete** units |
| Reranker pool "~32" | Pool = **300**; 32 is the local batch size |

## 12. Candidate → Verbatim stage map and verdicts

| Candidate | Verbatim stage | Verdict | Confidence |
|---|---|---|---|
| Unweighted RRF k=60 + `cap_per_source` off | fusion | **ship** | high — code-verified; replace Verbatim's score-space lane weights with rank-divisor boosts if biasing needed (recall_boost.py precedent) |
| Rank-space boosts `1/(k+rank/divisor)` {2,4,8} + flat additive {0.05,0.2,0.5} | fusion/rerank | **optional** | med — needed only if a lane must be privileged; avoids the lexicographic-collapse trap they measured |
| CE pair = `[q, "[Date: ..] ctx: text"]` + sigmoid + pool≤32 | rerank | **ship (pool≤32) / reject pool=300** | high — input format is the transferable part; 300-pool is 10× the 150ms budget |
| Multiplicative boosts `×(1+α(x−0.5))` α=.2/.2/.1 | rerank/boosts | **ship** | high — bounded ±10-27%, same style Verbatim already runs; proof_norm=0.5+ln(pc)/10 |
| 365-day linear decay floor 0.1 + period-end coarse dating | temporal boost | **ship** | high — one-liner, well-specified; coarse-date cap ≤0.5 is a nice detail |
| Temporal coverage selection (8-bucket round-robin) | temporal lane | **ship** | med-high — cheap Python, solves "answers cluster on dense stretch" |
| Temporal spreading BFS (batch20, ×0.7, μ 2.0/1.5/1.0, >0.2 frontier) | graph/temporal | **optional** | med — real machinery; only worthwhile once memory_links exists; simplified 1-hop variant first |
| Graph 1-hop additive `tanh(0.5·shared_entities)+sem_max+causal_max` | graph lane | **optional** | med — needs entity links; entity part alone (COUNT DISTINCT shared entity → tanh) is the cheap win |
| Retain links: temporal `max(0.3,1−Δh/24)` @24h cap20; semantic kNN50@0.7 | write-path | **optional** | med — write-time cost; temporal links are cheap (date-index scan), semantic needs retain-time ANN |
| Entity resolution: trgm 0.15 probe → gates → 0.5/0.3/0.2 ≥0.6 | entity | **optional** | med — no pg_trgm in SQLite; substitute token-Jaccard probe + same token-compat gate + 0.6 threshold |
| Observations LLM creates/updates/deletes | observations/write | **reject** | high — violates append-only byte-pin; salvage: verbatim-text dup drop, `proof_count=len(sources)`, recall-side supersession dedup |
| `interleave_fusion` for dedup recall | fusion (internal) | **optional** | med — only for a consolidation/dedup path |
| 16-term lowest-df BM25 cap via corpus stats | BM25 lane | **ship** | high — prevents match-set blowup; Verbatim already needs eligible-set df — same fix for the same reason |
| Budget = per-arm LIMIT {100,300,1000} + truncate ×2 + pack floor top-1 | pack | **ship** | high — floor = "never return empty if something matched" |
| `min_scores` post-rank floors (AND-ed, opt-in, no default) | rerank | **ship** | med — cheap escape hatch; their comment explains why fixed cutoffs fail |
| Per-(bank,fact_type) partial indexes | storage | **ship (concept)** | high — maps to per-lane index filtering on eligible set |
| Docstring staleness + propagated scores >1.0 | hygiene | **note** | scores needn't be bounded if fusion is rank-based — same license applies to Verbatim lanes |

UNRESOLVED: none. Every assignment sub-question has a code-verified answer above.