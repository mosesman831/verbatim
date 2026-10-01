# d3-graphiti-code — getzep/graphiti retrieval internals (code-read, rev 16cdf70, v0.30.2, 2026-09-21)

Scope: `graphiti_core/` of github.com/getzep/graphiti, Apache-2.0 (LICENSE + per-file headers; Zep-CLA required for contributors). Everything below is quoted from the checked-out source; file:line refers to that revision. Two things were *measured* on this 4-core box (numpy f32, SQLite :memory:), marked `[measured]`.

## 0. One-paragraph orientation

Graphiti's search is a four-scope hybrid: **edges** (`RELATES_TO`, carry `fact` + `fact_embedding`), **nodes** (`Entity`, `name` + `name_embedding` + `summary`), **episodes** (`Episodic`, raw `content`, BM25 only — no embeddings), **communities** (`Community`, LLM-written `name`/`summary` + `name_embedding`). Each scope runs its own enabled lanes (bm25 / cosine / bfs) in parallel at candidate limit `2*limit` (limit=10 → 20), then applies ONE per-scope reranker — RRF (Σ1/rank), one-pass MMR, cross-encoder, node_distance, or episode_mentions. There is **no cross-scope fusion** — results return as four separate ranked lists (`SearchResults`), and there is **no ANN index anywhere**: the "dense" lane is a brute-force per-row cosine scan in Cypher. License: **Apache-2.0** (confirmed, `LICENSE` head + file headers). Backends: Neo4j (primary), FalkorDB, Kuzu (deprecated), Neptune (OpenSearch for FTS).

## 1. Which stores/indexes the hybrid search uses

Per-provider `driver/` package; the retrieval layer is provider-agnostic Cypher text assembled in `search_utils.py`. Neo4j is not *required* — FalkorDB/Kuzu/Neptune are first-class `GraphProvider`s (`driver.py:59-63`) — but Neo4j is the dev default and the only backend the AGENTS.md notes as reliable under concurrent queries (FalkorDB driver drops connections under the semaphore_gather fan-out). Everything except the literal Cypher strings ports to SQL.

Index inventory (Neo4j, `graph_queries.py:131-139`; FalkorDB `graph_queries.py:98-119`):
- **Fulltext** (Lucene under Neo4j — BM25 since Lucene 6; RediSearch under FalkorDB; Kuzu FTS):
  - `edge_name_and_fact` on `()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]`
  - `node_name_and_summary` on `(n:Entity) ON EACH [n.name, n.summary, n.group_id]`
  - `episode_content` on `(e:Episodic) ON EACH [e.content, e.source, e.source_description, e.group_id]`
  - `community_name` on `(n:Community) ON EACH [n.name, n.group_id]`
- **Range** (Neo4j, `graph_queries.py:54-82`): per-label `uuid`, `group_id`, `name`, `created_at`; edges add `expired_at`, `valid_at`, `invalid_at` indexes — the bi-temporal fields are indexed.
- **Vector index: NONE.** `grep` for `db.index.vector`/HNSW/`CREATE VECTOR` returns nothing. `edge_similarity_search` (`search_utils.py:419-444`) is `MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) … WITH DISTINCT e,n,m, vector.similarity.cosine(e.fact_embedding,$search_vector) AS score WHERE score > $min_score ORDER BY score DESC LIMIT $limit` — a full scan with per-row cosine. Same pattern for nodes (`n.name_embedding`) and communities. → Their dense lane is literally the same operation as a numpy brute-force scan; it ports to SQL/numpy unchanged.

Query construction (`fulltext_query`, `search_utils.py:85-113`): Lucene-escape the raw query (`lucene_sanitize`, `helpers.py:79-113`, escapes all `+-&|!(){}[]^"~*?:\/` chars — note the map also escapes capital `OR/AND/NOT` letters `O,R,N,T,A,D` to prevent them becoming operators), bail to `''` if ≥128 whitespace tokens (`MAX_QUERY_LENGTH=128`), prepend `group_id:"g" OR …` filters, wrap in parens → the Lucene default (disjunctive OR) applies. FalkorDB path (`driver/falkordb/fulltext.py:58-83`) splits on 30 separator chars, drops 33 English stopwords, joins survivors with `|` (RediSearch union) + `@group_id:{…}` tag filter — also purely disjunctive. **No phrase weighting, no MUST clauses, no field weights: one flat OR-BM25 per scope.**

## 2. The fusion function and all constants

`rrf(results, rank_const=1, min_score=0)` — `search_utils.py:1775-1790`:

```
score(d) = Σ_lanes 1 / (rank_i(d) + rank_const) ,  rank_i 0-based, rank_const = 1
        = Σ_lanes 1 / rank(d)                      # rank in 1-based terms
```

**This is NOT Cormack k=60.** `rank_const=1` with a 0-based index makes it the pure harmonic reciprocal-rank sum — equivalent to canonical RRF with k→0. Verified: no caller anywhere overrides `rank_const` (all 10 call sites use the default). Worked example `[measured]`, 3 lanes × 5 candidates — BM25 `[A,B,C,D,E]`, cosine `[C,A,F,G,H]`, BFS `[X,C,Y,B,Z]`:

| candidate | Graphiti Σ1/rank | Cormack k=60 |
|---|---|---|
| C (r3,r1,r2) | **1.833** | **0.0484** |
| A (r1,r2) | 1.500 | 0.0325 |
| X (r1) | 1.000 | 0.0164 |
| B (r2,r4) | 0.750 | 0.0318 |

Headline: **a single lane-#1 scores 1.0 > a rank-5-in-all-3-lanes candidate at 0.6** (`3×1/5`); under k=60 the consensus candidate wins (`3/65=0.0462 > 1/61=0.0164`). Graphiti fusion is therefore strongly *top-heavy* — near "winner-take-lane" — rewarding the best hit of any single lane over broad mid-rank agreement. For Verbatim, whose failure mode is "BM25 had the evidence top-20 but the fusion buried it" (84 of 160 zero-hit questions), this is directly relevant: Σ1/rank protects lane-#1s from dilution. Risk flipped the other way: a noisy weak lane's #1 outranks 3-lane consensus — with 8 lanes Verbatim has more weak lanes than Graphiti's 3.

Other fusion/rerank constants, all from `search_utils.py:64-68` + `search_config.py` + `search_config_recipes.py`:

| constant | value | where |
|---|---|---|
| `DEFAULT_SEARCH_LIMIT` | 10 | search_config.py:29 |
| candidate pool per lane | `2*limit` (=20) | search.py:286,297,309 (every lane call) |
| `RELEVANT_SCHEMA_LIMIT` | 10 | search_utils.py:64 (dedup-path lane cap) |
| `DEFAULT_MIN_SCORE` (cosine lane threshold) | **0.6** | search_utils.py:65 → `WHERE score > $min_score` |
| `reranker_min_score` (post-rerank filter) | 0 | SearchConfig default; filters `score >= min_score` |
| `DEFAULT_MMR_LAMBDA` | 0.5 (Field default) — **but the shipped `COMBINED_HYBRID_SEARCH_MMR` recipe sets `mmr_lambda=1`** | search_utils.py:66; recipes:60,65,76 |
| `MAX_SEARCH_DEPTH` (BFS default) | **3** | search_utils.py:67 → `bfs_max_depth` Field default |
| `MAX_QUERY_LENGTH` | 128 tokens | search_utils.py:68 |
| `SEMAPHORE_LIMIT` (async fan-out) | 20 (env `SEMAPHORE_LIMIT`) | helpers.py:38,127 |
| `MAX_COMMUNITY_BUILD_CONCURRENCY` | 10 | community_operations.py:20 |
| `MAX_SUMMARY_CHARS` | 1000 | text_utils.py:26 |

MMR (`search_utils.py:1901-1939`) — one-pass, non-iterative: `mmr(d) = λ·cos(q,d) + (λ−1)·max_{d'≠d} cos(d,d')` over L2-normalized vectors, then sort desc, keep `≥min_score` (min_score=-2.0 default). Note `max_sim` is over ALL candidates, not already-selected — cheaper than Carbonell-Goldstein iterative but penalizes a candidate for resembling anything in the pool. And λ=1 in the shipped MMR recipe zeroes the diversity term → degenerates to a plain cosine re-sort. `[measured]` on {P:[.9,.1],Q:[.8,.6],R:[.7,.3]}, q=[1,0]: λ=0.5 → P(0.018),R(-0.026),Q(-0.086) — diversity kicks in; λ=1 → pure cos order P,R,Q.

`node_distance_reranker` (`search_utils.py:1793-1852`): NOT a real distance — one Cypher adjacency probe `(center)-[:RELATES_TO]-(candidate)`, score=1 if adjacent else ∞, then rank by `1/score` (adjacent=1.0, non-adjacent=0, center itself pinned at 0.1→score 10). It's a 1-hop boolean boost.

`episode_mentions_reranker` (`search_utils.py:1855-1898`): seed with Σ1/rank RRF, then re-sort descending by `count(MENTIONS edges)` per node — i.e., provenance/proof-count boost, comparable to Verbatim's proof-count boost but implemented as a *reorder*, not a bounded ±α.

Cross-encoder path (edges, `search.py:395-411`): RRF-seed → take top `2*limit` → `cross_encoder.rank(query, facts)` → filter `score ≥ reranker_min_score`. Pools: edges ≤20, episodes ≤limit=10, **nodes/communities uncapped** (full dedup'd union ≤60/≤40). Implementations: `BGE-reranker-v2-m3` local sentence-transformers (~568M params, `bge_reranker_client.py:36`); `gpt-4.1-nano` per-passage boolean logprob trick (`openai_reranker_client.py:60-111` — one API call/passage, `logit_bias{True,False}`, score=P(True)); `gemini-2.5-flash-lite` 0-100 numeric score normalized /100 (no logprobs).

## 3. BFS depth and fan-out

`edge_bfs_search` (`search_utils.py:450-571`), `node_bfs_search` (`785-878`): `MATCH path = (origin {uuid:…})-[:RELATES_TO|MENTIONS*1..{bfs_max_depth}]->(:Entity) UNWIND relationships(path) AS rel … WHERE type(e)='RELATES_TO' RETURN DISTINCT … LIMIT $limit`. So: **depth=3 default, directed outgoing-only, edge-type union, unbounded fan-out** — no per-node/per-hop branching cap; the only bound is the `LIMIT 2*limit` on final DISTINCT edges. Origins: caller-supplied `bfs_origin_node_uuids`, else a two-pass fallback seeds BFS from **all source_node_uuids of the other lanes' hits** (`search.py:332-353`) — realistically ≤40 origins (2 lanes × 20). BFS lane is OFF in the basic `search()` (EDGE_HYBRID_SEARCH_RRF = bm25+cosine only) and ON in `search_()`'s default cross-encoder recipe.

`[measured]` SQLite port, 100k nodes / 1M edges / deḡ=10, 40 origins, depth 3, cap 20: **~34 ms** (≈40×1110 ≈ 44k edge rows touched before DISTINCT+LIMIT). Depth 2 → ~4.4k rows ≈ 3-5 ms. This is the dominant per-query cost of the whole pipeline at scale, and it grows with degree^depth — the fan-out cap is the missing constant Graphiti doesn't have.

## 4. Edge invalidation — bi-temporal semantics

Five datetime fields on `EntityEdge` (`edges.py:265-281`): `created_at` (write time), `expired_at` (transaction-time tombstone), `valid_at`/`invalid_at` (event-time interval the fact held), `reference_time` (the producing episode's `valid_at` — the anchor for resolving "last week" → absolute `valid_at`/`invalid_at` via an LLM `extract_timestamps` call at ingest, edge_operations.py:576-620). **Nothing is ever deleted.**

Write path (`resolve_edges`, edge_operations.py:330-535): exact-normalized in-batch collapse → per-extracted-edge (a) `get_between_nodes` + EDGE_HYBRID_SEARCH_RRF restricted to `edge_uuids` → duplicate candidates, (b) unrestricted EDGE_HYBRID_SEARCH_RRF over the fact text → invalidation candidates, minus overlap → one LLM `dedupe_edges.resolve_edge` returning `duplicate_facts[]` + `contradicted_facts[]` idx → deterministic temporal rule (`resolve_edge_contradictions`, :538-573):

```
skip if   edge.invalid_at ≤ new.valid_at        # already dead before new fact began
       or new.invalid_at ≤ edge.valid_at        # new fact dead before edge began (no overlap)
else if edge.valid_at < new.valid_at:
       edge.invalid_at = new.valid_at
       edge.expired_at = edge.expired_at or now()   # set once
```

Read path: **invalidated edges are NOT auto-excluded.** Default `SearchFilters()` produces no temporal predicates; exclusion only happens if the caller passes `expired_at`/`invalid_at IS NULL` `DateFilter`s (`search_filters.py:149-272`, OR-of-ANDs per field, indexed). The LLM context pack renders `invalid_at or "Present"` (`search_helpers.py:6-21`) — i.e., Graphiti chooses to *surface* contradicted history and let the LLM reason over the timeline, not filter it. For Verbatim's eligibility stage, the obvious port is to treat `expired_at IS NULL AND (invalid_at IS NULL OR invalid_at > asof)` as an admission predicate — Graphiti deliberately does the opposite by default; that's a policy fork to call out, not a bug.

## 5. Community/summary derivation

`build_communities()` — explicit batch call (`graphiti.py:1549`), not automatic per-episode: `remove_communities` → `get_community_clusters` → per `group_id`, build neighbor projection (edge counts) → `label_propagation` (community_operations.py:93-138): each node takes the community of the plurality of its neighbors weighted by edge_count; adopt only if winning weight >1; iterate full-graph sweeps until fixpoint. Then per cluster, `build_community`: **hierarchical pairwise LLM summarization** — summaries paired and merged via `summarize_pair` LLM calls (fan-in halving, odd element carried over) until one remains, ≤1000 chars; then `generate_summary_description` LLM call for the community name; `name_embedding` embedded; HAS_MEMBER edges built. Incremental path (`update_community`, on `add_episode` if `update_communities=True`): node joins the *mode community of its neighbors* (max count of adjacent entities' communities) → `summarize_pair(entity.summary, community.summary)` + regenerate name — 2 LLM calls per joining node.

## 6. Cross-scope packing

`search_results_to_context_string` (`search_helpers.py`): edges → `{fact, valid_at, invalid_at|Present}` JSON, nodes → `{entity_name, summary}`, episodes → content, communities → {name, summary}. No byte budget in OSS code (server-side packs in Zep cloud). The four scopes stay as separate sections.

## 7. Cost model — per-query ms on the 4-core reference box

Measured primitives `[measured]`: brute-force cosine N×1024 f32 = 3.0 ms@10k, 6.6 ms@100k; RRF over 3×60 ids = 0.04 ms; one-pass MMR 60×1024 = 0.07 ms; SQLite 3-hop BFS, 40 origins, 1M edges = 34 ms. Assumed: FTS5 OR-BM25 top-20 ≈ 1-2 ms@10k, 2-5 ms@100k (posting-list merge, verified typical); adjacency/count probes ≈ <1 ms each via covering index.

| component (default `search_()` cross-encoder recipe, edges+nodes+episodes+communities) | @10k | @100k | arithmetic |
|---|---|---|---|
| 4× FTS lanes | ~5 | ~15 | 4 × (1-2 / 2-5) ms |
| 3× brute-force cosine (edges,nodes,comms) | ~9 | ~20 | 3 × (3.0 / 6.6) ms |
| 2× BFS (edges+nodes, depth 3, ~40 origins) | ~8-20 | ~68 | 2 × ~34 ms worst case, deḡ=10 |
| RRF per scope | <0.2 | <0.2 | µs |
| cross-encoder bge-v2-m3, ~20+60+10+40=130 passages × ~150-300ms/pass CPU | 20-40 **s** | same | reject default profile |
| cross-encoder MiniLM-class (~30M), same pool × ~10-25ms | ~1.3-3 s | same | optional (quality/max) only |
| **default-profile total (no CE)** | **~22-35 ms** | **~105 ms** | BFS-dominated tail |

Resident/index size: per edge ≈ fact(~0.3KB) + fact_embedding 4KB + FTS postings ~0.3KB + 5 datetimes + uuid ≈ **~5KB**; per node ≈ ~4.5KB; per episode ≈ text only ~0.5-2KB; per community ≈ ~4.5KB. A 1-episode:5-node:6-edge ratio ⇒ ~40KB/episode, ~4GB/100k-episode corpus — the dense lanes alone pin ~(N_edges+N_nodes)×4KB resident for the scan (410MB per 100k embedded items `[measured]` bytes-scanned). Graphiti leans on the DB to page this; a SQL-only engine must mmap/spill or quantize (int8 → 1KB/item, 4× less bandwidth).

## 8. Portability verdicts (Verbatim stage → recommendation)

| # | candidate | ports to SQL? | stage | verdict |
|---|---|---|---|---|
| 1 | Disjunctive OR-BM25 fulltext lane over {name+fact}/{name+summary}/{content} field-groups; lucene-escaped, ≤128 terms | yes — FTS5 MATCH, one table per scope | lane | **ship** — identical mechanism; validates current lane; note single combined-field indexes ≈ BM25F with uniform field weights — no evidence for fancy field weighting |
| 2 | Fusion Σ1/rank (rank_const=1 ≡ k→0) | trivial | fusion | **optional** — A/B against k=60: it rescues lane-#1s (Verbatim's #1 failure mode) but over-trusts weak lanes; if adopted, consider k≈5-10 as middle ground. Confidence: medium — needs the same LoCoMo harness measurement, mechanism is sound |
| 3 | Brute-force cosine lane, threshold 0.6 | yes — numpy scan, but needs *semantic* embeddings (hash-pinned local tier) | lane | **optional** — useless on blake2b non-semantic hashes; ship only with the neural-tier profile; 3-7 ms is fine |
| 4 | One-pass MMR, λ field-default 0.5 / shipped 1.0 | yes | rerank | **reject** — λ=1 degenerates to cosine re-sort; Graphiti itself doesn't meaningfully use MMR |
| 5 | Cross-encoder rerank over RRF-seeded pool (edges top-20) | yes (local model) | rerank | **optional** — MiniLM-class only, pool ≤20, quality/max profile; bge-v2-m3 (568M) and per-passage-LLM rerankers are 20-40s → reject for default |
| 6 | BFS lane: depth 3, unbounded fan-out, origins=other-lane sources | yes — adjacency table + bounded expansion | lane | **optional** — ship at depth ≤2 + per-hop frontier cap (e.g., 32) or cap origins to top-fused entities; depth-3-unbounded is the tail risk (~34ms@100k measured and grows with deg²) |
| 7 | node_distance (1-hop adjacency bool → 1/dist) | yes | rerank→boost | **optional** — demote to a bounded adjacency boost, it's not a real distance metric |
| 8 | episode_mentions count reorder (proof-count) | yes | rerank | **ship** — as bounded boost (Verbatim already has proof-count; this is independent corroboration of the feature, as a *post-fusion reorder* not just ±α) |
| 9 | Bi-temporal fields + interval-overlap invalidation at write; never delete; expired/invalid edges NOT auto-filtered | yes — plain columns | temporal/write-path | **ship semantics** — `expired_at`, `valid_at`, `invalid_at`, `reference_time` + the overlap rule are pure SQL; fork: exclude at eligibility (Verbatim model) vs surface-with-dates (Graphiti model) |
| 10 | `reference_time` anchoring of relative dates at ingest | yes | write-path | **ship** — directly fixes Verbatim's "unresolved relative time phrases" class: resolve at write-time against episode timestamp, not at query-time |
| 11 | Community layer: label-propagation clusters + pairwise-LLM summary tree + mode-community join | clustering yes; summaries need LLM | lane/write | **optional** — corpus-level precomputed lane; LLM cost ~O(n) calls per rebuild; defer behind quality profile or use non-LLM community text |
| 12 | Per-scope lanes fused independently, no cross-scope ranking; 2×limit overfetch | yes | fusion | **ship** — confirms Verbatim's peer-lane architecture and the 2× overfetch factor (independent corroboration) |
| 13 | Context pack: edge rendered `fact (valid_at → invalid_at|Present)` | yes | pack | **ship** — temporal-interval rendering into the pack is how dates reach the answerer |
| 14 | group_id partition predicate inside EVERY lane (isolation before ranking) | yes | eligibility | **ship** — Graphiti's group_id == Verbatim's eligibility-before-ranking rule; independent confirmation of the principle |
| 15 | OpenAI-logprob / Gemini numeric per-passage rerankers | no — paid API, N calls/query | rerank | **reject** — violates no-API-key default + per-passage cost |
| 16 | Entity dedupe: exact-normalized collapse → LLM judge over retrieved candidates | partially — LLM optional | write-path | **optional** — keep the exact-collapse fast path; LLM judge behind flag; no alias rules to port |

## 9. Answers to the provisional-constants cross-review

- **RRF k=60**: Graphiti ships k→0 (Σ1/rank). k=60 stays the defensible consensus default, but Graphiti is a worked counter-example where the top-heavy variant is intentional and reported-good — flag for A/B. My recommendation: keep k=60 default; test rank_const∈{1,5,10,60} on the LoCoMo harness — the failure-mode analysis (strong lane buried) predicts k≤10 helps.
- **BM25F field weights**: no evidence — Graphiti uses single flat indexes of concatenated field groups (`name+fact` as one indexed unit group actually separate fields in ONE index with uniform weight). Nothing to port.
- **Feature-rerank weights / boost alphas 0.2/0.2/0.1**: Graphiti has *no* post-fusion feature rerank at all — they swap the whole reranker per recipe instead of tuning additive boosts. Neither confirms nor refutes; neutral.
- **PPR hops/edge weights**: no PPR anywhere. Only graph mechanism is unweighted BFS + edge_count plurality in label propagation. Nothing to port; PPR remains unjustified by this codebase.
- **Entity alias rules**: none — entity resolution is LLM semantic dedupe (dedupe_nodes prompt: "if a descriptive label clearly refers to a named entity, treat as duplicates") over retrieval candidates. Non-portable as constants.
- **Cross-encoder pool 32**: Graphiti pools = `2*limit`=20 edges / uncapped ~60 nodes / `limit`=10 episodes. Their edge cap 20 corroborates a ~16-32 window; 32 is fine but they show smaller pools suffice for the reranker path.

## 10. Neo4j-coupled vs portable — summary

- **Purely portable (copy the constants/formulas):** Σ1/rank fusion, MMR formula, 0.6 cosine threshold, 2×limit overfetch, 128-term query cap + Lucene-escape + 33-stopword drop, node_distance adjacency probe, episode_mentions count, 5-field bi-temporal schema + interval-overlap invalidation rule, reference_time anchoring, label-propagation clustering, per-scope pipeline structure, 20-way async fan-out.
- **Neo4j-coupled but SQL-rewritable:** the four fulltext indexes → FTS5 tables; Cypher `*1..3` traversal → adjacency-table bounded BFS; brute-force `vector.similarity.cosine` per row → numpy scan over BLOB embeddings (identical semantics, faster constant); MENTIONS/HAS_MEMBER edge types → relation columns.
- **Not portable / rejected:** per-passage LLM rerankers (OpenAI/Gemini), anything depending on live LLM at query time, Neo4j Lucene scoring internals (use FTS5 bm25()), Zep-cloud context engine internals (closed source anyway).
- **Fusion constants → SQL equivalent:** yes with one asterisk — the *interesting* one is rank_const=1's top-heavy behavior; everything else (min_score gates, pool sizes, depth) maps 1:1.

UNRESOLVED: exact pass@k/f1 numbers for each reranker on a public corpus — Graphiti repo ships no ablation; paper reports only end-to-end DMR/LongMemEval deltas (vendor-reported). Would close with the graphiti test-suite evals run or Zep eval harness (tests/evals is longmemeval scaffolding only).
