# d2-mem0-code — mem0 OSS code read

Source of truth: github.com/mem0ai/mem0 @ `f8082a73` (2026-09-22), PyPI `mem0ai` **2.1.0**, Apache-2.0 (`LICENSE` line 1-3). All file:line refs below are this commit. The repo was read directly; docs (`docs/`) ship in-repo so doc claims are the vendor's own.

**Critical context:** this is the *v3* pipeline. Since ~April 2026 (README) mem0 OSS silently swapped from the published-paper architecture (two LLM calls + UPDATE/DELETE adjudication + Neo4j graph) to an **additive, single-LLM-call, hybrid-scoring** pipeline ported from the managed platform (`mem0/configs/prompts.py:975` — "Ported from platform/backend/shared/core/utils/prompt_builder.py"). Design notes for the *old* engine are obsolete; everything below is the new one.

---

## 1. What the OSS code actually does

### 1.1 Add path — `Memory._add_to_vector_store` (`mem0/memory/main.py:879-1206`)

Eight phases, exactly **one** LLM call when `infer=True`:

- **Phase 0** — session scope `{user,agent,run}_id`; reads last **10** messages from SQLite `messages` table (window hard-capped at 10, `storage.py:279-287`).
- **Phase 1** — embeds the concatenated messages, `vector_store.search(top_k=10, filters=scope)` → up to 10 semantically-nearest existing memories rendered into the prompt for dedup, with UUID→`"0","1"…` integer remapping ("anti-hallucination", `main.py:933-938`).
- **Phase 2** — single `llm.generate_response` with `ADDITIVE_EXTRACTION_PROMPT` (+`AGENT_CONTEXT_SUFFIX` when agent-scoped; +`custom_instructions`). Output `{"memory":[{"id","text","linked_memory_ids"?,"attributed_to"?}]}`. **ADD-only**: no UPDATE/DELETE events exist anymore (`add()` returns `event:"ADD"` only). `linked_memory_ids` — the platform's supersession chain — is parsed but **never written to any payload in OSS** (verify: `mem_metadata` at `main.py:1029-1039` keeps only `attributed_to`; the client SDK docs `client/main.py:470` describe supersession "the v3 linked_memory_ids chain" as platform behavior).
- **Phase 3** — `embed_batch(mem_texts)`.
- **Phase 4-5** — dedup: `md5(text)` rejected if in (top-10 existing hashes ∪ batch-seen). Exact-match only; near-duplicates survive.
- **Phase 6** — batch `vector_store.insert` + SQLite `history` ADD rows.
- **Phase 7** — `extract_entities_batch` (spaCy `en_core_web_sm`, NER+heuristics → types PROPER/QUOTED/TOPIC/IDENTIFIER, `utils/entity_extraction.py:751`) → per-entity upsert into sidecar collection `{coll}_entities` (`main.py:422`). Dedup rule: normalized-text exact match OR semantic score **≥0.95** (`main.py:1152`); match → append `memory_id` to `linked_memory_ids`, else new row.
- `infer=False` → raw message embed+insert, no LLM.

Update/delete paths exist as public methods (`update()` rewrites vector+payload and re-links entities; `delete()` strips the id from entity links). But **nothing inside `add()` updates or deletes** — superseded facts coexist; correctness at read time is delegated to ranking.

### 1.2 Retrieval path — `search` → `_search_vector_store` (`main.py:1379,1628`)

```
internal_limit = max(top_k*4, 60)                      # main.py:1641
1. query_lemma = lemmatize_for_bm25(query)             # spaCy, optional
2. query_entities = extract_entities(query)            # spaCy NER, ≤8 used
3. dense  = vector_store.search(vectors=embed(query), top_k=internal_limit, filters)   # ONLY candidate source
4. sparse = vector_store.keyword_search(query_lemma, top_k=internal_limit, filters)    # scores only
5. bm25_scores[id] = sigmoid(raw_bm25)                 # scoring.py:54
6. entity_boosts[id] = max_e sim_e*0.5*1/(1+0.001*(n_e-1)^2)  # main.py:1791-1808
7. drop candidates with semantic < threshold (0.1)     # scoring.py:111
8. score = (semantic + bm25 + entity) / max_possible; sort; top_k   # scoring.py:118-119
9. optional reranker.rerank(query, top_k_results, top_k) # main.py:1502 — final list only
```

Key structural facts:

- **Dense ANN is the sole recall mechanism.** BM25 and entities re-score the dense pool but cannot inject candidates (`docs/migration/oss-v2-to-v3.mdx`: "BM25 is a boost signal, not a recall expander"). A pure lexical hit with semantic <0.1 is dropped.
- Index: default `qdrant` provider, local embedded `QdrantClient(path="/tmp/qdrant")`, collection `mem0`, dense HNSW `Distance.COSINE`, `vector_size=1536` (`configs/vector_stores/qdrant.py:11-16`); a **named sparse vector `bm25`** (fastembed `Qdrant/bm25` ONNX encoder, `Modifier.IDF`) is stored per point — the "BM25" is a learned sparse embedding, not classical BM25 (qdrant.py:93-163). 15 of 26 vector stores implement `keyword_search` (pgvector uses `to_tsvector('simple', text_lemmatized)` + `ts_rank_cd`, pgvector.py:389-391; elasticsearch/redis/milvus native FTS); the rest degrade to semantic-only.
- Entity sidecar = a *second vector-store collection* holding `{data: entity_text, entity_type, linked_memory_ids[]}` — flat entity→memory bipartite links, not a traversable graph. Query side: ≤8 deduped query entities, `embed_batch` (a **second embedding call** per search), `entity_store.search(top_k=500)` per entity on a 4-thread pool, gate `sim ≥ 0.5`.
- Reranker: off by default (`rerank=False`); providers cohere (default model `rerank-v3.5`), huggingface (`BAAI/bge-reranker-base`), sentence_transformer (`cross-encoder/ms-marco-MiniLM-L-6-v2`), llm (`gpt-5-mini`), zero_entropy (`zerank-1`). Reranks **only the final top_k list**, not the internal pool.
- No temporal/recency/decay term anywhere in `score_and_rank` — `timestamp`/`reference_date`/`decay` params raise `ValueError` "not supported by the OSS Memory SDK" (`notices.py:130-134`). Only `expiration_date` (hide past-date memories) exists.
- History DB = plain sqlite3 file `~/.mem0/history.db` (tables `history`, `messages`), NOT the vector index.

### 1.3 Constants (verified in code)

| Constant | Value | Where |
|---|---|---|
| `search.top_k` default | 20 (was 100 in v2) | main.py:1383 |
| `search.threshold` default | 0.1 on **semantic score only** | main.py:1385, scoring.py:111 |
| candidate pool | `max(top_k*4, 60)` | main.py:1641 |
| entity sim gate | ≥0.5 | main.py:1793 |
| `ENTITY_BOOST_WEIGHT` | 0.5 | scoring.py:57 |
| linked-count discount | `1/(1+0.001*(n-1)^2)` | main.py:1802 |
| entity dedup | text-exact OR sim≥0.95 | main.py:621,1152 |
| entity search breadth | top_k=500/entity, ≤8 entities, 4 threads | main.py:1747,1775,1778 |
| BM25 sigmoid params | terms≤3:(5,0.7); ≤6:(7,0.6); ≤9:(9,0.5); ≤15:(10,0.5); else:(12,0.5) | scoring.py:31-40 |
| add-context | top-10 existing memories, last-10 messages | main.py:920,929 |
| upsell triggers | top_k≥50; ≥5 deletes; >2s query | notices.py:123,117,127 |
| defaults | LLM gpt-5-mini; embed text-embedding-3-small/1536d; reranker none | llms/openai.py:40; embeddings/openai.py:15-19 |

### 1.4 Storage backends

26 vector stores (`vector_stores/configs.py:13-38`: qdrant, chroma, pgvector, pinecone, mongodb, milvus, baidu, cassandra, neptune, upstash, azure_ai_search, azure_mysql, redis, valkey, databricks, elasticsearch, vertex_ai_vector_search, opensearch, supabase, weaviate, faiss, langchain, s3_vectors, turbopuffer, oracledb — count the dict: 26). 24 LLM, 15 embedder, 5 reranker providers. **0 graph stores** — `mem0/graphs/` deleted (~4000 lines removed per migration doc); Neo4j/Memgraph/Kuzu/AGE are platform-side now. SQLite for history/messages. `server/` = FastAPI wrapper over `Memory` + Docker Compose (pgvector+Neo4j compose file is stale).

---

## 2. Candidate mechanisms — formulas, cost, verdicts

Time estimates on the 4-core reference. Per-candidate Python fusion work is measured here: `score_and_rank` on an 80-item pool = **0.033 ms** (my run); 60 sigmoid evals <0.01 ms; 10×MD5 0.005 ms. spaCy `en_core_web_sm` NER ~10-25 ms/query-sentence CPU (published throughput ~1-4k tokens/s with NER; marked estimate — I did not install it). Qdrant-local HNSW ~1-10 ms @10k-100k; sparse query similar. OpenAI embed calls ~150-400 ms network each — mem0 OSS makes **two** per query (embed + embed_batch for entities), so default OSS search is network-dominated; irrelevant to Verbatim's local encoder except as proof a second embed call is acceptable overhead at their scale.

### A. Additive normalized-sum fusion (replaces/augments Verbatim fusion)

`combined = min((semantic + bm25_norm + entity_boost) / max_possible, 1.0)`, `max_possible = 1.0 + (1.0 if bm25_scores else 0) + (0.5 if entity_boosts else 0)` — per-query divisor, not per-item. Effective weights: semantic 40%, bm25 40%, entity 20% when all live. Threshold gates semantic **pre-fusion**.

*Worked example* (my run): 6 candidates, 5-term query → sigmoid (7,0.6). m2 (sem .85, bm25_raw 3.0→.083, no ent): .933/2.5=.373 → ranked last. m3 (sem .78, bm25 .646, ent .217): 1.643/2.5=.657 → first. So signals don't just add — the shared divisor *penalizes* candidates that don't fire every signal. Contrast with RRF where missing a lane costs rank-position but never score-mass.

*Cost:* O(pool) — 0.033 ms measured at pool 80; at 100k memories pool is still 60-80 → **<0.1 ms** at both scales. *Bytes:* zero index. *Verdict:* **optional** — worth an A/B against RRF k=60; it's a pointwise-mass fusion vs rank fusion, and the semantic-pre-gate makes it *more* brittle to a weak dense lane than Verbatim's lane-union. Confidence: medium — would flip to ship if Verbatim's dense lane is strong post-fix; what changes it: if lexical-only evidence keeps surfacing (BM25's 84/160 rescue cases), keep RRF union semantics and borrow only the sigmoid normalization.

### B. Sigmoid BM25 normalization with query-adaptive (midpoint, steepness)

`1/(1+e^(-s*(raw-m)))`, params from lemmatized term count (table above). Replaces min-max normalization — no per-query max needed, no tie-collapse when top hit ≈ next (directly fixes Verbatim's "min-max fusion tie-collapse" bug). *Worked:* raw 9.5 → .818 at (7,.6); raw 3.0 → .083. *Cost:* ~0.001 ms/query. *Verdict:* **ship** — cheap, parameter-free-ish, monotone; the five (mid,steep) buckets are vendor-tuned, reasonable priors for FTS5 bm25() raw scores after verifying Verbatim's score distribution is in the same ~0-20 range. Confidence: medium-high — recalibrate midpoints if FTS5 raw scores land lower (FTS5 bm25 returns negative-ish small scores; may need |score| or different mids).

### C. BM25-as-boost-only, dense-only candidate generation

*Verdict:* **reject for Verbatim.** Their own eval doc says exact/factual queries lean on BM25 as *primary* signal — the stance is internally inconsistent. Measured Verbatim data already shows BM25 wins head-to-head (any@20 .677 vs .546); making lexical a no-candidate lane throws away the lane-union that rescues 84/160 of Verbatim's misses. Confidence: high — this is the single clearest negative result in the codebase.

### D. Entity sidecar index + bounded entity boost

Mechanism: second index `{entity_text, linked_memory_ids[]}`; write-time dedup exact-lower OR sim≥0.95; query: ≤8 entities, breadth 500/entity, gate 0.5, `boost = sim*0.5/(1+0.001*(n-1)²)`, max-over-entities. The quadratic linked-count discount is a hub-suppressor: n=30→.54, n=50→.29, n=100→.09 (measured curve above) — popular entities nearly vanish. Portable to SQLite as a plain table: `entities(text_norm PK, type, linked_ids BLOB)` — exact-match dedup needs no embedding at all; the 0.95-semantic alias merge could be a Verbatim fuzzy-lane lookup instead. *Cost:* ≤8 index probes ~1-5 ms total at both 10k/100k (entity count ≈ memories×0.3). *Bytes:* ~40-80 B/entity + 16 B×links ≈ 5-15 MB/100k memories — trivially small. *Verdict:* **ship the mechanism** — it's a principled recipe for Verbatim's entity lane/alias rules (exact-norm first, fuzzy second, cap fan-in, hub-discount quadratic not linear). Constants 0.95/0.5/0.5/500/8 are vendor priors, not physics. Confidence: high on mechanism, medium on constants — sweep boost cap 0.25-0.75 and gate 0.4-0.6 on Verbatim's corpus.

### E. Pool sizing `max(top_k*4, 60)` + rerank only final top_k

*Verdict:* **ship** — validates Verbatim's 40-deep sub-lane caps are in the right order; mem0 uses ~60-80 fusion pool and reranks ≤20, so Verbatim's 32-deep cross-encoder pool is *larger* than mem0's effective rerank set. Confidence: medium-high.

### F. Platform decay multiplier (documented, not in OSS code)

Per `docs/platform/features/memory-decay.mdx`: `score *= factor ∈ [0.3, 1.5]` from touch-history (last-access + count, capped 20 touches); pool widened to `top_k*3` floor 50 before rescoring; fire-and-forget reinforcement write on hits; fallback activation from event_date/updated_at. *Verdict:* **optional (quality/max profile)** — mechanism is a bounded post-fusion boost exactly like Verbatim's, but the band is 5× wider than ±10% and requires persisted touch-history (write amplification on every search). What changes it: if Verbatim's stale-fact problem shows up in evals, try factor `1/(1+α*days_idle)` clamped to [0.5,1.3] first at α≈0.05-0.1 rather than adopting 0.3-1.5 blind. Confidence: medium — constants from docs only, no code/ablation public.

### G. Platform temporal reasoning (documented, not in OSS)

Extraction writes event-date metadata (when, ongoing/completed, precision, memory-type); query temporal intent classified "with no extra LLM call"; additive boost, semantic dominates. *Verdict:* **optional** — mechanism only; no constants published. Relevant to Verbatim's "unresolved relative time phrases" bug: the fix is write-time date normalization (their prompt resolves "last week"→absolute dates against Observation Date — the OSS prompt builder has this machinery but `add()` never passes a timestamp, so OSS grounds everything to *today*).

### H. ADD-only write path (already adopted)

Single-call additive extraction + MD5 dedup + top-10 dedup-context + last-10 window. Aligns with Verbatim's "LLM may only add facts." Mem0 chose to *delete* the UPDATE/DELETE adjudication pass even on the platform — strong evidence the second call wasn't paying rent (their claim: ~half the extraction latency, +20 LoCoMo). *Verdict:* **ship (concept already aligned)** — the novel bit for Verbatim: dedup-context = top-10 semantic neighbors shown to the writer + cheap MD5 exact-hash as a second belt. Confidence: high.

### I. fastembed `Qdrant/bm25` sparse embeddings — **reject**: ONNX model download, weaker than real BM25 for this purpose, FTS5 `bm25()` already in Verbatim.

### J. spaCy `en_core_web_sm` lemmatizer+NER — **optional (quality profile only)**: ~12 MB download violates the no-model default; lemma-with-ing-fallback trick (`lemmatization.py:45-48` — append original `-ing` forms alongside lemmas to dodge noun/verb ambiguity) is portable logic if Verbatim ever does light stemming. Confidence: high.

### K. Legacy two-call UPDATE/DELETE adjudication — **reject**: removed by vendor upstream.

---

## 3. OSS vs platform quality — the honest split

- **arXiv 2504.19413** (Apr 2025): mem0 J = 67.13/51.15/72.93/55.51 (single/multi/open/temporal); mem0g 65.71/47.19/75.71/58.13. Issue #2800's maintainer reply confirms those numbers were generated on the **platform**, not OSS: "on the platform we have made some improvements in terms of addition and search" (Contextual ADD + custom instructions).
- **Current docs numbers** (LoCoMo 92.5, LongMemEval 94.4, BEAM 64.1/48.6 @ top_200 budget, ~7k tokens/query): `docs/core-concepts/memory-evaluation.mdx` states outright — "Scores reflect Mem0's managed platform, which includes proprietary optimizations not available in the open-source SDK." The migration doc's 91.6 vs eval-page 92.5 discrepancy is just re-run drift; both are platform.
- **Independent reproductions** (old pipeline): issue #2800 — ~0.486 mean J vs paper 0.67; issue #3943 — "30-50% locally vs 60%+". No independent v3 OSS reproduction found.
- **What the platform adds over OSS today** (`docs/platform/platform-vs-oss.mdx`): native graph memory (co-occurrence graph → score boost, *not* typed relations — "connections are inferred from co-occurrence"), memory decay, temporal reasoning, Dream (supersede/merge always-on + synthesis Pro), `timestamp`/`reference_date`, app_id scoping, webhooks, batch ops, custom categories, feedback, summaries. The core loop (additive extraction, hybrid semantic+BM25+entity scoring) is shared — OSS v3 *is* a port of the platform recipe minus those features.
- So: **headline quality ≈ platform.** Expect OSS v3 to land between the old ~0.5 repro and platform 0.925 — the missing pieces are exactly the temporal/graph/decay/dream features above. Vendor's own hedge: "directionally similar gains but not identical numbers."

## 4. What ports to SQLite (Verbatim-shaped answer)

Fully portable as-is: `score_and_rank` (30 lines over a dict pool), sigmoid normalization, entity boost math, `max(4k,60)` pool, MD5 dedup, last-10 window, entity sidecar as a btree table. Portable in mechanism, not constants: decay multiplier, temporal boost, entity dedup thresholds. Not portable/not needed: fastembed sparse ONNX, spaCy model, HNSW vectors, Qdrant, Neo4j. **Nothing in mem0 OSS requires an ANN index** — the retrieval contributions Verbatim can use are all query-time re-scoring of a bounded pool, which is exactly Verbatim's fusion stage.

## 5. Byte cost if Verbatim adopted the OSS stack verbatim (rejected, for scale intuition)

Dense 1536d fp32 = 6,144 B + HNSW ≈128 B/node + payload ~300-600 B + sparse ~100-200 B ≈ **~7 KB/memory → ~700 MB @100k** (plus entity collection ~30k×~6.5 KB ≈ 190 MB). The SQLite-adapted versions of candidates A/B/D cost <50 MB/100k total — 20× cheaper because Verbatim needs no dense vectors for these mechanisms.

## 6. Provisional-constant cross-check for the Verbatim synthesis

RRF k=60: fine, no contradicting evidence (mem0 doesn't use RRF). Semantic-only pre-gate threshold 0.1 cosine: actively harmful if adopted (see C). Entity alias rules: adopt mem0's two-tier (exact-normalized → fuzzy≥0.95-equivalent). Boost alphas 0.2/0.2/0.1: mem0's effective post-normalization weights (40/40/20%) are ~2× stronger than Verbatim's ±5-10% bounded boosts — mem0's signals *compete* for score mass rather than nudge. Cross-encoder pool 32: consistent with mem0 (reranks ≤20) — keep.

*UNRESOLVED:* exact OSS-v3-vs-platform delta on LoCoMo — no public reproduction exists yet; closable by running memory-benchmarks' `--backend oss` vs `cloud` on the same corpus. Also unresolved: whether mem0's sigmoid midpoints are calibrated to `ts_rank_cd` or fastembed scores (code accepts either; midpoint table is provider-agnostic, so treat midpoints as priors to recalibrate on FTS5 score distributions).
