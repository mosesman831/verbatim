# a2-mem0 — Mem0 managed platform vs OSS retrieval

Scope note: numbers below are labeled **[VENDOR]** (mem0.ai docs/blog/paper), **[INDEP]** (Maximem reproduction), or **[CODE]** (read directly in github.com/mem0ai/mem0 @ f8082a7, 2026-09-22). The repo's `evaluation/` is a submodule of mem0ai/memory-benchmarks — the eval harness Maximem audited.

## 1. What the managed platform does that OSS does not

| Capability | Platform v3 | OSS SDK (`mem0ai`) |
|---|---|---|
| Extraction | Single LLM call, ADD-only; async temporal enrichment writes event_start/end, memory type, state_key later | Same single-pass ADD-only prompt ported into OSS (`ADDITIVE_EXTRACTION_PROMPT`, `generate_additive_extraction_prompt`), sync LLM call; no temporal pass |
| Entity graph | Built-in, always on: entity nodes + co-occurrence edges (no typed relations; 'connections inferred from co-occurrence'), feeds a ranking boost | Entity *store* only (separate vector collection, `linked_memory_ids` payload); no graph traversal at all — old `enable_graph`/Neo4j path REMOVED in v3 |
| Retrieval signals | semantic + keyword + entity + temporal; 'fused via rank scoring' (constants undisclosed) | cosine + sigmoid-BM25 + entity boost, additive normalize (formula below) |
| Temporal reasoning | Query intent classified (7 modes, no LLM call), additive post-retrieval rerank | Not present; passing `reference_date` raises `ValueError` |
| Memory decay | Per-project toggle, scaling factor in [0.3, 1.5] on top of final score | `decay=True` param exists but only prints 'not supported by OSS' notice |
| Reranker | Managed catalog, +150–200 ms, `rerank=True` (v3 default: OFF) | Pluggable: cohere / huggingface / llm / sentence_transformer / zero_entropy; applies to the truncated top_k only |
| Filters | AND/OR/NOT + comparisons | Same operator set, translated per-store |
| Conversation context on add | Platform auto-pulls prior turns for the same ids | OSS pulls `get_last_messages(session_scope, limit=10)` locally |

Paper-era architecture (arXiv 2504.19413, the OLD two-pass algorithm, now superseded by v3): extract with context (summary S + last m=10 messages + new pair) → for each fact, retrieve top s=10 similar memories → LLM tool-call chooses ADD/UPDATE/DELETE/NOOP. Mem0g added LLM entity-type extraction + LLM triplet (vs,r,vd) generation, cosine node-match over threshold t, LLM conflict-resolver marking edges invalid (never deleting), Neo4j store, dual retrieval (entity-anchor subgraph expansion + whole-query-vs-triplet embedding match). All LLM ops on GPT-4o-mini. **This v1 design is what the famous OSS code did until v3; v3's ADD-only, no-delete pipeline exists precisely because UPDATE/DELETE 'was where context got destroyed'** — independent confirmation of Verbatim's never-overwrite rule.

## 2. Recency / 'Memory Decay' function — exact state of disclosure

**The closed-form is not published.** Published constraints [VENDOR, docs/blog]:

- Every memory stores its last **<=20 access timestamps** (retrieval hits count as accesses).
- Factor `d ∈ [0.3, 1.5]` multiplies the post-fusion score. Anchors: just accessed ≈ 1.5x · touched today 1.2–1.4x · idle days 0.6–1.0x · idle weeks 0.4–0.6x · months+ ≈ 0.3x floor.
- Pool widens to `top_k × 3`, floor 50, before scaling; sort on unclamped product; public `score` re-clamped to [0,1]; truncate to top_k.
- Reinforcement write is fire-and-forget on a bounded executor (no read-path cost).
- Legacy fallback: first touch = `event_date` if present (future event_date → ≈1.0–1.5x 'fresh/full activation' — docs are slightly inconsistent between 1.0x and full), else `updated_at`.
- Threshold filtering happens BEFORE scaling (a dampened result can return score < threshold; intentional).
- Vendor FAQ: band is 'calibrated to be conservative: wide enough to meaningfully reorder candidates, narrow enough to never dominate' — the 'wide decay' in the task description refers to this wide-but-bounded band, not a named formula.

The mechanism (multi-timestamp access history → bounded activation → score multiplier) is ACT-R base-level-activation-shaped: canonical reconstruction `A = Σ_j (t_now − t_j)^(−d)`, j ≤ 20 recent touches, mapped through a bounded squash to [0.3, 1.5]. A Verbatim-compatible instance consistent with every published anchor:

```
scale(m) = clamp( 0.3 + 1.2 * sigmoid( ln( Σ_j exp(−Δt_j / τ) ) ), 0.3, 1.5 )
   Δt_j = days since access j;  j ≤ 20 stored stamps;  τ ≈ 30 days (tune)
   cold-start: single pseudo-touch at event_date else updated_at
```
Cheaper sufficient form that hits the same anchors with 2 integers per memory (last_access_at, access_count):
`scale = clamp(0.3 + 1.2 * (0.65*exp(−days_since_last_access/τ1) + 0.55*tanh(count/5) * exp(−days_since_last_access/τ2)), 0.3, 1.5)` — τ1≈7d, τ2≈45d gives ≈1.5 fresh, ~1.0 at ~1wk, ~0.5 at weeks, →0.3 asymptote.

Verbatim stage: post-fusion bounded multiplier (recency lane/boost).
Per-query cost: O(pool) — <1 ms at 10k/100k memories; storage ≤20·8B+8B ≈ 168 B/memory (or 16 B with the 2-int variant).
**Verdict: optional (quality/max), default OFF.** Vendor itself makes it opt-in and its own example shows an evergreen allergy fact dropping out of top-5 after a trivia article got reinforced; vendor LongMemEval table shows knowledge-update −2.6pts under the temporal+decay build. Decay fights 'evergreen evidence' questions — keep it out of the default profile.

## 3. Extraction: what gets stored vs discarded

Single LLM call per `add`; JSON `{"memory":[{id,text,attributed_to,linked_memory_ids}]}`; memory text is LLM-GENERATED prose (NOT byte-pinned — divergent from Verbatim).

Stored [CODE, prompts.py L468–947]: personal facts, preferences, plans, relationships, health, opinions, hobbies, emotional states, entity attributes (breed/model/color), implicit preferences inside requests, facts inside shared documents/images, transitions captured as old→new pairs ('switched from almond to oat milk after almond sensitivity'), assistant-generated specifics (recommendations, plans, researched info, agreements) and named third-party speakers' personal facts; 15–80 words, self-contained (pronouns resolved to names), temporally grounded via Observation Date ('last week'→absolute week), numerically exact, proper nouns/titles preserved verbatim, qualifiers preserved ('assistant manager' ≠ 'manager').

Discarded: greetings/filler/phatic turns; assistant echoes of user statements (extract once from user's version); vague assistant characterizations unless user confirms; assistant meta-commentary; 'meta-extraction' ('User asked X to be shortened' — extract the content instead); implicit-attribute inferences (no gender/age/ethnicity from names); details imported from existing memories into a new extraction (no context contamination); semantically equivalent facts already captured (dedup); within-response duplicates (keep richer of two).

Dedup [CODE]: md5(normalized extracted text) against hashes of top-10 semantically-similar existing memories + intra-batch set. Entities per memory: spaCy en_core_web_sm NER ∪ technical-identifier regex ∪ proper-name ∪ quoted-string ∪ topic-phrase extraction; normalized = lower/strip/collapse-ws; merged into a global entity when exact-normalized match OR embedding sim ≥0.95.

For Verbatim: the *rule list* transfers to an optional LLM tier verbatim-ish (adapt: generated text must quote pinned source spans); the ADD-only + linked_memory_ids decision is already Verbatim's rule — mem0's v3 migration is a second data point that dropping UPDATE/DELETE improved accuracy AND halved extraction latency. **Verdict: ship (as extraction-policy spec for the optional LLM tier; default path unchanged).**

## 4. Scores: vendor vs independent

| Benchmark | Score | Harness | Label |
|---|---|---|---|
| LoCoMo overall | **92.5%** (cat 1–4, 1,540q; single-hop 94.6 / multi-hop 95.4 / open-domain 82.3 / temporal 92.5) | mem0 platform v3 + temporal + decay, top_200, ~6,956 tok/query, answer/judge model **never disclosed** | VENDOR |
| LongMemEval overall | **94.4%** (500q; temporal 97.0 / multi-session 88.0 / knowledge-update 93.6 ↓2.6) | same, ~6,787 tok/query | VENDOR |
| BEAM | 64.1 @1M tokens · 48.6 @10M | vendor | VENDOR |
| LoCoMo (v3 base, pre-temporal/decay) | 91.6–92.5 (two vendor posts disagree on whether April build was 91.6 or 92.5) | VENDOR |
| LongMemEval v3 base | 90.4–93.4 (posts disagree) | VENDOR |
| LoCoMo paper-era (2025) | J = 66.88% mem0 / 68.44% mem0g; best RAG ~61; full-context ~73 (26k tok, p95 17.1s) | VENDOR (paper, GPT-4o-mini stack) |
| LongMemEval | **57.5%** → **73.8%** across the April-14 update (gap to 93.4 claim: −19.6) | Maximem open harness: paid hosted product, gpt-5 answerer, binary judge w/ explicit CORRECT+WRONG conditions, 5-seed mean | INDEP (competitor-run; harness open-sourced) |
| LongMemEval Supermemory | 71.3 obs vs 85.2 claim (−13.9); own 3-model sweep: 81.6 gpt-4o / 84.6 gpt-5 / 85.2 Gemini-3P (+3.6 across 2 frontier generations, memory layer constant) | INDEP |
| Zep | 71.2 vendor-published; third-party report 63.8 | INDEP (not rerun) |

Maximem's attributed causes of the mem0 gap (each pinned to commit+line, SHA-256, Wayback): **14 dataset-specific equivalence rules in the answer prompt mapping 1:1 onto public LongMemEval question ids; a hidden CoT block applied before the visible answer the judge sees; a 'lean toward yes' judge instruction with a 5-step gauntlet only before marking WRONG (none before CORRECT); a one-directional gold-override that can promote wrong→correct but not demote.** Caveat: Maximem is itself a vendor (Synap 92.0/93.2 with suspiciously flat 100% categories) — treat its absolute numbers as directional; the claimed-vs-observed *gap* structure and the pinned eval-prompt diffs are the reliable part.

So-what for Verbatim: (a) never import benchmark equivalence rules or judge-shaping prompts — same class of artifact the task rules already forbid; (b) real memory-layer progress between the two Maximem runs is +16.3 pts — vendor posts are roughly self-consistent about the *delta* even if absolute claims don't reproduce; (c) ~7k tokens/query is the vendor's own context-pack budget — a sane sanity ceiling for Verbatim's byte budget.

## 5. OSS code constants & the fusion formula

`mem0/memory/main.py::_search_vector_store` (L1628) + `mem0/utils/scoring.py`:

```
internal_limit = max(4 * top_k, 60)                     # over-fetch, both lanes
semantic_candidates = vector_store.search(top_k=internal_limit, filters)
kw_hits = vector_store.keyword_search(lemmatize(q), top_k=internal_limit)
                                                      # ES: multi_match BM25; Qdrant: sparse bm25; pgvector: ts_rank_cd on text_lemmatized — NOT uniform BM25!
bm25_norm(x) = 1 / (1 + exp(−s·(x − m)))               # sigmoid normalization
  (m, s) by lemmatized query term count: ≤3 → (5.0,0.7); ≤6 → (7.0,0.6);
  ≤9 → (9.0,0.5); ≤15 → (10.0,0.5); else (12.0,0.5)    # calibrated to ES-scale raw scores
entity_boost(m) = max over query entities e of
     sim(qe, entity_e) * 0.5 * 1/(1 + 0.001·(n_linked_e − 1)^2)
  where: query entities deduped, ≤8; entity_store.search top_k=500, sim ≥ 0.5;
         4-thread pool; ENTITY_BOOST_WEIGHT = 0.5
combined = (cos + bm25_norm + entity_boost) / max_possible
  max_possible = 1 + 1{bm25_active} + 0.5·1{entity_active}   # adaptive divisor
  gate: cos < threshold (default 0.1) → dropped BEFORE combining
```

Key structural facts: **the candidate set is built ONLY from semantic results** — a keyword-only or entity-only hit that isn't in the semantic top-max(4k,60) cannot be returned at all; BM25 and entity are rescoring signals, not lanes. Reranker (if on) sees only the already-truncated top_k list. Filters/eligibility (user_id required; expiration_date suppressed unless show_expired) are applied inside store queries before scoring — consistent with eligibility-before-ranking.

Write path (`_add_to_vector_store`): last-10 session messages → top-10 existing memories (UUID→int alias for anti-hallucination) → 1 LLM call → batch embed → md5 dedup → batch insert + history rows → entity batch link (exact or sim≥0.95 merge) → raw messages saved. `PAST_MESSAGE_TRUNCATION_LIMIT=300` chars/message in prompt.

Per-query cost, 4-core, SQLite-equivalent scale: ~9 embeds worst case (1 query + ≤8 entities) + 1 vector search top-60 + 1 keyword search top-60 + ≤8 entity lookups(top-500 each) + O(60) combine → with a local embedder ≈ 30–80 ms at 10k–100k — same order as Verbatim's measured 46/85 ms.

## 6. Candidate formulas → Verbatim

1. **Sigmoid BM25 normalization** `n(x)=1/(1+exp(−s(x−m)))`, m,s query-length-adaptive (values above) — stage: lexical lane scoring before fusion; <1 ms; retune (m,s) against SQLite FTS5 `bm25()` magnitude (negative, lower=better — mem0's constants assume ES-style positive scores) — **ship**.
2. **Adaptive-divisor additive fusion** `(sem + bm + ent)/(1+1{bm}+0.5{ent})` — stage: replaces RRF — **optional**: valid only because all three signals are pre-normalized to [0,1]-ish; RRF k=60 is more robust to scale mismatch across heterogeneous lanes (fuzzy/typed/graph). Keep as a rerank-stage alternative.
3. **Entity store + bounded boost** — entity table (normalized_text UNIQUE per scope, type, `linked_memory_ids` JSON or join table); query entities ≤8; match by normalized-text/alias (no embeddings in default profile — mem0's sim≥0.95 merge becomes exact-alias or hash-embedding-nearest); boost = match_strength · 0.5 · 1/(1+0.001(n−1)²), max over entities, folded inside existing bounded-boost budget — index ≈100–500 B/memory amortized (entities are globally deduped); lookup ≤8 indexed point queries <2 ms — **ship** (the mechanism; alias rules need a non-embedding substitute — the real gap for a hashing encoder).
4. **Memory Decay [0.3–1.5] multiplier on ≤20 access stamps** — formula unpublished; reconstruction in §2 — stage: post-fusion multiplier — cost <1 ms; ≤168 B/mem — **optional**, default off (evergreen-fact risk; vendor's own counter-example).
5. **Temporal metadata + intent-driven additive rerank** — write: event_start, event_end|null, state{ongoing|completed}, precision, type ∈ {plan, state, event, relationship, preference, absence, timeless}, state_key for evolving-fact chains (~64–128 B/mem, written by background job — matches Verbatim's async extraction); read: rule/lexicon temporal-intent classifier → {historical_range, current_state, duration_state, upcoming, soft_recency, +2} with NO LLM call, then bounded additive boost for candidates whose [event_start,event_end] covers/overlaps query intent; never filters, semantic dominates — vendor: +4.1 LoCoMo overall / +6.7 temporal / −1.9 open-domain, +1 ms median — stage: temporal lane/rerank — **ship** (bounded-boost form; the classifier also fixes Verbatim's unresolved-relative-time bug class properly).
6. **state_key supersession chain** — new state closes old via event_end, ADD-only — stage: write-path linking + temporal rerank (current_state intent prefers open states) — **ship**.
7. **Single-pass ADD-only LLM extraction checklist** (§3 rules) — stage: write-path, optional LLM tier — **optional**: pin quotes to source bytes to satisfy Verbatim's byte-pin invariant.
8. **spaCy NER + lemma pipeline (en_core_web_sm, lemma+keep-ing trick)** — **reject as-is**: model download violates no-network default; ship a stdlib substitute (capitalized-span + quoted-string + possessive/case-fold normalizer; lemma via rule suffixes or FTS5 porter tokenizer) — mechanism equivalent for entity/keyword lanes.
9. **Cross-encoder/LLM reranker on final list** — pool = returned top_k (mem0 reranks only top_k, not the 60-pool!) — a small local cross-encoder over ≤32 candidates ≈ 40–120 ms on 4 cores — fits 'quality/max' ≤150 ms budget — **optional**; platform's own +150–200 ms figure confirms the budget is realistic.
10. **Semantic-pool-only rescoring (keyword/entity can't inject candidates)** — **reject**: maximally explains mem0's own open-domain weakness (82.3 vendor; Maximem-parallels) and Verbatim's open-domain gap (0.196 vs 0.370 BM25) — keep peer-lane injection.
11. **Semantic-score pre-fusion threshold gate (0.1)** — **reject**: gates on one signal pre-fusion; Verbatim gates only on eligibility.
12. **Typed-triplet LLM graph (mem0g: Neo4j, LLM triplets, conflict-resolver)** — **reject** for default: LLM write-path + ~2x store tokens; platform itself dropped it for schema-free entity co-occurrence.
13. **Entity co-occurrence adjacency + 1–2 hop expansion (platform Graph Memory)** — SQLite-realizable: entity↔memory join table already implied by (3); 1-hop expansion = `memories → entities → memories` join bounded by degree; ≈<10–30 ms at 100k for typical degree ≤50; adds ~(E·deg) rows — **optional**: vendor attributes multi-hop gains partly to it but the mechanism is conflated with the boost in (3); evaluate separately.
14. **Over-fetch constants** `max(4·k,60)` candidates, decay widen `3·k` floor 50 — stage: lane caps — **ship** (sanity anchors for Verbatim's 40-deep caps — mem0 uses similar depth).
15. **expiration_date suppression + last-10-messages session context + md5-normalized-text dedup** — **ship** (all cheap, already-aligned mechanisms).
16. **Vendor eval-side artifacts** (equivalence rules, judge gauntlets, gold-override) — **reject**: eval-prompt artifacts, explicitly out of scope/forbidden.
17. **~7k token context budget** — stage: pack — **ship** as a sanity ceiling, not a formula.

Provisional constants under review → evidence: RRF k=60 — mem0 platform 'rank scoring' is consistent with rank-fusion but undisclosed; OSS is additive-normalized, not RRF — **no vendor evidence to move k; keep 60** (standard, Cormack 2009). BM25F field weights — mem0 stores one lemmatized text field; multi-field evidence absent — **no evidence**. Boost alphas — mem0's effective additive weights are 1.0/1.0/0.5 normalized; decay band is multiplicative [0.3,1.5]; Verbatim's bounded ±5–10% is **far more conservative** than vendor bands — keep, since vendor regressions (knowledge-update −2.6, open-domain −1.9) trace to aggressive re-ordering. Cross-encoder pool 32 — consistent with mem0 reranking only top_k — **32 confirmed sane**. Entity alias rules — vendor relies on embedding sim (0.95 merge / 0.5 match); for non-semantic hashing encoder **replace with normalized-text + possessive/case-fold + optional alias table**.