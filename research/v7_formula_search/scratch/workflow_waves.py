"""Verbatim V7 formula search — research fan-out (SCRATCH orchestration).

Wave A (literature map) + Wave B (ranking formulas) run in parallel, then
Waves C (memory structure), D (OSS code reads), E (new-combination math).
Each agent writes a structured note; the script dumps all notes to
eval/v7/notes/<label>.json for synthesis by the orchestrating session.
"""

import asyncio
import json
import os

NOTES_DIR = "/workspace/memorysys/eval/v7/notes"

CTX = """CONTEXT — Verbatim is a local-first agent memory engine: Python + SQLite (one file, no server, no Neo4j, no Qdrant, no hosted vector service on the default path). Consumer API: Memory.add / search / inspect / forget. Hard rules of the engine:
- Every delivered sentence must byte-match retained source bytes ("byte pin" — a quote must match the stored payload). Generated text with no pin is not a memory.
- Eligibility (authorization, purpose, quarantine, suppression, generation) is applied BEFORE ranking. BM25 statistics must be computed on the ELIGIBLE set, not the whole table. Approximate candidate generation is allowed; approximate authorization is not.
- Default install works with no API key, no network, no model download. Embeddings today are blake2b subword-ngram feature hashing (encoder id hashing:subword-ngram:v1) — explicitly NON-semantic. An optional neural/static embedding tier is allowed only as a hash-pinned local artifact. An LLM may add a fact only when every quote byte-matches source; it must never update/delete/overwrite prior facts.
- Add-ack (write acknowledgement) stays model-free; embedding and extraction are background jobs.

PIPELINE TODAY (post-fix target): query analysis -> peer candidate lanes (lexical FTS5/BM25, fuzzy, dense/sparse-semantic, entity, time, graph, typed facts, observations) -> eligibility admission -> score/rank fusion -> optional rerank -> context pack under a byte budget. Current fusion is RRF k=60 with lane weights {1.0 strong lanes, 0.75 mid, 0.5 weak} and bounded post-fusion boosts (recency/temporal/proof-count, +-5-10%).

MEASURED STATE (2026-09-22, consumer Memory.search retrieval only, no reader, LoCoMo categories 1-4, 1536 evidence-bearing questions):
- Verbatim evidence any@10 = 0.546, any@20 = 0.590
- A 40-line Okapi BM25 over the same turn text: any@10 = 0.596, any@20 = 0.677
- Open-domain any@10 = 0.196 vs BM25 0.370
- 160 of 1536 answerable questions returned zero items; BM25 still had the evidence in its top-20 for 84 of them
- Search p50/p95 ~= 46/85 ms on ~590 turns
Known code bugs (typed-lane gating, term-coverage verdict drops, possessive/case-sensitive entity match, min-max fusion tie-collapse, 40-deep sub-lane caps, unicode61-only FTS, skipped dense lane, unresolved relative time phrases, no speaker field) are already filed and will be FIXED in the next build — design formulas for the post-fix engine, not workarounds for the bugs.

HARDWARE/LATENCY: reference machine 4-core CPU. Search p95 targets: tens of ms without a cross-encoder and <=150 ms with a small one, at 10k-100k memories. A method needing ~300 cross-encoder passes per query must show a retrieval gain large enough to justify an optional slower profile ("quality"/"max"), else reject for the default profile.

PROVISIONAL CONSTANTS UNDER REVIEW (confirm, replace, or reject each with evidence): RRF k=60; BM25F field weights; feature-rerank weights; boost alphas 0.2/0.2/0.1; PPR hop and edge weights; entity alias rules; cross-encoder pool capped at 32."""

RULES = """RETURN CONTRACT — respond via provide_structured_output with:
- topic: your question id
- summary: 8-12 sentences of what you found (the "so what" for Verbatim, including the single most important number or constant you established)
- notes_markdown: a LONG-FORM working note — target 1500-4000 words. This is the primary artifact a synthesis engineer will paste from; thin notes are a failure. Required structure: (1) every candidate formula with its exact equation, all parameters with values and where each came from, (2) per-query time complexity ending in milliseconds on the 4-core reference machine at BOTH 10k and 100k memories — show the arithmetic, (3) index/resident size in bytes per memory or per 100k memories where applicable, (4) which Verbatim stage it replaces or augments (lane / fusion / rerank / graph / temporal / entity / write-path / pack), (5) a comparison table wherever alternatives exist, (6) a verdict per candidate: ship (default profile) | optional (quality/max profile) | reject — with a one-line reason AND a "confidence / what would change this" note.
- sources: array of strings "url | vendor|paper|repo|calculated | why it mattered" — one entry per claim you rely on, not a token bibliography
- verdicts: array of strings "candidate -> ship|optional|reject: one-line reason"

DEPTH REQUIREMENTS (this is a two-day deep search, not a skim):
- Read PRIMARY sources: the actual papers (abstracts are not enough — pull the equations, ablation tables, and hyperparameters from the body) and the actual repository code. For code questions, quote the exact constant values and file paths/line numbers you found them at.
- When a number matters (a threshold, k, a weight, a latency), verify it against at least two independent sources where they exist, or derive it yourself and show the derivation.
- Worked examples required: for every formula, compute it on a small concrete example (e.g., 3 lanes x 5 candidates for fusion; a 2-entity merge for entity resolution; one query through the temporal policy) so the reader sees the mechanics, not just the algebra.
- If you can run a small calculation or prototype locally (pure Python, /tmp files only, no installs that need network beyond pip-stdlib), DO IT and report the numbers — a measured number beats a quoted one.
- Vendor self-reported scores and independent reproductions stay in SEPARATE columns — mark each.
- A paper ablation on a different corpus is evidence about a mechanism, not a prediction of Verbatim's any@10; label extrapolations and show assumptions.
- Do not call paid APIs, do not spend money, do not download model weights, do not download benchmark datasets whose license is not clearly permissive (LoCoMo is CC BY-NC 4.0 — papers are fine). If a page fails to fetch, record the miss and continue.
- Do not copy benchmark gold answers, question ids, or equivalence rules into any lexicon, prompt, or formula feature.
- Prefer a number you computed/derived to a number you copied. Show the arithmetic.
- Coverage check before you finish: re-read your assignment list and confirm every numbered sub-question has an answer in notes_markdown. If one is unanswerable, write "UNRESOLVED: <question> — <why> — <what would close it>" rather than skipping it."""


def P(qid, question):
    return CTX + "\n\nYOUR ASSIGNMENT — question " + qid + ":\n" + question + "\n\n" + RULES


WAVE_A = [
    ("a1-hindsight-tempr",
     "Hindsight TEMPR retrieval and its ablations. Read the Hindsight paper (arXiv 2512.12818) AND the vectorize-io/hindsight repository. Extract: (1) the exact TEMPR retrieval formula — what lanes exist, how scores combine, all constants; (2) the entity-resolution weights (reported as name-similarity 0.5, co-occurrence 0.3, recency 0.2, merge threshold 0.6 — verify against code); (3) the 365-day linear decay — exact shape; (4) link expansion — hop limit, fan-out, edge weights; (5) the reranker pool cap (reported ~32 — verify) and which reranker model; (6) every ablation table in the paper — what each component contributes to LongMemEval-S; (7) their reported numbers (83.6 gpt-oss-20b / 89.0 gpt-oss-120b / 91.4 Gemini-3 on LongMemEval-S, ~85.67 LoCoMo OSS-120B) marked VENDOR. Which pieces are portable to SQLite + CPU-only?"),
    ("a2-mem0",
     "Mem0's retrieval algorithm: managed platform versus OSS mem0ai. Read docs.mem0.ai how-it-works, the platform v3 migration notes, their temporal-reasoning blog post, and the OSS mem0ai repo retrieval path. Extract: (1) what the managed platform actually does that OSS does not (graph memory, rerankers, LLM extraction pipeline shape); (2) Mem0's recency/decay function — the reported 'wide decay' curve, exact formula and parameters; (3) their extraction: what gets stored vs discarded; (4) their published vendor scores (~92.5 LoCoMo, ~94.4 LongMemEval with gpt-4o answering/judging, ~7k tokens/query — mark VENDOR) vs the independent Maximem 2026 reproduction (~73.8 LongMemEval, judge-rule sensitivity); (5) OSS code constants and complexity. What is actually transferable to a no-network SQLite engine?"),
    ("a3-zep-graphiti",
     "Zep/Graphiti bi-temporal hybrid search. Read the Zep paper (arXiv 2501.13956) and the getzep/graphiti repo. Extract: (1) the bi-temporal model — event valid-time vs ingestion/record time, exact edge invalidation semantics; (2) their hybrid retrieval — which signals (semantic cosine, BM25, graph breadth-first), how they fuse (RRF? what k? cross-encoder? MMR for diversity?), all constants from code; (3) episode/node/edge data model; (4) community/summary derivation — deterministic or LLM; (5) their reported numbers vs independent reproductions; (6) license of graphiti. What of this maps to SQLite tables + SQL-only graph expansion?"),
    ("a4-chronos",
     "Chronos event-calendar ablation. Read arXiv 2603.16862 (Chronos — if the arXiv id does not resolve, search for the paper by name: 'Chronos' event calendar agent memory / temporal reasoning). Extract: (1) what the event-calendar structure is; (2) every ablation — how much temporal structure contributes vs plain retrieval; (3) their temporal query handling (relative phrases, time-aware expansion); (4) measured deltas and the retrieval baseline they compare against; (5) anything portable to a SQLite valid-time/record-time index. If the arXiv id is wrong, say so and cover the closest real paper."),
    ("a5-mastra-observational",
     "Mastra observational memory. Read Mastra's observational-memory implementation (mastra-ai/mastra repo — search for 'observational memory', 'memory processor', watcher/reflector pattern). Extract: (1) the observation layer architecture — how observations are produced from raw turns (deterministic or LLM?); (2) thresholds and triggers for consolidation; (3) how observations cite/link back to source turns; (4) how retrieval treats observations vs raw turns in ranking; (5) constants and token budgets. What is the deterministic-only equivalent for byte-pinned observations?"),
    ("a6-longmemeval",
     "LongMemEval paper techniques + dataset terms. Read arXiv 2410.10813. Extract: (1) the question types and what evidence looks like per type; (2) session decomposition technique; (3) fact-augmented keys; (4) time-aware query expansion; (5) chain-of-note — exact mechanism and measured gain per technique from the paper's tables; (6) the dataset license/terms — is downloading LongMemEval clearly permitted for research? report the actual terms. Which techniques transfer to a no-LLM-default pipeline (e.g., time-aware expansion can be deterministic)?"),
    ("a7-locomo",
     "LoCoMo benchmark structure. Read the LoCoMo paper (arXiv 2402.17753) and its dataset card. Extract: (1) category definitions 1-5 precisely — what evidence answers each category, which are adversarial/unanswerable; (2) the known category-id swap issue in the released data (find it in repo issues or reproduction notes); (3) dataset license (CC BY-NC 4.0 — confirm); (4) how evidence any@k is computed and what an evidence item is (dialog turn? sentence?); (5) what published retrieval (not QA) numbers exist on it. This calibrates which categories a formula must win — open-domain and multi-hop are the stated priorities."),
    ("a8-memory-survey-sweep",
     "Sweep of other agent/conversational memory retrieval systems worth stealing from. Cover A-MEM (agentic memory), MemoryBank, ReadAgent, COMEDY, Nexus/RAG long-memory variants, MemGPT/Letta archival recall, LangMem, and any 2025-2026 conversational-memory retrieval papers with strong retrieval (not just QA) results. For each: the retrieval formula in one line, whether it needs an LLM at query time, and whether any mechanism is portable to SQLite+CPU. End with a ranked shortlist of the 5 most portable mechanisms we did not already list (lanes/RRF/BM25F/cross-encoder/entity/temporal/PPR/observations)."),
]

WAVE_B = [
    ("b1-bm25-bm25f",
     "BM25 and BM25F exact formulas. From Robertson & Zaragoza's primer and the Lucene/Anserini implementations: (1) Okapi BM25 full equation, k1 and b defaults across Lucene (1.2/0.75) vs Anserini (0.9/0.4), IDF formulation including the negative-idf floor; (2) BM25F per-field formula — per-field tf saturation then combine (the 'correct' variant) vs combine-then-saturate, published field-weight recommendations (title/body/anchor); (3) how to compute df/idf on a SUBSET (the eligible set) — do we need per-scope docfreq tables, and what does that cost in bytes; (4) FTS5's bm25() auxiliary function — does it accept per-column weights (check SQLite docs), can BM25F be done in FTS5 or must it be Python-side; (5) dialogue-turn fields worth testing: text, speaker, resolved-entities, time-bucket label. Give the recommended BM25F field list and weights for conversational turns."),
    ("b2-rrf-sensitivity",
     "RRF deep dive and k sensitivity. From Cormack, Clarke & Buettcher 2009 and follow-on analyses: (1) the RRF formula and why k damps head-vs-tail; (2) published sensitivity results for k in {10,30,60,90} — how flat is the optimum; (3) when RRF loses to CombSUM/CombMNZ (score-informed fusion) — the conditions (calibrated comparable lane scores, few lanes, strong score signals); (4) weighted-RRF variants and whether learned per-lane weights help with lanes of very different quality; (5) RRF failure modes with a noisy/spammy lane — does rank-only fusion let garbage lanes inject noise, and mitigations (min-rank gates, weighted caps). Verbatim fuses 6-8 lanes of very heterogeneous quality — is rank-only RRF the right shape, and at what k?"),
    ("b3-combsum-calibration",
     "Score calibration before linear fusion. Cover: (1) per-lane score normalization options — min-max (fragile to outliers/ties), z-score, logistic fit, isotonic regression; what each needs per lane (a held-out query set? a fixed prior on score shape?); (2) CombSUM vs CombMNZ — when MNZ's coverage bonus wins; (3) whether a small isotonic/logistic calibration per lane can be fit OFFLINE on a dev slice and shipped as constants (storage: a handful of floats per lane); (4) the tie-collapse problem — min-max maps tied scores to 1.0; which normalizations avoid this; (5) a concrete recipe: lane score -> calibration -> weighted linear combine, with weight priors from lane quality. Is calibrated CombSUM likely to beat RRF k=60 for our lane mix, or does rank fusion win because lane scores aren't comparable?"),
    ("b4-ltr-features",
     "Learning-to-rank on eligibility-safe features. Options: pointwise logistic regression, small GBDT (e.g., LightGBM depth<=3), LambdaMART, over lane-derived features (per-lane rank, normalized score, presence flags, term coverage, entity overlap, time distance, proof count, query-class one-hots). Answer: (1) how much labeled data each needs (dev slice ~200-500 judged queries realistic?); (2) which features are eligibility-safe (computed only over the eligible set — anything leaking global stats must be recomputed per-scope); (3) expected gain over tuned RRF from published multi-lane/LTR comparisons; (4) serialization: a linear model ships as ~20 floats in SQLite — what does a GBDT cost in bytes and inference ms for ~200 candidates; (5) overfitting risk on tiny dev sets — when does LTR underperform hand-tuned RRF. Recommend: linear reranker vs GBDT vs stay-RRF for default profile."),
    ("b5-cross-encoder",
     "Cross-encoder rerankers for a CPU-only local engine. Cover ms-marco-MiniLM-L-6-v2 / L-12-v2, bge-reranker-base/large, MonoT5-small, and the newest small CEs (e.g., jina-reranker, rank-bert). For each: parameter count, ONNX availability, license (check each — Apache-2.0 vs MIT vs other), published nDCG@10 gains over BM25 on MS MARCO/BEIR at varying pool depth, and per-pair scoring latency on CPU from published benchmarks or parameter math (assume seq len 64-200, batch 8-32, 4 cores). Also: hashing/truncating inputs — effect on latency linear in seq len? Deliver: a comparison table and which 1-2 models merit a 'quality' profile, with expected any@10 gain range over BM25-only labeled as extrapolated."),
    ("b6-ce-distillation",
     "Distilling a cross-encoder into something cheap. Find papers showing CE -> bi-encoder or CE -> linear-features distillation actually transfers (e.g., cross-architecture knowledge distillation for ranking, 'DistilBERT'/TinyBERT rankers, rank-distillation literature, monoT5->dual-encoder transfers). Answer: (1) which transfers hold and typical nDCG retention; (2) is a linear model over BM25+dense+entity+time features trained on CE scores plausible — published precedent?; (3) smallest student that keeps ~95% of teacher gain; (4) inference ms estimate for a distilled tiny model at k=128 on 4-core CPU. Verdict for Verbatim: ship distilled-linear as default, optional CE for quality profile, or skip."),
    ("b7-colbert",
     "ColBERT late interaction vs single-vector bi-encoders. Cover: (1) MaxSim equation precisely; (2) storage cost per document: tokens x dim x bytes — at 128 dims fp32 vs int8, per 100k turns of ~80 tokens; (3) PLAID/centroid pruning — how much of the cost it removes; (4) CPU latency for MaxSim over ~200 candidates with token matrices in SQLite blobs (compute the flops: candidates x q_tokens x d_tokens x dim); (5) published gains of ColBERT-class vs MiniLM bi-encoder on BEIR/open-domain; (6) quantized ColBERTv2 results. Can late interaction live in SQLite blobs with acceptable ms on 4 cores, or is it quality-profile-only?"),
    ("b8-splade",
     "SPLADE and sparse lexical expansion. Cover: (1) the SPLADE equation — masked-LM logits aggregated with max, FLOPS regularization; (2) published gains over BM25 on BEIR (~+5-10 nDCG points — verify) especially on open-domain/paraphrase-heavy slices; (3) whether the expansion can be PRECOMPUTED at write time into postings (doc-side expansion only, query-side kept tiny) so search stays SQLite; (4) encoding cost per turn on CPU and model sizes/licenses (naver/splade variants, ~110M params, MIT?); (5) sparse-vec alternatives: DeepImpact, uniCOIL, EPIC — doc-term weight precomputation. Could a doc-side-expanded postings lane ship offline-default? Feasibility verdict."),
    ("b9-static-embeddings",
     "Static embedding tables: model2vec, potion, sentence-transformers static-retrieval models, GloVe-class mean-pool. Extract: (1) the encoding formula (token lookup + mean pool + L2 norm + optional PCA); (2) published MTEB retrieval numbers vs MiniLM-L6/L12 — realistic fraction of neural quality (~80-90%?); (3) encode time per query in microseconds; (4) resident bytes: vocab size x dims x 4B for model2vec-base (~30k x 256?), int8-quantized option; (5) licenses of model2vec/potion/static-retrieval-minilm; (6) whether a static table beats hashing subword-ngrams on paraphrase queries — any published direct comparison. Recommend: first semantic tier for offline default — static table vs small ONNX bi-encoder vs stay-hashing."),
    ("b10-ann-vs-exact",
     "Approximate-vs-exact dense retrieval at memory scale. Compute/read: (1) exact cosine over 100k x 384d fp32 vectors = 100k x 384 x ~4 flops ~= 15M flops + IO of ~150MB — ms on 4 cores with blocked reads from SQLite blobs vs mmap; (2) int8 quantization: 4x smaller, rescore top-m fp32 — published recall curves; (3) Matryoshka truncation — published MRL results at 64/128/256 dims (fraction of full-dim quality); (4) HNSW at N=100k d=384: recall@10 vs exact at efSearch={16,64}, build memory, library needs (hnswlib/sqlite-vec/sqlite-vss — licenses, sqlite-vec is young but MIT); (5) verdict: does the default profile need ANY ANN at 10k-100k, or exact int8+rescore? Give ms numbers."),
]

WAVE_C = [
    ("c1-entity-resolution",
     "Entity resolution without an LLM — the alias/merge formula. Compare: (a) Hindsight's published formula (name-similarity 0.5 + co-occurrence 0.3 + recency 0.2, merge threshold 0.6 — verify in their repo); (b) Dedupe (Fellegi-Sunter logistic active learning); (c) Senzing-style conservative deterministic matching; (d) a simple alias sieve (canonical form + nickname table + co-mention counts). Requirements from Verbatim: deterministic, conservative — same-name-different-person must NOT merge (e.g., two 'Caroline' speakers), while a nickname merges only when co-occurrence agrees. Deliver the recommended formula: exact similarity features (Jaro-Winkler? token overlap? edit distance?), the merge equation and threshold, the abstain band, per-pair complexity at 10k entities, and the 'do-not-merge' guard rules."),
    ("c2-coreference",
     "Coreference resolution for dialogue memory — deterministic sieves vs small models. Cover: (1) Hobbs' algorithm and mention-recency/gender/number agreement sieves — expected precision/recall on dialogue pronouns from classic results; (2) how far back pronouns resolve in chat corpora (mostly 1-3 turns?); (3) small local coref models runnable on CPU (fastcoref/FCoref ~ ling Mess, sizes, ms per turn, license); (4) an abstaining design: expand only unambiguous single-antecedent cases, else leave the turn unresolved — what fraction of anaphoric mentions does that recover (estimate from published sieve precision/recall); (5) where resolved antecedents plug in: a query-side expansion feature vs write-time entity links. Recommend previous-turn-only vs lookback-N sieve with abstain, with the expected recovered-event rate on dialogue."),
    ("c3-temporal-policy",
     "Temporal ranking policy. Compare as complete designs: (a) hard window + time-bucket spread (existing: tightest covering interval first, interleave buckets); (b) additive/multiplicative decay — exponential vs linear; Mem0's reported 'wide decay' vs Hindsight's 365-day linear — exact shapes; (c) state-key current/historical labels (supersession): 'what is true now' ranks latest value while still surfacing predecessor. For each: the formula, a FALSE-CURRENT failure mode (stale fact outranking current), and how it respects the constraint 'a temporal boost must not outrank a much stronger semantic/lexical match' (bounded multiplicative cap? additive within-tier only?). Also: interval algebra basics for valid-time containment, and dual-time (event-time vs recorded-time) query semantics. Recommend one design with constants."),
    ("c4-graph-expansion",
     "Bounded graph expansion in SQLite. Compare for a claim/entity co-occurrence graph: (a) one-hop SQL expansion (entity -> claims join, fan-out cap, indexed); (b) bounded personalized PageRank (restart alpha, hop limit, edge weights by co-occurrence count/recency, fan-out cap per node); (c) entity-overlap-only scoring (no walk). Compute per-query cost at 10k and 100k nodes: one-hop = O(fan-out) indexed lookups; PPR via push algorithm ~ O(1/(alpha*eps)) pushes on a bounded subgraph — convert to ms at realistic edge density (~5-20 edges/node). When does hop-2 PPR beat one-hop analytically (multi-entity queries? sparse mentions?)? Recommend hop limit, fan-out cap, alpha, edge weights — or 'entity-overlap only' if the math says expansion can't pay."),
    ("c5-consolidation",
     "Consolidation and derived views without generation. Cover: (1) evidence-backed observation rows — deterministic aggregation (entity+predicate -> observation with proof_count and source pins), near-duplicate merge policies (shingle/MinHash on claim text — thresholds), supersession chains for evolving attributes; (2) freshness semantics — when new unconsolidated evidence exists, how should observation freshness read (degraded/derived-only flags?); (3) published or repo evidence that consolidated/typed-fact retrieval beats raw-turn retrieval (Hindsight observations, Mem0 extraction, Zep edges) — VENDOR vs independent marks; (4) when consolidation pays: which query classes benefit, and the cost (index bytes, write-path work). Recommend the deterministic consolidation shape for Verbatim."),
    ("c6-dual-time-resolution",
     "Dual-time indexing and relative-phrase resolution. Design: (1) resolve 'last week'/'3 days ago'/weekday names/'in March' against message time vs session start vs now — precedence rules and ambiguity handling (LongMemEval uses session-relative anchors); (2) SQLite layout for bi-temporal data: valid_from/valid_until + recorded_from/recorded_until, indexes that keep interval-containment queries fast at 100k rows; (3) known-at query semantics (as-of recorded-seq) already in code — what the query-time resolution adds; (4) edge cases: timezone-free dialogue, 'this morning' spanning sessions, conflicting cues -> abstain+warn (existing behavior). Give the deterministic resolution table (phrase -> offset/interval against which anchor)."),
]

WAVE_D = [
    ("d1-hindsight-code",
     "Read the actual code of github.com/vectorize-io/hindsight (clone or browse — it's public). Extract the TEMPR retrieval implementation: the real lane set, fusion formula and constants in code, the reranker pool cap (verify ~32) and reranker used, link-expansion code (hop limit, fan-out, edge weights), entity-resolution weights and merge threshold, the 365-day decay implementation, and observation generation (LLM or deterministic). Also: storage layout (what DB), per-query complexity, and the LICENSE. Report code file paths for each constant. Separate what the code does from what the paper claims."),
    ("d2-mem0-code",
     "Read the actual OSS code of github.com/mem0ai/mem0 (the open-source path only — not the managed platform). Extract: the retrieval path (vector search over what index, optional graph store dependencies, reranker usage), the add path (LLM extraction calls, dedup/update logic), decay or recency handling if any in code, key constants (top-k defaults, score thresholds), storage backends supported, and the LICENSE. Then state plainly: which of Mem0's reported quality comes from the OSS code vs the managed platform per docs/issues. What retrieval mechanism in OSS is portable to SQLite?"),
    ("d3-graphiti-code",
     "Read the actual code of github.com/getzep/graphiti. Extract: the hybrid search implementation — which stores/indexes (Neo4j required? FalkorDB?), the fusion function (is it RRF — what k — or cross-encoder rerank, or MMR), all fusion/rerank constants, breadth-first search depth and fan-out in code, edge invalidation semantics (bi-temporal fields), community/summary derivation (LLM), and the LICENSE (Apache-2.0?). Which parts are Neo4j-coupled vs portable — e.g., could their fusion constants serve our SQL-only equivalent?"),
    ("d4-ranx-pyserini",
     "Read ranx (github.com/AmenRa/ranx) and pyserini code/docs. Extract: (1) ranx fusion function implementations — RRF, CombSUM, CombMNZ, weighted variants — exact semantics (score fields needed, normalization built-in?); (2) Anserini/pyserini BM25 defaults (k1=0.9 b=0.4) and the Lucene defaults (1.2/0.75) — which matters for short dialogue turns; (3) how ranx normalizes scores before sum fusion (min-max?); (4) pre-built index patterns we could mimic in SQLite; (5) licenses. Deliver the ported formulas as pseudocode for our lane-fusion stage."),
    ("d5-fts5-sqlite",
     "What SQLite gives us natively — capabilities inventory for the formula spec. Verify from SQLite docs and extension repos: (1) FTS5 bm25() auxiliary function — does it accept per-column weights (the bm25(tbl, w...) argument form), k1/b defaults inside FTS5 (0.9/0.4 or 1.2/0.75?), can we get per-field BM25F by multi-table or column-weight tricks; (2) FTS5 tokenizers: unicode61 options, porter stemmer, trigram tokenizer (substring/prefix queries for the fuzzy lane); (3) sqlite-vec and sqlite-vss status, license, and suitability at 100k vectors; (4) FTS5 external-content tables and per-scope statistics — can docfreq be per-scope; (5) json1/generated columns for state-key lookups. Deliver: what ships in SQL vs what must be Python, with doc links."),
]

WAVE_E = [
    ("e1-ce-kcurve-latency",
     "Work the cross-encoder depth-vs-latency curve concretely. For an ms-marco-MiniLM-L-6-v2-class cross-encoder (~23M params, 6 layers, hidden 384) scoring (query, turn) pairs at seq len ~96-160 tokens on a 4-core CPU with ONNX int8: (1) derive per-pair ms from published CPU benchmarks or transformer FLOPs math (2*params*tokens FLOPs per pass; ~2-6 GFLOPs/s/core effective for int8 ONNX — state your assumption); (2) produce the p95 ms table for pool depth k in {0,16,32,64,128,300} at batch sizes {8,32}; (3) find published evidence of any@10/NDCG vs pool depth (typically saturates ~50-100) to pick the smallest k within 0.005 of best; (4) verdict: default-profile k, quality-profile k, or reject CE for default. Show all arithmetic."),
    ("e2-ppr-vs-onehop",
     "Work PPR-vs-one-hop expansion mathematically. Model a claim/entity co-occurrence graph: N in {10k,100k} nodes, mean degree {5,20} (state assumptions from dialogue entity density). Compute: (1) one-hop SQL expansion — one indexed join, result cap; ms at both N; (2) two-hop with fan-out cap F in {20,50} — join count and ms; (3) bounded PPR via the push/forward-push algorithm: number of pushes ~ O(1/(alpha*rmax)) with alpha in {0.15,0.2}, eps/rmax tuned so subgraph stays ~500-2000 nodes — ms at both N and when it degenerates to one-hop; (4) score effect: when does hop-2 evidence plausibly lift multi-hop any@10 (queries needing 2-link chains — what fraction of LoCoMo multi-hop is 2+ hops?). Recommend: one-hop cap F, PPR hop limit + alpha + edge-weight shape, or overlap-only — with the ms table."),
    ("e3-embed-tier-sizing",
     "Size the first semantic tier precisely. For a static mean-pool table (model2vec-class: ~30k vocab x 256d) and a small ONNX bi-encoder (MiniLM-L6-class, ~23M params): compute (1) resident bytes per 100k memory units — stored vectors at fp32 (256d=1KB) vs int8 (256B) vs truncated 64d int8 (64B), plus the table itself (~30MB fp32 / ~7.5MB int8); (2) query encode time: static ~10-50us lookup+mean vs ONNX ~5-15ms on 4-core CPU; (3) expected any@10 gain on paraphrase/open-domain queries over hashing baseline — argue from published MTEB retrieval numbers (static retrieves ~80-90% of MiniLM quality; hashing is subword-only) labeled as extrapolated; (4) shipped-bytes budget: is a ~7.5MB int8 static table an acceptable hash-pinned artifact. Recommend the tier: static-int8 vs ONNX-MiniLM vs stay-hashing, per profile."),
    ("e4-fact-index-ceiling",
     "Estimate the quote-pinned fact-index ceiling vs raw turns. Reason: a typed-facts lane holds short subject-predicate-object rows whose quotes byte-match source turns (human/rule-extracted = oracle; LLM-extracted only if it pays). (1) Why facts could beat turns: dedup (one fact per repeated mention), length normalization (BM25 on 10-token facts vs 100-token turns), entity-exact joins, state-key current flags; (2) why they could lose: extraction coverage (facts only exist where extracted — hybrid lane needed), quote-pin constraint (fact text must byte-match => near-verbatim only); (3) estimate the ceiling: from published extraction-memory-vs-raw-text deltas (Hindsight/Mem0 ablations, marked appropriately) and an own arithmetic argument — e.g., if 30% of evidence turns lack extractable facts, a facts-only cap is ~0.7x raw coverage; a facts-PLUS-turns lane union caps at the better of the two per question; (4) verdict: what any@10 lift would justify turning T2 extraction on for a quality profile vs staying off. Deliver a defensible ceiling estimate and the decision rule."),
    ("e5-coref-fixture",
     "Prototype a deterministic coreference sieve on invented dialogue. Write a small fixture (10-15 invented multi-turn dialogues with pronouns, possessives, and name re-mentions — your own text, NOT benchmark data) under /tmp. Implement and measure two policies: (a) previous-turn-only: resolve a pronoun to the last explicitly-named entity in the immediately preceding turn; (b) lookback-N (N=3) sieve with abstain: resolve to the single most-recent compatible antecedent (match number/gender heuristics, skip if 2+ candidates tie) else leave unresolved. Report: resolution rate, correct-resolution rate on the fixture's hand-labeled truth, fraction of 'event records' each policy recovers (a turn carrying an entity reference now linked), and the abstain rate for (b). Also sketch where the resolved entity plugs into retrieval (write-time entity link vs query-time expansion). Deliver the fixture, the numbers, and a ship verdict — all your own computation."),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "summary": {"type": "string"},
        "notes_markdown": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
        "verdicts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["topic", "summary", "notes_markdown", "sources"],
}


async def run_item(phase, label, question):
    # no mode override: children of a free SWE-2 session must inherit
    # SWE-2 (explicit devin_mode is rejected server-side). Finished child
    # sessions linger awake and hold their slot, so transient session-
    # creation 429s get a patient retry loop instead of failing the item.
    last_exc = None
    for attempt in range(4):
        try:
            return await agent(P(label, question), phase=phase,
                               schema=SCHEMA, label=label)
        except WorkflowAgentError as exc:
            last_exc = exc
            wait_s = [120, 300, 600][min(attempt, 2)]
            log(f"[{label}] attempt {attempt + 1} failed ({exc}); "
                f"retry in {wait_s}s")
            await asyncio.sleep(wait_s)
    return {"topic": label,
            "summary": f"AGENT FAILED after retries: {last_exc}",
            "notes_markdown": "", "sources": [], "verdicts": []}


async def run_phase_group(groups, batch=4):
    """Run several phases' items through ONE global concurrency cap.

    The org caps concurrent child sessions, so all waves share a single
    queue: at most `batch` children run at once across every wave. The
    barrier between groups still holds — C/D/E start only when A+B finish.
    """
    queue = [(phase, label, q) for phase, items in groups for label, q in items]
    results = []
    for i in range(0, len(queue), batch):
        chunk = queue[i:i + batch]
        thunks = []
        for ph, label, q in chunk:
            async def _t(ph=ph, label=label, q=q):
                return await run_item(ph, label, q)
            thunks.append(_t)
        results.extend(await parallel(thunks))
        log(f"batch done: {min(i + batch, len(queue))}/{len(queue)} agents finished")
    return results


async def main():
    os.makedirs(NOTES_DIR, exist_ok=True)
    await register_workflow({
        "name": "verbatim-formula-search",
        "description": "V7 formula search: literature + ranking + structure + code + combos",
        "product": "Verbatim memory engine",
        "soft_time_limit_minutes": 50,
        "phases": [
            {"title": "wave_a", "detail": "Literature map: vendor systems + benchmark papers",
             "labels": [l for l, _ in WAVE_A]},
            {"title": "wave_b", "detail": "Ranking formulas: IR theory, CEs, embeddings",
             "labels": [l for l, _ in WAVE_B]},
            {"title": "wave_c", "detail": "Memory structure: entity/coref/temporal/graph/consolidation",
             "labels": [l for l, _ in WAVE_C]},
            {"title": "wave_d", "detail": "OSS code reads + SQLite capabilities",
             "labels": [l for l, _ in WAVE_D]},
            {"title": "wave_e", "detail": "New combinations + worked arithmetic",
             "labels": [l for l, _ in WAVE_E], "soft_time_limit_minutes": 60},
        ],
    })

    log("Waves A+B: 18 agents, batches of 4")
    ab = await run_phase_group([("wave_a", WAVE_A), ("wave_b", WAVE_B)], batch=4)
    log("Waves A+B done; launching C+D+E")

    cde = await run_phase_group(
        [("wave_c", WAVE_C), ("wave_d", WAVE_D), ("wave_e", WAVE_E)], batch=4)
    log("All waves done; writing notes")

    all_notes = list(ab) + list(cde)
    for note in all_notes:
        label = note.get("topic", "unknown").replace("/", "_")
        path = os.path.join(NOTES_DIR, label + ".json")
        with open(path, "w") as fh:
            json.dump(note, fh, indent=2)
        log(f"wrote {path} ({len(note.get('notes_markdown', ''))} chars)")


if __name__ == "__main__":
    asyncio.run(main())
