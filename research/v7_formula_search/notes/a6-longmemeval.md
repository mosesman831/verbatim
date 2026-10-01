# a6-longmemeval — LongMemEval (arXiv 2410.10813) techniques, license, and transfer analysis for Verbatim

Paper: Wu, Wang, Yu, Zhang, Chang, Yu — "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025). Repo: github.com/xiaowu0162/LongMemEval. All table numbers below are quoted verbatim from the arXiv HTML (v3-era text); arithmetic verifications are flagged where I recomputed them.

## 1. Question types and what evidence looks like

LongMemEval = 500 questions over five abilities, instantiated as seven question types:

| question_type (data name) | ability | evidence shape |
|---|---|---|
| single-session-user (`single_hop`) | IE | one user-side statement inside one task-oriented session |
| single-session-assistant (`assistant_previnfo`) | IE | one **assistant**-side statement (unique to this benchmark; LoCoMo lacks it) |
| single-session-preference (`implicit_preference_v2`) | IE | user profile; rubric-judged personalized response |
| multi-session (`two_hop`, `multi_session_synthesis`) | MR | evidence spread across 2–6 sessions; aggregation/comparison |
| knowledge-update (`knowledge_update`) | KU | **two versions** of a fact; correct answer requires the *latest* — E.5 shows ~10% of correct answers needed the pre-update item too |
| temporal-reasoning (`temp_reasoning_implicit`, `temp_reasoning_explicit`) | TR | session timestamp metadata AND/OR explicit in-text time mentions ("last weekend", "in March") |
| abstention (`*_abs` id suffix) | ABS | 30 false-premise questions mutated from other types; correct answer is "I don't know" |

Evidence mechanics (§3.2, §A.1–A.3): an ontology of 164 user attributes → background paragraphs → ~1000 LLM-proposed questions per type with ~5% human yield → evidence statements embedded *indirectly* in self-chatted task-oriented sessions (≤10 rounds each) → 70% human-edited → haystack compiled from 25% ShareGPT + 25% UltraChat + 50% simulated sessions; timestamps randomly assigned in May 2023 unless evidence-anchored. Two frozen settings: LongMemEval_S (~115k tokens, ~40–50 sessions) and LongMemEval_M (500 sessions, ~1.5M tokens). Retrieval metrics: Recall@k, NDCG@k over `answer_session_ids` / `has_answer` turn labels; QA judged by gpt-4o-2024-08-06 prompt (97–98% agreement with humans).

Relevance to Verbatim: the eval emphasizes *indirectly stated* evidence — lexical gap between question and evidence is real. And KU + TR both exercise the temporal index, which Verbatim currently lacks ("unresolved relative time phrases" is already a filed bug).

## 2. Candidate techniques, formulas, parameters, evidence

Setup defaults in the paper: retriever = Stella V5 1.5B dense; indexer/extractor = Llama-3.1-8B (10 ICL examples for fact extraction; user-side messages only); reader sees top-k items **sorted by timestamp**, in JSON format, with Chain-of-Note prompt; greedy decoding, ≤800 gen tokens.

### C1 — Value granularity: session → round (session decomposition)

Mechanism: decompose each session into rounds (one user message + one assistant response) as the storable value; alternatively compress into summaries/facts.

Result (Fig 5, LongMemEval_M): rounds > sessions for GPT-4o reader; ≈equal for 8B reader; facts/summaries as *values* hurt overall but help multi-session reasoning. Llama-8B reader degrades beyond ~3k retrieved tokens; GPT-4o keeps improving past 20k.

No new formula — it's a granularity choice. Verbatim stage: **write-path / pack granularity**. Cost: none at query time; index row count ~N×(avg rounds/session) with tiny per-row overhead (measured FTS5 overhead ~10% going 60-token docs → more, smaller rows).

### C2 — Fact-augmented keys (document expansion)

Mechanism: `key = concat(extracted_user_facts, value_text)` — classic document expansion (Tao 2006, Efron 2012). Facts extracted only from user-side utterances. Notably, a *separate* fact index with rank-merged results (true multi-path retrieval) was tried and **lost badly** (Table 10, §E.3): paper's explanation — rank merging multiplies index size by (m+1) and dilutes ranks.

Key numbers (Table 3, LongMemEval_M, Stella V5):

| Key design | value=session R@5 | R@10 | value=round R@5 | R@10 |
|---|---|---|---|---|
| K = V | 0.706 | 0.783 | 0.582 | 0.692 |
| K = fact (alone) | 0.642 | 0.814 | 0.530 | 0.654 |
| K = keyphrase (alone) | 0.482 | 0.576 | 0.282 | 0.392 |
| K = V + fact (key merge) | **0.732** | **0.862** | **0.644** | **0.784** |
| K = V + fact (rank merge, E.3) | 0.618 | 0.754 | 0.478 | 0.568 |

Paper's headline: "+9.4% recall@k, +5.4% QA" — I verified these are **relative** percentages: mean over the four recall cells gives (3.7+10.1+10.7+13.3)/4 ≈ 9.45%; absolute recall@10 gain ≈ +6.5pts; QA absolute ≈ +3.4pts. Condensed keys *alone* never beat V — the win is strictly in augmenting, not replacing.

**No-LLM transfer**: the "facts" can be a deterministic extraction channel — entities/numbers/dates/normalized noun-phrases emitted as an auxiliary key field. Byte-pin is safe: keys are index-side, never delivered. Two implementations measured on my box (SQLite FTS5, 60-token doc + 10-token expansion ≈ +17% source bytes):

| impl | formula | index/mem @100k | query @10k | query @100k |
|---|---|---|---|---|
| concat into body | bm25(fts) on expanded text | 794 B (+21% vs 655 plain) | ~0.1–0.3 ms | 0.4–1.7 ms (+≈0.3 ms worst) |
| BM25F aux column | `bm25(t, w_body, w_aux)`, w_aux≈0.5–1.0 | 814 B (+24%) | ~0.1–0.3 ms | 0.4–1.7 ms |

BM25F is preferable to concat: it keeps expansion tf out of the body's length normalization. Query cost is a wash; size cost ~150 B/mem. Stage: **index/lane internals** (strengthens the lexical lane; evidence against a *separate* fused fact lane).

### C3 — Time-aware indexing + query expansion (temporal prefilter)

Mechanism (§5.4, App D): at index time, extract `(date, event)` pairs whose date is stated or inferable, anchored by the session timestamp; at query time an LLM M_T maps the question + question_date to `{start,end}` JSON or `N/A`, and retrieval is restricted to in-range items.

Results (Table 4, TR subset of LongMemEval_M, M_T=GPT-4o): recall gain +11.3% (round) / +6.8% (session) — I verified: round cells (0.421→0.451, 0.499→0.495, 0.489→0.526, 0.550→0.722) mean-relative = (7.1−0.8+7.6+31.3)/4 = 11.3%. **But** with M_T=Llama-8B, gains flip negative on some cells (round K=V+fact R@5: 0.489→0.481): the weak model hallucinates ranges on non-temporal questions and prunes out evidence. Table 11 shows three false-positive examples.

**Verbatim transfer — and it's a structural fit, not just a trick**: the range is applied as a *prefilter*, which in Verbatim terms is exactly **eligibility admission, computed before ranking**. The deterministic version: (a) write-path — rule-extract dates from text (dates, "N days ago", month names) resolved against item timestamp → `item_dates(item_id, ymd)` table; (b) query-path — precision-first regex resolver over the question; emit a range only on explicit temporal anchors, else `N/A` → no filter. Precision ≈ 1.0 by construction on covered patterns; a resolver that never guesses cannot reproduce the Llama-8B failure mode.

My ~60-line prototype resolver scored 14/14 on a test set including all four Table 11 examples (GPT-4o matched: correct range on "in March and April"; correct `N/A` on "how long had I been taking lessons…", "how many days before 'Rack Fest'…", "which seeds were started first…"). Latency measured: **6 µs/query**.

Measured index+filter cost (SQLite, ~1.5 event-dates/mem avg):

| N | index size | 30-day range filter |
|---|---|---|
| 10k | ~0.5 MB (49 B/mem incl. rows+B-tree) | 0.70–0.81 ms (1.2k hits) |
| 100k | ~4.9 MB | 11.3 ms (12k hits — fetch-bound) |

12k hits at 100k is worst-case fetch cost; intersecting inside the FTS5 MATCH (rowid-list) or a roaring-bitmap prefilter takes narrow ranges sub-ms; 30-day windows over a ~1-year corpus are the upper end of selectivity ~8%. For tighter safety, apply the range as eligibility *intersection* with a fallback: if the eligible set empties or shrinks below k, drop the filter (never return zero because of a parse). Stage: **temporal/eligibility**.

### C4 — Chain-of-Note reading (CoN) + JSON item format + timestamp ordering

Mechanism (Yu et al. 2023 variant, Fig 13): reader first emits a per-item evidence "note" then reasons over the notes; items rendered as JSON records; retrieved items **always sorted by timestamp**. Decoding: greedy, ≤800 tokens.

Gains (Fig 3b/Fig 6, oracle retrieval): up to +10 absolute QA points — GPT-4o oracle 0.870→0.924, L-70B 0.744→0.848, L-8B 0.710→0.710 (flat), some minis slightly negative. On the non-oracle S setting: GPT-4o 0.606→0.640. E.5: ~90% of correct answers required correct retrieval; 15–19% of all instances = correct retrieval but wrong generation.

Verbatim transfer: CoN itself is a *reader* technique and produces generated text — conflicts with byte-pin for delivered content and there's no reader in the Memory.search path. **Reject for the default pipeline.** Two free sub-parts do transfer: (a) **timestamp ordering of the context pack** — present pack items chronologically; cost O(k log k) ≈ µs at k≤budget; (b) **structured item records in the pack** (a small JSON/text header per item with date + speaker) is byte-pin-safe as metadata framing, not delivered prose.

### C5 — Supporting findings worth carrying into the design

- **Retriever ablation (Table 9)**: on LongMemEval_M, dense > BM25 everywhere (session K=V R@5: BM25 0.634, Contriever 0.723, Stella 0.720). Caveat: this is a 500-session/1.5M-token haystack with *indirectly stated* evidence — a different corpus regime from Verbatim's LoCoMo measurement where BM25 beat Verbatim. Label as corpus-dependent evidence, not a contradiction.
- **User-side-only keys**: when sessions/rounds are keys they index only user utterances — evidence for a speaker field in the index (aligns with Verbatim's missing-speaker-field bugfix).
- **Item ordering in pack**: timestamp-sorted (above).
- **KU retrieval failure mode** (E.5): correct answers sometimes needed the *pre-update* item too — for update-type queries, don't suppress older versions; the temporal lane should return both sides of an update.

## 3. Worked examples

**C2 mechanics (why key-merge beats rank-merge).** 5 candidates {A..E}, lane1 = body-lexical [A,B,C,D,E], lane2 = fact-lane [C,A,D,B,E], lane3 = entity/time [E,D,C,B,A]; RRF k=60, lane weights 1.0/0.75/0.5: score(X) = Σ_lanes w_lane/(60 + rank_lane(X)). Scores: A 0.03618, B 0.03566, C 0.03610, D 0.03559, E 0.03512 → fused order A,C,B,D,E. Verified numerically. If lane2's top item were a false-positive fact match it would drag it into top-k regardless of body evidence — exactly the dilution the paper observed (session K=V+fact R@10: key-merge 0.862 vs rank-merge 0.754). **Design note for Verbatim**: an extracted-fact *channel* should be folded into the lexical document (concat or BM25F aux field), not exposed as a separate fused lane. This does NOT generalize to "kill all lanes" — the paper's rank-merge compared one duplicated lane against one merged index; Verbatim's lanes are orthogonal evidence sources (temporal, entity, fuzzy) and its eligibility-first rule is orthogonal to fusion. The evidence supports BM25F-with-aux-field for the fact channel specifically.

**C2 worked micro-score (BM25F).** Query expansion term `napoli_ate_at` appended to aux field; doc1 (evidence) has body tf=1 + aux tf=1; doc2 (distractor "napoli travel guide") body tf=3, aux tf=0. With bm25(k1=1.2, b=0.75), N=1000, avgdl=80, weights w_body=1.0/w_aux=0.6: doc1 ≈ 1.0·bm25(body)+0.6·bm25(aux) = 1.594 vs doc2 = 1.184 — the aux channel can outrank a term-frequent distractor. (Illustrative mechanics, not a measured corpus result.)

**C3 worked policy trace.** `question_date = 2023-04-27`, `q = "Which airline did I fly with the most in March and April?"` → resolver emits `(2023-03-01, 2023-04-30)` → eligible set := items with any extracted event date OR item timestamp inside the range → rank only that set (BM25 stats computed on eligible set per Verbatim's rule — the paper filters before ranking, same ordering). `q = "How long had I been taking guitar lessons when I bought the new guitar amp?"` → `N/A` → no constraint applied, full eligible set. Guard: if range ∩ eligible < k, fall back to unfiltered (a resolver must never produce the Llama-8B outcome — zero results from a hallucinated window).

**C1 sizing.** Round granularity: LongMemEval_M session ≈ 1.5M tokens / 500 sessions ≈ 3k tokens per session vs ≈300 tokens per round (≤10 rounds); for Verbatim's ~590-turn LoCoMo corpora the unit is already turn-like — adopt "one round/turn per item; never a whole session blob" as the pack unit and treat fact-only values as rejected (paper: they lose detail except on MR).

## 4. Provisional constants under review — what this paper says

- **RRF k=60**: paper doesn't use RRF; its negative result on rank merging (Table 10) weakly supports fewer, higher-quality lanes — keep k=60 (Roberts-correct standard value), but treat "one lane per deterministic extraction channel" as disproven: fold extractions into fields.
- **BM25F field weights**: no paper evidence; my suggestion w_aux ∈ [0.5, 0.8] is derived, not measured — tune on the post-fix eval, don't take from here.
- **Boost alphas 0.2/0.2/0.1**: no bearing — temporal evidence should be a hard eligibility filter where a range exists, not a ±boost.
- **PPR / entity alias / cross-encoder pool**: out of scope for this paper (HippoRAG is only surveyed in Table 2/§C.2 — entity graph + PPR retrieval; no constants given).

## 5. Complexity summary (4-core reference, measured on this box unless noted)

| candidate | per-query cost @10k | @100k | index growth | stage |
|---|---|---|---|---|
| round granularity | 0 (pack granularity) | 0 | ~+10% rows | write-path/pack |
| fact-key expansion (deterministic extract + BM25F field) | ~0.1–0.3 ms | 0.4–1.7 ms (+0.3 ms max) | +140–160 B/mem | index/lane |
| temporal prefilter (resolve + date index) | ~0.7 ms | ~11 ms worst-case (sub-ms w/ bitmap intersect) | ~49 B/mem | eligibility |
| CoN reading | n/a — no reader | n/a | 0 | reject (reader) |
| timestamp-sorted pack | <0.01 ms | <0.01 ms | 0 | pack |
| separate fact lane + rank merge | adds a full second index (~+100% aux) | | ~2× aux index | reject (E.3 evidence) |

## 6. Dataset license — is downloading permitted?

**Yes, unambiguously.** Three independent confirmations: GitHub repo carries MIT License; the HF dataset cards (`xiaowu0162/longmemeval` and its replacement `longmemeval-cleaned`) declare `license: mit`, are public and **not gated**; the paper's ethics statement commits to MIT for the data release. Sources inside the haystack are permissive: ShareGPT (Apache-2.0 via FastChat) and UltraChat (MIT). Caveats: eval-side judge needs an OpenAI API key (tool dependency, not a license); IRB-exempt human curation noted; ~3 GB total. The Sept-2025 cleaned files (`*_cleaned.json`) supersede the originals.

## 7. Verdicts

- **C1 round granularity** → **ship (default)**: zero-cost pack granularity improvement; matches paper's session→round win. Confidence: high — but Verbatim items are already turn-like, so expected gain is small; revisit if pack units ever exceed ~1k tokens.
- **C2 deterministic fact-augmented key (BM25F aux field)** → **ship (default)**: +9.4% rel recall / +6.5 abs pts R@10 in paper; measured <+0.3 ms and +160 B/mem on my box; byte-pin-safe (index-only). Confidence: medium — paper used LLM facts + dense retriever; the deterministic extraction channel (entities/numbers/dates/noun-phrases) is a mechanism extrapolation and may capture a fraction of the gain. Would change this: ablation on the post-fix LoCoMo harness.
- **C3 deterministic time-aware prefilter** → **ship (default)**: +6.8–11.3% rel recall on TR subset; aligns with eligibility-before-ranking; my deterministic resolver hit 14/14 incl. all four paper examples and runs at 6 µs/query; date index ~49 B/mem, ≤11 ms worst-case filter at 100k. Confidence: medium-high — resolver coverage is bounded by pattern list; the fallback-on-empty guard is mandatory. Would change this: measured TR-question recall in eval; pattern misses should degrade to no-filter, never to a wrong window.
- **C4 Chain-of-Note reading** → **reject (default)**: reader-side LLM technique producing generated notes — violates byte-pin for delivered content; no reader in the measured path. Its two free sub-parts — timestamp-ordered pack and structured item headers — ship separately. Confidence: high for the default profile; would revisit only if Verbatim adds an answering layer.
- **Separate fact lane + rank merging** → **reject**: paper-measured large regression vs key merging (R@10 0.862→0.754 session); also multiplies index size by (m+1). Confidence: medium — single-corpus, dense-retriever evidence; direction and mechanism (rank dilution) are consistent.
- **LongMemEval as a second eval corpus** → **optional**: MIT-licensed, downloads cleanly; S set fits well (~115k tokens ≈ Verbatim scale). Needs a retrieval-metric harness over `answer_session_ids`/`has_answer` (their QA judge needs OpenAI).