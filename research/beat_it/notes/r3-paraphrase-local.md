# Paraphrase bridging for the residual ~38% true misses — measurement + ranked recipe

**Scope**: question B — local (no-LLM, no-network) paraphrase bridging. Compared (a) static-embedding lane potion-base-8M int8, (b) RM3/PRF on BM25, (c) char n-gram soft-tf, (d) dialogue-context expansion, (e) bundled WordNet synonyms. All lifts measured on the real LoCoMo dev slice unless tagged otherwise.

## Setup and validity of the measurement base

- Corpus: `load_corpus('locomo', split='dev', env={'VERBATIM_EVAL_LOCOMO':'1'})` — `eval/v7/corpora.py:455-530` applies `SplitSpec(unit='conversation', dev_pct=40, tag='v7-split-v1')` from `eval/v7/dataset_registry.py` → dev = conv-30/42/47/48/49 = 2877 items, 990 tasks, **777 answerable** (213 cat5 adversarial answerable=False scored separately). [code, measured]
- Baseline arm: `FlatBM25Arm` exact semantics re-implemented (`eval/v7/arms.py` — Okapi k1=1.2 b=0.75, `\w+` tokenizer lowercased, idf ln(1+(N−df+.5)/(df+.5))). [code]
- **Fidelity check**: my harness any@10 = **0.5894** vs parent's measured 0.587 (Δ0.002, noise). Miss decomposition: 319 misses = **196 truncation-shaped (61%)** + **123 true-gap (39%)** — independently reproduces the parent's ~62/38 split, so flat-BM25 is an honest proxy for measuring the paraphrase lane. [measured]
- All numbers below are any@10 over the 777 answerable tasks; `gap recovered` = fraction of the 123 true-gap misses whose gold enters top-10. [measured]

## Measured results table (any@10, 777 answerable dev tasks)

| method | any@10 | Δ vs base | notes |
|---|---|---|---|
| raw BM25 (verbatim flat formula) | 0.5894 | — | p50 1.47ms/query [measured] |
| + RM3 (F10 E20 λ0.5) | 0.5881 | −0.001 | **0/123 gap recovered** [measured] |
| + RM3 (F5 E10 λ0.3) | 0.6059 | +0.017 | morphology luck, not paraphrase |
| + char-3gram Jaccard θ0.45 (w0.5, E≤10) | 0.6177 | +0.028 | on raw; see stem interaction |
| + speaker names in query | 0.6008 | +0.011 | weak [measured] |
| + WordNet top-2 synsets (w0.5) | 0.6073 | +0.018 | raw only; **negative on stemmed** |
| + docctx window ±1 turn | 0.7048 | **+0.115** | dominant single fix [measured] |
| + docctx window ±2 | 0.6762 | +0.087 | worse than ±1 — noise |
| + score-propagation w0.7 (no re-index) | 0.7284 | +0.079 | half the window effect, free [measured] |
| potion-base-8M int8 alone | 0.5637 | — | any@20 0.6602; 7.6MB table + 0.74MB doc matrix [measured] |
| potion-retrieval-32M int8 alone | 0.4389 | — | **worse than 8M** on dialogue; 32MB [measured] |
| raw + RRF60 with cos | 0.6332 | +0.044 | rank fusion only [measured] |
| raw + score-fuse α·cos (α=1.2) | 0.6461 | +0.057 | [measured] |
| **stemmed BM25 (porter)** | 0.6486 | +0.059 | ≈ verbatim's existing stem channel |
| stem + ±1 window | 0.7606 | +0.112 vs stem | window still dominates [measured] |
| stem + ±1 + ngram | 0.7568 | −0.004 | ngram marginal ≈ 0 after stemming [measured] |
| stem + ±1 + cos1.2 (fp32) | **0.8005** | +0.040 vs stem+win | full recipe [measured] |
| stem + ±1 + cos1.2 (int8) | 0.7954 | — | int8 costs ~0.005 any@10 [measured] |

Latency/memory: query encode 0.081ms p50 / 0.16ms p95 [measured]; exact int8/f32 cosine scan over 2877 docs ~0.3ms [measured]; model2vec CI ~8µs/query [vendor]; int8 table 7.6MB, per-doc int8 vectors 0.74MB for the dev corpus (~0.26 KB/item → ~26MB @100k items) [calculated].

## Per-option findings

### (b) RM3 / pseudo-relevance feedback — measured zero, dead code anyway

- RM3 expansion `P(t|FB) ∝ Σ_{d∈top-F} P(t|d)P(q|d)`, F∈{5,10}, E∈{10,20}, λ∈{0.3,0.5}: best config 0.6059 (+0.017) but **0/123 true-gap tasks recovered** — the lift is morphology collisions, not paraphrase. F10E20 is a net negative. [measured]
- Mechanism: PRF needs ≥1 *correct* doc in the feedback set to pull its vocabulary. Zero-overlap paraphrase queries are precisely the case where top-F is systematically off-topic, so expansion injects topical noise (e.g. 'destress' pulls 'stress'-adjacent filler, not 'stress relief' docs). PRF has no bootstrap for the failure mode we actually have. [inference from measurement]
- **Dead flag**: `PoolProfile.prf_round` at `verbatim/core/types_v7.py:411` has no consumer — grep over `verbatim/` finds only the dataclass field; POOLS HIGH sets `prf_round=True` but nothing executes it. Verdict: remove or ignore; do not implement. [code]
- Literature context: classic RM3 gains (+5–15% avgPrec) were measured on AQUAINT news with F=10–50 (Abdul-Jaleel et al. TREC 2004; Lv & Zhai CIKM 2009) — a collection where top-F docs are usually relevant; conversational QA breaks that precondition. [paper]

### (c) Char n-gram / soft-tf — redundant with the stem channel

- Character-3-gram Jaccard neighbour expansion (θ=0.45, ≤10 neighbours/query-term, weight 0.5): +0.028 over *raw* BM25. [measured]
- **On the stemmed baseline the marginal is ≈0 (0.7606→0.7568, noise)** — verbatim already runs an FTS5 porter channel (`unit_fts_stem` posting collection, `verbatim/retrieval/v7/lexical.py:1084–1090`), so every morphology fix n-grams would deliver ('mentorship'↔'mentored' share stem 'mentor') is already covered by production code. Residual value ≈ typo repair only ('reccomendation'). [measured, code]
- Verdict: skip — pay a second postings index + fuzzy-match scoring for lift the stem channel already provides. SPLADE (learned sparse expansion) was already rejected for default in formula_search: +7–8 nDCG BEIR [paper] but naver checkpoints are CC BY-NC-SA [license] and 1–3ms/query [vendor] — same conclusion.

### (e) WordNet synonyms — measured negative

- NLTK WordNet+omw, top-2 synset lemmas per query term, weight 0.5, in-vocab only: +0.018 on raw; **−0.002 to −0.019 on stemmed** (0.6486 → 0.6293–0.6461). [measured]
- Synsets inject polysemy ('stress' → tenseness/strain/stress line), and the dialog register isn't dictionary-shaped — synonym expansion behaves like uncalibrated vocabulary noise on first-person small-talk. [inference]
- Verdict: reject. Any lexical resource worth bundling would need dialogue-domain tuning, which defeats "zero-cost local resource".

### (d) Dialogue-context — biggest single lever, but *document-side*, and the code already half-exists

- Query-side speaker append ('Did Jon get mentorship …' → '+Jon Gina'): +0.011. Speaker names add little because turns already carry them ('[time] Jon: I got mentored'). [measured]
- **Document-side ±1 same-session turn window (concatenate neighbours into the indexed unit): +0.115** — the single largest non-model fix. Fixes the dominant miss shape: bare-reply golds ('Alright, see you tomorrow!', 'Wow Jon, same here!') whose scoreable vocabulary lives in the adjacent turn. ±2 dilutes (0.676). [measured]
- **Zero-reindex variant — score propagation** `s'_i = s_i + w·max(s_j for j∈same-session neighbours)`: w=0.7 → **0.7284 (+0.079 over raw, +0.024 over stem alone… ~70% of the window effect)** with no index change, just a candidate-time neighbour sweep. [measured]
- Verbatim link: `POOLS.neighbor_window` exists but is consumed only in `verbatim/retrieval/v7/pack.py:770` as post-hoc context decoration (V7-12.05) — it never enters scoring. The machinery (session/day keys, window parameter) is already designed; the win requires moving context from pack-time to score/index-time. [code]
- Literature: this is context-window expansion for conversational QA — CAsT/TREC conversation-QA systems append dialogue history to queries/docs (Dalton et al., CAsT overview); SECOM (ICLR 2025, https://proceedings.iclr.cc/paper_files/paper/2025/file/e56f394bbd4f0ec81393d767caa5a31b-Paper-Conference.pdf) independently shows segment granularity beats turn granularity on LOCOMO. [paper]

### (a) potion-base-8M int8 static-embedding lane — works, modest marginal after the free fixes

- Alone on dev: **0.5637 any@10 / 0.6602 any@20 (int8)** vs BM25 0.589 — alone it is ≈ BM25, not better. [measured]
- **Fusion is the point**: score-sum `bm25_norm + α·cos`, α=1.2, cos top-60 injection → +0.040 over stem+window (0.7606→0.8005 fp32; 0.7954 int8). RRF60 (0.745) and max-fusion (0.761) both lose to score-sum. [measured]
- **potion-retrieval-32M is a regression here**: 0.4389 alone vs 8M's 0.5637 — retrieval tuning (MS MARCO/gooaq-style question→passage pairs) doesn't transfer to first-person dialogue; 32M is also 4× the bytes (32.3MB int8 vs 7.6MB). formula_search's b9 note recommends retrieval-32M as primary — measured evidence says **ship base-8M** for this distribution. [measured]
- Cost: model table 7.6MB int8 (29,528×256, MIT license) [vendor, repo note b9]; doc vectors 0.74MB/2877 items [measured]; encode 0.08ms p50 [measured]; scan ~0.3ms exact [measured]. Well inside the 500ms deadline budget — adds <1ms.
- Pure-cosine rescues only 19/123 true-gap tasks (median gold cos-rank 89) — the embed lane is a *complement*, not a saviour; it shines as a guaranteed candidate slot for zero-lexical-overlap golds (e.g. 'destress'→'stress relief' cos-rank 4, where score-fusion α still buries it under strong-but-wrong lexical hits). [measured]

## Residual after the full recipe (159 misses, stem+win+cos)

Post-recipe misses decompose as ~40% multi-hop *aggregation* ('both X and Y' — retrieval returns one side), ~25% temporal anchoring ('open_domain' chronology questions), ~15% cat5-style premise/preference inference, ~20% remaining deep paraphrase. These are lane problems (facts/typed/obs lanes emitting zero candidates — the sibling task), not paraphrase-bridging problems. Per-category progression stem → +win → +cos: multi_hop 0.548→0.635→0.754; open_domain 0.404→0.383→0.468; single_hop 0.674→0.846→0.853; temporal 0.726→0.743→0.771. [measured]

## Literature anchors (dialogue-QA retrieval)

- **LoCoMo paper** (Maharana et al., ACL 2024, https://aclanthology.org/2024.acl-long.747/): its RAG baselines retrieve with dpr / contriever (default) / dragon+ / openai over segment- and session-granularity units — `snap-research/locomo` `task_eval/rag_utils.py:38–93`. Dense bi-encoders are the reference answer to paraphrase bridging in this domain; the static-table lane is the local approximation of that lane. [paper, code]
- **DRAGON+** (facebook/dragon-plus): BERT-base dual-encoder, ~110M params, MARCO dev 39.0 / BEIR 47.4 [vendor HF card] — the quality ceiling for a learned retriever but ~40× the bytes of potion-8M int8 and a torch/ONNX runtime; violates the static-artifact constraint. The measured gap static-vs-dual-encoder on dialogue is worth a spot-check if the 8M lane under-delivers in production.
- **SECOM** (ICLR 2025): granularity of the memory unit matters — segment-level units beat turn/session units on LOCOMO (and Long-MT-Bench+) for both retrieval and E2E QA [paper]. Supports the ±1-context result from the index side: the right unit is bigger than one turn, smaller than a session.
- **CAsT / conversational QA**: TREC CAsT systems resolve context by appending dialogue history to the query/doc (Dalton et al., CAsT overview) [paper]. Measured here: query-side context (speaker names) is weak (+0.011); doc-side context is strong (+0.115) — consistent with the CAsT finding that the *utterance* needs its neighbours, not the question.
- **RM3 provenance**: Abdul-Jaleel et al. TREC 2004 (relevance models on TREC ad-hoc); Lv & Zhai CIKM 2009 (comparative study — RM3 consistently top). Gains were demonstrated on news-wire collections where feedback-set relevance is high; this precondition fails for zero-overlap dialogue queries. [paper]
- **SPLADE** (learned sparse expansion, b8 note): +7–8 nDCG BEIR [paper] but CC BY-NC-SA on the best checkpoints + 1–3ms/query [vendor, license] — rejected for the same reasons the formula_search already recorded.
- **LongMemEval** (a6 note): index-merge (facts as a BM25F field, i.e. doc expansion) beat rank-merge R@10 0.862 vs 0.754 — index-side expansion > fusion-side expansion; directly parallel to doc-context > query-expansion here. [paper]
- No prior RM3/PRF evaluation exists anywhere in `research/` (grep-verified) — this note is the first measurement.

## Reproduction details

- Scratch dir `/tmp/pb/` (parent VM: check for survival): `harness.py` (BM25Index + dev loader + baseline runner), `methods.py` (RM3, char-3gram, speaker, window index, WordNet, scorer), `embed.py` (potion encode/int8/fusion), `combo.py`, `stemmed.py`, `residual.py` (per-category + still-missing dump).
- Doc-context implemented as second postings index over `doc + neighbours(doc,±1,same session)` where session = ref prefix `(.+):D(\d+):(\d+)`; score-propagation variant propagates `w·max(neighbour score)` at candidate time — no re-index.
- Window index build cost ~O(N·3), same postings machinery; in SQLite terms this is "index `turn + LAG/LEAD(turn)` text" — expressible as a view/trigger on the unit table, or via the existing `neighbor_window` machinery at score time.
- potion-base-8M int8 encoding of all 2877 dev items + 777 queries: wall <10s including model load [measured]. The 8µs/query vendor number is C++-pinned; Python+model2vec measured 0.08ms p50 here [measured].
- Answerable-vs-adversarial split matters: cat5 tasks (213) are excluded from any@10 denominators — scoring them against a wrong-premise recall metric would double-count lane misses that are really premise-detection failures.

## Recommended stack (ranked by lift per added ms/MB)

1. **Stem channel**: already in production (`lexical.py:1084–1090`); measured +0.059 — verify it's actually nominated per query (4/8 lanes emit nothing today).
2. **±1-turn same-session context at index/score time** (+0.11): extend `neighbor_window` from `pack.py:770` decoration into the lexical lane's scoring (or re-index units as turn+neighbours). Zero new bytes, ~0ms. Alt: score-propagation w0.7 (+0.08) with no re-index.
3. **potion-base-8M int8 dense lane, score-sum fused at α≈1.2** (+0.04 marginal; +0.21 total over raw): 7.6MB MIT table + ~0.26KB/item doc vectors + <1ms. Give its top-10 a guaranteed pool slot — score-fusion alone still buries zero-overlap golds (cos-rank 4 → fused rank ~47).
4. **Skip**: RM3 (0/123, flag is dead code), char n-gram (marginal ≈0 after stemming), WordNet (negative), potion-retrieval-32M (regression on dialogue), speaker-append (+0.011).

## Verdicts

- **Ship ±1 context at score/index time** — +0.115 any@10 measured, 0ms, 0MB; the single highest lift/dollar in the set. Wire `POOLS.neighbor_window` (currently pack-time decoration, `pack.py:770`) into candidate scoring; score-propagation (w=0.7, +0.079) is the no-reindex fallback.
- **Ship potion-base-8M int8 as the dense lane** — +0.040 marginal over stem+window at 7.6MB + <1ms; measured fused total 0.7954 int8 / 0.8005 fp32 vs 0.589 raw baseline (+35%). Give its top-10 an unconditional pool slot (union-before-fusion): zero-overlap golds reach cos-rank ~4 but die at fused-rank ~47 under pure score-sum. **Prefer base-8M over retrieval-32M** — retrieval tuning is a measured regression (0.439 vs 0.564 alone) on dialogue and costs 4× the bytes; the formula_search recommendation should be updated.
- **Do NOT implement `prf_round`** — measured 0/123 true-gap recovery at all configs; feedback docs for zero-overlap queries are off-topic by construction. The `types_v7.py:411` flag is already dead; either delete it or leave unimplemented rather than spending the F+E re-query cost it implies.
- **Skip char n-gram and WordNet** — n-gram marginal ≈0 once the porter channel is active (redundant machinery, second postings index); WordNet synonyms are measured negative (−0.02) via polysemy noise on colloquial text.
- **Note for the sibling lane task**: residual 159 misses post-recipe are multi-hop aggregation / temporal / premise problems — the facts/typed/obs lanes emitting zero candidates matter more than any further lexical bridging.