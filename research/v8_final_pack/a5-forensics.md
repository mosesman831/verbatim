# V8 forensic artifacts — LoCoMo dev store, verbatim vs flat_bm25

**Store**: rebuilt at repo HEAD `1c86f4f` via `eval.v7.track_r --dataset locomo --split dev --arms verbatim,flat_bm25` (workdir `/tmp/v7w`, 2877 items / 990 tasks / 777 answerable / 213 cat-5). Fresh-run numbers (drift vs the spec's post-fix targets is expected): verbatim attribution = delivered 529 / **lane_miss 99** / rank_shift 91 / packed_out 3 / abstain 55 / unsupported 213; MRR@10 **0.2862** vs flat_bm25 **0.3984**; cat-5 premise any@10 **0.399** vs **0.577**.

**Method & fidelity**: all measurements replay the real pipeline on the rebuilt store — `LaneContextV7` assembled exactly as `Memory._search_v7` (same eligibility, budget MID, policy `default`, deadline ~2s, manifest limit 64/20), `run_search` per question, pack items → corpus refs via the deterministic `external_id` hmac mapping (2877/2877 items resolved; first-ref-wins dedup mirrored). Phase-0 check: 38/40 replays reproduce the report's delivered list exactly; 2 differ by one ref (deadline-slicing jitter). LOLO base MRR 0.2862 = report 0.2862.

Artifacts on disk: `/tmp/lane_miss_forensic.json`, `/tmp/rank_forensic_displacers.json`, `/tmp/lolo_mrr_ndcg.json`, `/tmp/lofo_mrr_ndcg.json`, `/tmp/cat5_hypothesis.json`.

## 1. lane_miss_forensic (V8-07.01) — 99 answerable questions

Each question is classed by the FIRST stage at which its *best-surviving* gold unit died — the fix that would rescue the question (deepest stage any gold unit reached). Per-unit stages and term fates are in the JSON.

| class | questions | spec bucket | evidence |
|---|---:|---|---|
| produced_then_lost | 60 | (g) other, post-lane | gold in lex lane output (≤cap 200), lost downstream — 54 units cut at fused→scored `rerank_pool=100`, 21 units cut at scored→pack surfaced window (~64) |
| lane_cap | 37 | (c) | nominated but ranked below the lane's top-200 output |
| zero_overlap | 2 | (d) | gold shares only the stopword "the" (non-content terms never nominate); real lexical overlap = 0 |
| eligibility | 0 | (f) | confirmed 0 — as required |
| deadline | 0 | (e) | no lane timeout was the binding loss |
| df_pre_gate | 0 | (a) | never all-terms-gated (gating fires on some term in 55/99 but never kills every overlap term) |
| nomination_budget | 0 | (b) | `nomination_dropped` fired on only 4 questions and never took all overlap terms |

Question-level produced_then_lost split: 41 rerank_pool-only, 15 pack-window-only, 4 both.

**Mechanism**: the lex lane over-nominates. Mean nominated/scored set = 649 units per lane_miss query (median 620) against `lane_cap=200` (mean overflow 449) — high-df content terms (speaker names like "jon"/"gina", df~500+) nominate hundreds of turns, so gold survives nomination easily but then dies by rank-position at the lane cap (37q) or at the fused→scored→pack boundaries (60q). Lane statuses: lex ok 99/99, graph partial 99/99 — no lane errored; this is a ranking-economics failure, not a production failure.

**Ranked fix order**: (1) post-lane rank-window attrition — rerank_pool 100 / pack-64 window cut gold already in lane output (60q); (2) lane-cap attrition under over-nominated lex (37q) — same root: nomination economy / term-IDF weighting upstream shrinks both; (3) effective-zero-overlap (2q — ctx/dense routing can absorb these). Nothing to fix at eligibility/deadline/df-gate/budget — all zero.

## 2. rank_forensic (V8-11.01) — 139 questions (bm25 gold@1, verbatim gold≠1)

Verbatim's gold sits at rank 2–4 in 83/139 questions (rank2: 33, rank3: 30, rank4: 20); only 12 absent from the surfaced window entirely. This is a top-of-list discrimination problem, not recall.

Displacer lane provenance (top-10 non-gold scored items): lex 1382, ent 1276, dense 1153, time 547, graph 424, typed 228, obs 79.

Mean feature contributions, displacers vs gold:

| feature | displacer mean | gold mean | weight |
|---|---:|---:|---:|
| ent_idf | **0.264** | 0.153 | 0.6 |
| t_prox | 0.201 | 0.188 | 0.4 |
| speaker_match | 0.229 | 0.243 | 0.4 |
| rrf_norm | 0.746 | 0.776 | 1.0 |
| cov_idf | 0.289 | **0.421** | 0.9 |
| phrase | 0.146 | **0.231** | 0.3 |
| lane_agree | 0.067 | 0.091 | 0.3 |

Displacers win on `ent_idf` (entity-IDF over-firing on wrong-entity turns); gold should be winning on `cov_idf`/`phrase` but the margins don't survive the weight mix.

**Leave-one-lane-out** (lane removed from the POLICY TUPLE via `ablation_lanes`, full `run_search`, n=990):

| arm | mrr@10 | Δmrr | ndcg@10 | Δndcg | any@10 |
|---|---:|---:|---:|---:|---:|
| base | 0.2862 | — | 0.3267 | — | 0.5364 |
| −lex | 0.1200 | −0.166 | 0.1453 | −0.181 | 0.2798 |
| −dense | 0.2236 | −0.063 | 0.2648 | −0.062 | 0.4737 |
| −time | 0.2790 | −0.007 | 0.3194 | −0.007 | 0.5273 |
| −fuzzy | 0.2829 | −0.003 | 0.3240 | −0.003 | 0.5364 |
| −obs | 0.2941 | +0.008 | 0.3346 | +0.008 | 0.5485 |
| −graph | 0.2982 | +0.012 | 0.3395 | +0.013 | 0.5535 |
| −typed | 0.3218 | +0.036 | 0.3541 | +0.027 | 0.5444 |
| −ent | 0.3628 | **+0.077** | 0.3980 | **+0.071** | 0.6212 |

**Leave-one-feature-out** (rerank weights minus one feature, faithful S4–S8 replay on captured `res.fused`, n=990; base replay mrr 0.2871 / ndcg 0.3272 — the ~25ms the feature stage costs buys this):

| removed | mrr@10 | Δmrr | Δndcg | verdict |
|---|---:|---:|---:|---|
| ent_idf | 0.3413 | **+0.054** | +0.052 | harmful — biggest single win available |
| rrf_norm | 0.2954 | +0.008 | +0.003 | mildly harmful |
| t_prox | 0.2892 | +0.002 | +0.003 | neutral-negative |
| cov_idf_ctx, ident_exact, life_state, corrob, perspective_fit, event_pred, lane_agree | 0.2865–0.2871 | ±0.001 | ±0.003 | dead weight on this corpus |
| speaker_match | 0.2815 | −0.006 | −0.006 | mild earner |
| phrase | 0.2373 | −0.050 | −0.048 | earns |
| cov_idf | 0.2289 | −0.058 | −0.056 | earns |

## 3. Cat-5 hypothesis test (V8-11.02) — study set n=48 of 213 cat-5

Study set = questions where flat_bm25 puts a premise turn top-10 and verbatim does not (verbatim cat-5 premise any@10 = 0.399 vs bm25 0.577 overall).

| condition | n | rate |
|---|---:|---:|
| (a) premise-turn speaker ≠ asked speaker | 39/48 | 0.813 |
| (b) `speaker_match` == 1.0 fired on ≥1 displacer | 14/48 | 0.292 |
| **both hold** | **13/48** | **0.271** |

**0.271 < 0.50** — the hypothesis fails the gate. Deeper mechanism finding: displacer `speaker_match` values are 439×0.5 neutral, 39×1.0, 2×0.0 — because `u_speaker` resolves ONLY from lane `signals["speaker_canon"]` (ent/graph/obs emit it; lex/dense don't) — the feature can't even see most displacers' speakers, and premise units themselves score 0.5 in 44/48 questions. The speaker-mismatch structure is real (81% of premises are the other speaker), but `speaker_match` doesn't act on it — it neither fires on displacers nor penalizes the premise.

## Verdicts

1. **lane_miss ranked fix order**: (i) post-lane rank-window losses (rerank_pool=100 cut for 54 units / surfaced-64 pack cut for 21 units → 60 questions); (ii) lex lane-cap attrition under nomination flood — mean 649 nominated vs cap 200 (37 questions), same root cause as (i) and the lever that shrinks both; (iii) zero effective overlap (2). Eligibility = 0 as required; deadline, df pre-gate, and nomination budget each = 0 — no fix needed there.
2. **LOFO/LOLO losers**: `ent` lane is net-harmful (−ent: +0.077 MRR / +0.071 nDCG — removing it alone recovers ~2/3 of the 0.112 bm25 MRR gap); `typed` harmful (+0.036); `graph`/`obs` mildly harmful (+0.012/+0.008). At feature level `ent_idf` is the most harmful weight on the board (−ent_idf: +0.054 MRR), `rrf_norm` mildly harmful (+0.008), and 7 of 13 features are dead weight (±0.001). Earners: `lex` (−0.166) and `dense` (−0.063) lanes; `cov_idf` (−0.058) and `phrase` (−0.050) features; `speaker_match` a mild earner (−0.006). The ~25ms of rerank is earned by essentially 3 features.
3. **Speaker-weight arm decision: `current`** — both-holds rate 0.271 is far below the 0.50 threshold. Halving or zeroing `speaker_match` cannot fix cat-5: the feature's `u_speaker` is lane-signal-gated (only ent/graph/obs-sourced units carry `speaker_canon`), so it engages on <10% of displacers and never penalizes wrong-speaker premise turns. If the speaker mechanism is ever wanted for cat-5, the missing piece is u_speaker visibility, not weight — but per the measured gate, the arm stays `current`.
