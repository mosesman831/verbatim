# V8 §23 lexical/fusion arm measurements — real dev store (locomo dev, 990 tasks / 2,877 items / 4,143 units)

**Store**: `/tmp/v7w/mem.db` rebuilt once by `eval.v7.track_r --dataset locomo --split dev --arms verbatim,flat_bm25` (~50min wall incl. query phase; DB static since 03:25 UTC, 249MB, encrypted — `mem.db.key` required beside the file). Arms attach to the prebuilt store via a `PrebuiltArm` that rebuilds `_source_ref` by byte-matching `source_revisions.payload` → `item_document()` — no re-ingest per arm. All arms run as separate processes (monkeypatch isolation), `now_us` frozen at 1758000000000000.

**Harness validation**: my `base` arm = any@10 **0.5374**, MRR **0.2806**, lane_miss **102** vs the same-store `track_r` reference 0.5404 / 0.2834 / 104 — reproduces within frozen-clock noise (±0.003). Parent's stated baseline 0.546/0.290/~99-104 consistent. flat_bm25 reference: 0.5869 / 0.3984.

**Baseline per-stage gold-loss classification** (forensics pass, all 990 tasks, gold-level): of the 102 lane_miss tasks' 111 gold source_ids — 80 reach `fused` but die at `rerank_pool=100`, 30 reach `scored` but die at the pack ~64 window, 1 sits in pack, 29 never appear in lex lane output (all-gold-absent tasks: 13). Nomination union: mean **673.8**, max 2877; lex candidates at lane_cap 200 ≈ every task (197,299 total). Queries nominate only ~4-5 terms each (`nominated_terms`≈4-5) — the flood is per-term posting *union*, not term count. df landscape (lex_df, N_eligible=2877 source rows): median df 3, mean 29.6; top terms are stopword/time tokens ('on' 3647, 'pm' 2560, '2023' 2168, 'a' 2095, 'it' 2062, 'and' 1825); speaker-ish mid-df terms sit at ~500-800 ('john' 630). Baseline prefetch gate already blocks df > max(1400, 0.2·N_E) before fetch.

## Arm 1 — nomination economy

| arm | any@10 | Δ | mrr@10 | Δ | lane_miss | nom_mean | notes |
|---|---|---|---|---|---|---|---|
| base | 0.5374 | — | 0.2806 | — | 102 | 673.8 | |
| θ=0.15 (df>431 cut) | 0.5465 | +0.009 | 0.2881 | +0.008 | 101 | **514.6** | bites (−24% union), 487 term-drops; small gain |
| θ=0.30 (df>863) | 0.5384 | +0.001 | 0.2866 | +0.006 | 105 | ~674 | cuts only ~15 terms — inert |
| θ=0.50 (df>1438) | 0.5414 | +0.004 | 0.2824 | +0.002 | 103 | ~674 | ≈ prefetch-floor baseline — inert |
| top-K=8 terms | 0.5404 | +0.003 | 0.2890 | +0.008 | 107 | 673.8 | K never binds (queries nominate ≤5 terms) |
| top-K=16 | 0.5414 | +0.004 | 0.2836 | +0.003 | 103 | 673.8 | inert |
| top-K=24 | 0.5434 | +0.006 | 0.2880 | +0.007 | 103 | 673.8 | inert |
| **cooc ≥2** | **0.6010** | **+0.064** | **0.3337** | **+0.053** | **87** | **109.8** | union 674→108 (6.2×), lex cands 197k→91k; 4/989 tasks fell back to union (empty cooc) |
| cooc + θ0.30 | 0.6030 | +0.066 | 0.3366 | +0.056 | 88 | — | θ adds ~nothing once cooc intersects |

Transition detail (cooc): 25 rank_shift→delivered + 16 abstain→delivered + 1 packed_out→delivered gains vs 5 delivered→rank_shift + 3 →abstain losses; delivered→delivered retention 527/535. Cooc's gains come from *cleaner mid-rank ordering* (the intersected union concentrates multi-term units) more than lane_miss closure.

## Arm 2 — rank windows (all NEGATIVE on top-10; only reclassify misses)

| arm | any@10 | Δ | mrr@10 | Δ | lane_miss | rank_shift | lat_p50 |
|---|---|---|---|---|---|---|---|
| base | 0.5374 | — | 0.2806 | — | 102 | 81 | 843ms |
| rerank_pool 150 | 0.5263 | −0.011 | 0.2741 | −0.007 | 104 | 81 | 747ms |
| rerank_pool 200 | 0.5212 | −0.016 | 0.2690 | −0.012 | 105 | 83 | 1002ms |
| pack window 96 | 0.5404 | +0.003 | 0.2856 | +0.005 | 79 | 102 | 1104ms |
| pack window 128 | 0.5414 | +0.004 | 0.2867 | +0.006 | 74 | 108 | 810ms |
| rp150 + pw96 | 0.5273 | −0.010 | 0.2725 | −0.008 | 80 | 105 | 838ms |
| rp200 + pw128 | 0.5172 | −0.020 | 0.2677 | −0.013 | 63 | 122 | 869ms |
| θ030 + rp150 + pw96 | 0.5273 | −0.010 | 0.2671 | −0.014 | 79 | 106 | 824ms |
| θ030 + rp150 + pw96 + slim | 0.5929 | +0.056 | 0.3460 | +0.065 | 77 | 85 | 1070ms |

Widening windows converts lane_miss → rank_shift (produced golds surface at depth 64-128 but never top-10) — `produced_then_lost` golds sit at fused rank ~60-200 and rerank can't lift them into the top-10. Combined with θ: still negative. Kill this arm family.

## Arms 3-5 — lexical micro-arms

| arm | any@10 | Δ | mrr@10 | Δ | lane_miss | measured extra |
|---|---|---|---|---|---|---|
| df_floor=3 | 0.5485 | +0.011 | 0.3079 | +0.027 | 101 | rare stems rescued into union |
| df_floor=4 | 0.5616 | +0.024 | 0.3208 | +0.040 | 102 | bigger rescue budget, no junk flood observed |
| rescue_rows=25 | 0.5384 | +0.001 | 0.2823 | +0.002 | 105 | rescue_added=0 on sampled tasks — 1-2-unit terms already nominate |
| rescue_rows=50 | 0.5394 | +0.002 | 0.2892 | +0.009 | 103 | same — inert on this corpus |
| single_match | 0.5384 | +0.001 | 0.2807 | 0.000 | 105 | `single_match=1` fires on 0/184 miss-subset tasks — queries are all multi-term; inert |

## Arm 7 — fusion.dense_form

| arm | any@10 | Δ | mrr@10 | Δ | packed_out |
|---|---|---|---|---|---|
| capped (base) | 0.5374 | — | 0.2806 | — | 1 |
| inversion | 0.5333 | −0.004 | 0.2832 | +0.003 | 4 |

Dense-unique candidates are ~15% of fused (90,985/622,170) baseline, ~54% under the ship set — but dropping dense-only pool members trades 4 delivered→packed_out losses for ~nothing. Capped wins; dense-unique golds aren't reachable by pack anyway.

## Arm 8 — fusion.lex_anchor (multiplier on flat 1.0 base, all intents)

| arm | any@10 | Δ | mrr@10 | Δ |
|---|---|---|---|---|
| lex ×1.10 | 0.5455 | +0.008 | 0.2873 | +0.007 |
| lex ×1.15 | 0.5475 | +0.010 | 0.2885 | +0.008 |

Marginal positive; absorbed entirely once cooc/earners ship (lex's relative share already rises).

## Arm 9 — claims plane (V8-13.01)

Claims tables on real dev store: `state_facts` 156 rows (mostly empty as predicted), `valid_intervals` 7,122 (structural, 1/claim), `t2_facts` present. Typed lane pack provenance (all 990 tasks): **8,220 pack slots (~13% of ~63k)** — claims-sourced candidates DO reach pack, they're just not earning:

| arm | any@10 | Δ | mrr@10 | Δ | lane_miss |
|---|---|---|---|---|---|
| base | 0.5374 | — | 0.2806 | — | 102 |
| −typed only | 0.5525 | +0.015 | +0.3165 | +0.036 | 107 |
| earners {lex,fuzzy,dense,time} | **0.6323** | **+0.095** | **0.4284** | **+0.148** | 84 |

LOLO replicates: −typed alone ≈ +0.036 MRR as predicted; the remaining +0.11 comes from cutting ent/graph/obs (ent held 48,810 pack slots baseline — 77% — while being net-harmful).

## Arm 10 — rerank slimming

| arm | any@10 | Δ | mrr@10 | Δ | lat_p50 |
|---|---|---|---|---|---|
| base 13-feat | 0.5374 | — | 0.2806 | — | 843ms |
| drop 8 {cov_idf_ctx, ident_exact, life_state, corrob, perspective_fit, event_pred, lane_agree, ent_idf} | 0.6030 | +0.066 | 0.3506 | +0.070 | 773ms |
| keep {rrf_norm, cov_idf, phrase, speaker_match, t_prox} (identical set) | 0.6061 | +0.069 | 0.3542 | +0.074 | 1044ms |

LOFO replicates exactly: the 5 kept features are the earners; dead-weight + ent_idf removal is worth +0.07 MRR. (keep4 vs slim Δ is run noise — same feature set.)

## Ship-set interaction table

| combo | any@10 | Δ | mrr@10 | Δ | lane_miss | lat_p50 |
|---|---|---|---|---|---|---|
| cooc + earners | 0.6848 | +0.147 | 0.4562 | +0.176 | 74 | 317ms |
| cooc + slim | 0.6444 | +0.107 | 0.3946 | +0.114 | 77 | — |
| earners + slim | 0.6313 | +0.094 | 0.4342 | +0.154 | 87 | 429ms |
| **cooc + earners + slim** | **0.6859** | **+0.149** | **0.4619** | **+0.181** | **75** | **370ms** |
| **cooc + earners + slim + dff4** | **0.6859** | **+0.149** | **0.4762** | **+0.196** | **75** | **284ms** |

Ship set vs base: +0.149 any@10 / +0.196 MRR / −27 lane_miss / −66% p50 latency. Transitions on the full ship: 54 rank_shift→delivered + 29 abstain→delivered + 12 lane_miss→delivered gains vs 18 delivered losses; retention 517/535. Pack provenance under ship: dense 48,462 / lex 44,316 / time 13,526 / fuzzy 150 — clean earners-only mix.

Caveat: arms *churn* misses rather than pure-closing them — best combo still leaves 75 lane_miss, and only ~12 of base's 102 close directly. The residual miss population is different tasks, not a shrinking fixed set.

## Verdicts — nominated ship set + constants (§23 lexical/fusion arms)

**SHIP:**
1. `lexical.nom_cooc = true` — require ≥2 nominated terms co-occurring (AND-of-postings), fallback to union when empty (4/989 tasks). Union 674→110 mean; kills the flood properly where θ/K can't reach. Constants: `cooc_min = 2`.
2. `lanes = [lex, fuzzy, dense, time]` (earners-only) — the single biggest arm: +0.095/+0.148, −52% p50 latency. Drop ent, graph, obs, typed.
3. `rerank.features = {rrf_norm, cov_idf, phrase, speaker_match, t_prox}` — drop cov_idf_ctx, ident_exact, life_state, corrob, perspective_fit, event_pred, lane_agree, ent_idf. +0.07 MRR, ~25ms+ stage cost saved.
4. `lexical.df_floor = 4` — rare-stem rescue at df<4; +0.024 any / +0.040 MRR, stacks on the ship set (+0.014 MRR over cooc+earners+slim). df_floor=3 smaller but positive.

**KILL / leave as-is:**
- `nominate_df_theta`: θ≤0.15 barely bites (cuts ~45 terms, −24% union, +0.009) — superseded by cooc; θ≥0.30 inert. Leave `None`.
- `nominate_terms_max`: K∈{8,16,24} never binds (queries nominate ~4-5 terms). Leave 32.
- Rank windows: rerank_pool {150,200} and pack window {96,128} only convert lane_miss→rank_shift, Δany@10 −0.01..−0.02, +latency. Keep `rerank_pool=100`, `_MAX_LIMIT=64`.
- `dense_form`: keep `capped`; inversion drops real pack members (4 delivered→packed_out) for no gain.
- `lex_anchor` 1.10/1.15: +0.008/+0.010 in isolation but redundant under earners lanes — leave 1.0 (or set 1.10 if a future eval shows it surviving the ship set).
- `rescue_rows` {25,50} and `single_match`: inert on this corpus (0 firings/adds) — leave off.
- **Claims lane (V8-13.01): KILL on this corpus** — typed holds 13% of pack slots while net-negative (+0.036 MRR when removed); state_facts is 156 rows so the lane fires on structural/t2 facts only. Revisit when real claim volume lands.

Ship set = cooc + lanes{lex,fuzzy,dense,time} + slim-rerank + df_floor=4 → **any@10 0.686 (+0.149), MRR 0.476 (+0.196), lane_miss 75 (−27), p50 284ms (−66%)** vs base on the same store/clock.

## Method notes
- Store: real dev store, `/tmp/v7w/mem.db` (2,877 sources / 4,143 units / 7,122 claims); per-arm procs open per-worker copies (WAL multi-reader), no re-ingest.
- Paired: `now_us` frozen @ 1758000000000000; `search(limit=20)` measured + `search(limit≤_MAX_LIMIT)` surfaced probe per task; deterministic seed=0.
- Residual churn is real: miss sets rotate across arms (ship still misses 75; 12/102 base misses closed directly). lane_miss reductions partly reflect class migration, not just closure.
- Forensics pack-provenance counts are lane×item slots (multi-lane items count per lane), not unique items.
- Raw artifacts: `/tmp/v7results/<arm>.json` (full track_r reports + arm_config), `/tmp/v7results/fore_*.json` (stage-level gold maps + lane stats), `/tmp/v7cfgs/*.json` (arm configs), drivers `/tmp/v7drive/{run_arm.py,forensics.py,sweep.sh,make_cfgs.py}`, proto patches in `/tmp/verbatim_proto/verbatim/retrieval/v7/{_proto,lexical,pipeline}.py` (env-gated, repo untouched).
