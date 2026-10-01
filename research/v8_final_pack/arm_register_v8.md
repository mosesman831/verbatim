# V8 Arm Register — Overnight Measurement Pack (updated 03:00 UTC)

Post-fix baseline (HEAD 1c86f4f, LoCoMo dev 990q/2877 items):
item any@10 0.542-0.546, session 0.845-0.852, MRR 0.286-0.290, p50 325-658ms@500-2000ms, stmts/q ~4,529-4,905.
flat_bm25: item 0.587, session 0.884, MRR 0.398. All values below measured on real dev store by the named agent.

## §23 arm decisions — MEASURED

| §23 arm | options | measured | verdict | src |
|---|---|---|---|---|
| **graph.gate** | {on, off} | OFF: any@10 0.553 vs 0.542 ON, mrr 0.298 vs 0.282, p50 −306ms, −3,213 stmts/q; 0 graph-exclusive gold top-10 in ANY arm | **ABLATE lane** | a1 |
| graph.edge_types | {co_mention, same_session, temporal_near, adjacent_turn, causal} | adjacent_turn-only: 479 gold-reach, fused_any 541 (best of ALL arms incl. baseline); temporal_near 478/532; same_session 469/526; co_mention worst signal:cost; causal useless | if lane kept: adjacent_turn first, drop co_mention+causal | a1 |
| graph rescue (K_fan=8, batch unit_id IN, deadline checks) | {baseline, rescue} | units stmts 509→6.9, lane −43%, ok reachable — but ties OFF (0.552 vs 0.553) | patch works, lane still doesn't beat OFF | a1 |
| graph need-gate | spec's compound gate | degenerate: AND fires 0/990, OR fires 94.8% (canons≥2 nearly universal); scarcity cond `core_union<4·limit` fires 1/990 | gate as specced unusable | a1 |
| graph seed source | derived vs manifest vs fused-top20 | seeds not the bottleneck: all still saturate frontier, 0 exclusive top-k; better seeds → less surviving contribution | derived fine if lane kept | a1 |
| **lanes (LOLO, n=990)** | base MRR 0.2862 | −lex 0.1200 (−0.166), −dense 0.2236 (−0.063), −time 0.2790 (−0.007), −fuzzy −0.003, −obs +0.008, −graph +0.012, −typed +0.036, **−ent +0.077 MRR/+0.071 nDCG/+0.085 any@10** | ent net-HARMFUL (removes ~2/3 bm25 MRR gap); typed/graph/obs harmful; earners lex,dense | a5 |
| **rerank features (LOFO)** | 13 features, ~25ms stage | ent_idf removed +0.054 MRR (most harmful); rrf_norm +0.008; 7/13 dead (cov_idf_ctx,ident_exact,life_state,corrob,perspective_fit,event_pred,lane_agree); earners: cov_idf −0.058, phrase −0.050, speaker_match −0.006 | drop ent_idf + dead 7; rerank earns ~25ms via 3 features | a5 |
| **rerank.speaker_match_weight** | {0,0.2,0.4} | cat-5 hypothesis FAILS: both-holds 0.271 < 0.50 — u_speaker is lane-signal-gated (only ent/graph/obs emit), can't see displacers | **current (0.4)**; real fix = u_speaker visibility (b4 measuring) | a5→b4 |
| **lane_miss root cause (99)** | — | 60 produced_then_lost (54 cut at fused→scored rerank_pool=100, 21 at scored→pack ~64), 37 lane_cap (lex over-nominates mean 649 vs cap 200), 2 zero_overlap; eligibility/deadline/df-gate/nomination = 0 each | fix = nomination economy + rank windows (b5 measuring) | a5 |
| temporal.events_weights | {0.7/0.3,0.5/0.5,0.3/0.7} | a4 events-loosening 0.5/0.5 + subject_backfill: ndcg +.046, mrr +.053, any@10 .680 — best single-arm gain but 5 churn misses | ship with tuning (tighter gate or fusion compensation) | b3 |
| temporal mentions-union (V8-09.04) | | 4,258 rows, 0 misanchors; 3/3 rescue golds surface in-pool (rk1/2/5), 1 converts | **SHIP** + needs fusion weight to convert 2 | b3 |
| temporal fallback ordering (V8-09.06) | | scan_truncated 101→2, closes 3 misses, ~40% fewer rows | **SHIP** | b3 |
| temporal claims anchor (V8-09.05) | | 2026-misanchors 212→2, 98/98 relatives correct, ~0 recall effect | **SHIP for correctness** | b3 |
| temporal mention-beats-session (V8-09.08) | | fires, converts nothing beyond mentions-union | **SKIP** | b3 |
| temporal as_of (V8-09.01/02) | | mechanism correct (May→2022) but global now-re-anchor net NEGATIVE (−0.029 any@10) | **rescope: window-anchor-only / relative-expr queries** | b3 |
| context.mode/w/W/M_ctx/ctx_field_weight | sweeps | pending b1 | | b1 |
| facets.whole_share / dense_per_facet / group_maxpool | | pending b2 | | b2 |
| ent.per_canon_quota | {25,50} | pending b2 | | b2 |
| lexical.df_floor/K_rescue/rescue_rows/single_match + fusion.dense_form/lex_anchor + claims | | pending b5 | | b5 |
| scheduler.R_post/post_pool | | post-pool max(4×limit,64): −8.2ms, 0 recall change → SHIP | SHIP 4×limit | a4 |
| elig.cache_size | {64,128} | 20→0.03ms/hit (data_version+rowid-MAX+write_epoch sig) | SHIP 128 | a4 |
| lexical.stem_lru | | calls/q 4,467→13.3, lex −11.6ms, wall −14ms | SHIP | a4 |
| dense.B_max/embed_batch | | 512-row repack: scan 75.6→7.2ms, lane 91.8→19.9ms, candidates bit-identical | repack + coalesced write | a3 |
| dense.N_d | {0,5,10,20} | 0/137 misses uniquely rescued | N_d=0 | a3 |
| dense.tier | | f32-native to ~200K rows; sqlite-vec buys nothing | f32 block scan | a3 |

## Latency ledger

| stage | before | after fix | src |
|---|---|---|---|
| eligibility | ~20ms×2 | 0.03ms/hit cached | a4 |
| lexical | ~35ms | ~23ms (stem memo −11.6) | a4 |
| dense | ~92ms | ~20ms (repack) | a3 |
| graph | 305ms p50, partial 990/990 | ABLATED → 0 | a1 |
| post/fusion | ~10ms | −8.2ms post-pool trim | a4 |
| rerank feature stage | ~25ms | earned by 3 features only; dropping dead 7+ent_idf nets +MRR AND −ms | a5 |
| stmts/q | 4,529-4,905 | ~500 (fts5vocab+unit_doclen+lane batches) | a2 |
| **projected p50** | 325@500 / 658@2000 | **~150-200ms** with graph off + repack + caches; needs post-lane recheck | composite |

## W0 required artifacts — DELIVERED
- lane_miss_forensic.json ✅ (60 produced_then_lost / 37 lane_cap / 2 zero_overlap)
- rank_forensic LOLO+LOFO ✅ (jsons in dir)
- cat-5 hypothesis ✅ (0.271 < 0.50 → speaker arm = current)
- graph ablation ✅ (OFF dominates ON)

## Lane budget picture post-ablation (candidate §23 disposition)
Active lanes if graph+ent+typed+obs dropped as harmful: lex, dense, time, fuzzy(skipped 767/990 anyway), claims(pending b5). Lex+dense are the earners; ent needs rebuild not tuning (its harm is entity-IDF over-fire on wrong-entity turns — ent_idf 0.264 displacer-mean vs 0.153 gold-mean).

## Update 04:30 UTC — b1/b2/c1 landed

| arm | measured | verdict | src |
|---|---|---|---|
| **context.mode/w/W/M_ctx** | BEST: propagate+inject w=0.9 W=2 M_ctx=25: **+0.041 ans any@10** (0.583→0.624), +0.039 all@10, +0.057 any@20, +0.099 adversarial, lm −20%, rs −16%, abstain 58→48, delivered +44; cost +15ms ctx/+50ms e2e/2 stmts; 0 contamination on 149K audited edges | **SHIP pi-w0.9-W2-M25** | b1 |
| context.ctx_field (V8-06.03) | +0.023 any@10 but −0.006 mrr at 209-385ms/q | **REJECT** (s'+w·max dominates quality+14-25× latency) | b1 |
| multi-hop ship-set | decomp(≥2 canons+coord marker) + WHOLE_SHARE=0.5 + FACET_LANES={lex,ent}: +0.005 any/+0.007 all/+0.007 prop, mh +0.008 any/+0.009 prop, p50 −15ms, zero regressions | **SHIP** | b2 |
| V8-10.03 conjunctive joint | 22K promotions, 0 flips — co-mention pool already reachable | **DON'T SHIP** | b2 |
| ent.per_canon_quota {25,50} | quota50: mh all@10 −0.032, 30% churn | **DON'T SHIP** | b2 |
| pack.group_maxpool | catastrophic: mh all@10 0.087→0.000 w/ expand; session-sibling flood | **REJECT** | b2 |
| honest bound | cat-1 all@10 stuck 0.087 vs target 0.20 — it's unit-granularity/cross-session packaging, not ranking; next lever = wider units or session-aggregation packs | flag for V8 | b2 |
| AMB provider | verbatim_provider.py 429 lines, smoke-tested (50 docs → 500-600 ctx tok @k=10, raw_response=None, 0 user leakage, deterministic order) | ready-to-wire | c1 |
| AMB corrections | retrieve/ingest are SYNC (async = to_thread → threading.Lock); REGISTRY hardcoded zero-arg → env-var config; engine.recall(RecallRequest.valid_at_us=query_timestamp) NOT engine.search; harvest max_len=1200 SKIPS long pieces → must chunk ≤1000; require_review=False insufficient → operator admit pass via apply_transition; run_pending(scope=None) needs private _ingester; concurrency=1 | verified | c1 |

## Update 05:15 UTC — b4 landed

| arm | measured | verdict | src |
|---|---|---|---|
| **ent_idf→0 (THE cat-5 + overall fix)** | real-pass: cat-5 0.404→**0.493**, all 0.556→**0.617**, EVERY cat up (mh +.040, od +.064, sh +.070, temp +.040), verdict unchanged; ent0+sm0 → cat-5 0.521 offline. Premise displacers: ent_idf +0.268 mean-gap, gold actually wins cov_idf −0.044. 46 bm25-only cat-5 hits all surfaced in fused pool — residual is RERANK not recall | **SHIP ent_idf=0** (largest verified single-weight win) | b4 |
| u_speaker visibility fix | engagement 5.8%→99.1% works — but sm at any positive weight costs cat-5 (0.404→0.286 @0.4): cat-5 gold is 72-78% OTHER-speaker turns | ship fix ONLY with sm=0 or asymmetric variant | b4 |
| speaker_match weight | sm=0: cat-5 0.413 (best); answerable rises with weight +0.007-0.016; symmetric match net-negative | **sm=0** (or asymmetric +match only) | b4 |
| wrong-speaker penalty | delivery-FPR ~0.3-0.4% safe but cat-5 never better than baseline | not worth it | b4 |
| abstention floor variants | premise-coverage floor DEAD: fires 0/990 (asked-speaker loose topical support ubiquitous); every tightening loses refusals; pure weak-pool worse (0.221/0.206 vs 0.216/0.179) | keep premise_mm as-is; FPR is pool-strength problem | b4 |

Convergence: a5 LOFO said −ent_idf +0.054 MRR; a5 LOLO said −ent +0.077 MRR; b4 says ent_idf→0 real-measured +0.089 cat-5/+0.061 all. THREE independent measurements → ent machinery is the top kill-list item.

## Update 05:35 UTC — c2 landed (write-path)

| arm | measured | verdict | src |
|---|---|---|---|
| **indexrel — find_open instr() pre-filter** | prop_unstructured 450→~30ms/doc; conv-41 762.6→326.0 (−57%), exact compare still runs (zero false negatives) | **SHIP — #1 write fix** | c2 |
| checkpoint by WAL bytes | −33.5ms/doc vs every-8-jobs | SHIP | c2 |
| skipcomention at write | −6.5ms/doc + kills 843-unit ctx reload (matches a1 net-harmful at query) | SHIP (a1-consistent) | c2 |
| batchvec / defercons / misc batching | −9 / −10-15 / −8-16 ms/doc | SHIP | c2 |
| bulk tx batching | commits already 0.7ms — NOT the lever (2,930 stmts/doc dominate) | skip | c2 |
| composite4 | 762.6 → **270.5ms/doc** measured (bulk 269.9) | vs 150 target = 1.8× over | c2 |
| floor w/ unarmed fixes (aliases −35, rel_write −20, prescan −15) | ~190ms/doc modeled — still over | **<150 requires conversation-scoped job fusion** (drain sees conv as unit) | c2 |

## Update 07:05 UTC — c3 landed (scheduler + 100K)

| arm | measured | verdict | src |
|---|---|---|---|
| **R_post two-phase (cheap-first)** | any10 +0.03/+0.06 recall bump BUT deadline bounding WORSENS: a@1000/2000 overshoot 0, b/c overshoot 17-45; graph 64-iter granularity +~230ms past global wall | **DO NOT SHIP as specced** — needs hard per-iter preemption contract + global-wall guard | c3 |
| scheduler as recall lever | b/c@2000 any10 0.605/0.597 vs a 0.573 | only worth it for recall, not latency | c3 |
| **100K readiness** (103,575 units twin) | e2e p50 2,273ms (15× target); **eligibility breaks first**: make_eligible ~206ms + _watermarks sig ~275ms > target BEFORE lanes; dense 769ms linear, graph 609ms, lex 250ms (18K df-inflated noms), ingest drain ~1.2s/doc | **150ms needs: cheap cache sig (data_version/generation counter), approximate dense index or candidate subset, df floors, conversation-scoped drain** | c3 |
| chunked dense scan | can't help at 100K — gather+first-blocks dominate | skip | c3 |

## Update 08:30 UTC — b5 landed (lexical/fusion — CAPSTONE)

**SHIP SET measured: cooc + earners{lex,fuzzy,dense,time} + slim-rerank + df_floor=4 → any@10 0.686 (+0.149), MRR 0.476 (+0.196), lane_miss 75 (−27), p50 284ms (−66%) — BEATS flat_bm25 0.587.**

| arm | measured | verdict | src |
|---|---|---|---|
| **nom_cooc** (≥2 nominated terms AND-of-postings, fallback union 4/989) | +0.064 any/+0.053 MRR, union 674→110 mean, lex cands 197k→91k | **SHIP cooc_min=2** | b5 |
| **earners-only lanes {lex,fuzzy,dense,time}** | +0.095 any/+0.148 MRR, −52% p50 | **SHIP — biggest arm** | b5 |
| **slim rerank {rrf_norm,cov_idf,phrase,speaker_match,t_prox}** | +0.069-0.074 MRR | **SHIP** | b5 |
| **df_floor=4** | +0.024 any/+0.040 MRR, stacks on ship set | **SHIP** | b5 |
| df_theta {0.15,0.30,0.50} | θ0.15 +0.009 (weak); θ≥0.30 inert | skip — superseded by cooc | b5 |
| nominate_terms_max {8,16,24} | never binds (queries nominate ~4-5 terms) | leave 32 | b5 |
| **rank windows rerank_pool {150,200} / pack {96,128}** | NEGATIVE −0.01..−0.02 any: converts lane_miss→rank_shift but produced golds sit fused-rank 60-200, never top-10 | **KILL — keep rp=100,_MAX_LIMIT=64** | b5 |
| dense_form inversion | −0.004, 4 delivered→packed_out | keep capped | b5 |
| lex_anchor {1.10,1.15} | +0.008/+0.010 solo, redundant under earners | leave 1.0 | b5 |
| rescue_rows {25,50}, single_match | 0 firings — inert | leave off | b5 |
| **claims/typed lane (V8-13.01)** | 13% of pack slots while net-negative (+0.036 MRR removed); state_facts=156 rows | **KILL on this corpus** | b5 |
| −typed LOLO | +0.015 any/+0.036 MRR | confirms a5 | b5 |

Residual: ship set still leaves 75 lane_miss but ~12/102 base misses close directly — misses churn, not a shrinking fixed set. Rank windows can't fix produced_then_lost (golds at fused rank 60-200 never reach top-10) — needs score-quality not window-width.

**Stacking estimate**: b5's 0.686 does NOT include b1 ctx (+0.041), b3 temporal (+0.011), b2 multihop (+0.005) — orthogonal changes that should stack → composite ~0.70+ possible (c4 measuring interactions).

## Update 08:45 UTC — c4 landed (COMPOSITE — all 10 items, paired same-store)

**comp−typed−obs (full ship set): item any@10 0.6929 / session 0.8818 / MRR 0.4441 / ndcg 0.4729 / p50 185.3ms / stmts 445 — BEATS flat_bm25 on EVERY item-level metric (bm25: 0.5869 / 0.8838 sess / 0.3984 / 0.4181 / 4.3ms).**

| arm | any@10 | mrr | p50 | stmts-p50 |
|---|---|---|---|---|
| base64 (stock+freeze) | 0.5444 | 0.2815 | 669.5 | 4,896 |
| comp (−graph −ent +rest) | 0.6667 | 0.3792 | 222.2 | 496 |
| comp−typed | 0.6879 | 0.4408 | 197.8 | 451 |
| **comp−typed−obs** | **0.6929** | **0.4441** | **185.3** | **445** |
| flat_bm25 | 0.5869 | 0.3984 | 4.3 | — |

Sub-arm answer (item 2): remove ALL of ent, typed, obs — −typed is STRONGER in composite (+0.061 vs +0.036 alone, ×1.7).

Per-cat comp−t−o: adversarial .6573, multi_hop .5635, open_domain .4681, single_hop .7483, temporal .7543.
Attribution: delivered 525→601, lane_miss 100→30, abstain 54→50.

Interactions: any Σ +0.18-0.20 → composite +0.149 (~25-30% eaten, overlapping produced-then-lost rescues); mrr ≈ neutral; p50 amplified (−484 vs Σ −363 — stmt collapse compounds); temporal cat +0.063 (×5.7 — windows surface mention-union); **multi_hop all@10 REGRESSES −0.032** (0.087→0.056 — DECOMP fires on junk canons 'of'/'both' when ENT removed; fix = ≥2 REAL-entity-canon gate).

Latency: p50 185.3 vs 150 target → −35ms over; 3.6× faster than baseline.
Stage times comp−t−o: t_lanes 662.7→~125, t_ctxprop ~14, t_total 739.7→~200.

Reviewer notes: b3's restricted-fallback `EXISTS … MATCH` re-ran MATCH per unit → up to 400k probes/q; c4 rewrote as uncorrelated `IN (SELECT … MATCH)` — probes collapse ~200× (p99 stmts 265,828→3,265). Item-9 claims-anchor is inert by construction (no v7 read path consumes valid_intervals).
