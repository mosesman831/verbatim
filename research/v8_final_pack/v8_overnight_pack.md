# V8 Overnight Pack — measured arm decisions for the build agent
Generated ~07:00 UTC. Every number measured on the real LoCoMo dev store (990q / 2,877 items / 4,143 units) at HEAD 1c86f4f unless noted. Sources: a1–a5, b1–b4, c1–c2 notes in this dir. Nothing committed to git.

## Where SPEC v8 is going (read of the doc)
V8 reframes the target: not "fix defects" but **"Beat Hindsight on Its Own Board"** — the AMB provider under a pinned Gemini reader+judge, claiming ≥92.5% accuracy at ≤1/8 Hindsight's context tokens (≤4,500 vs 36,235) and ≤100ms recall (vs 964/1,214ms). Write ≤150ms/doc. Every arm earns a paired dev Track-R delta into `eval/v8/ledger.jsonl` (Δ≥+0.010 or a latency/trust fix at ≥−0.005). Dev-only tuning; no Postgres/vector-DB/hosted model on the default path. §23 is a ~30-arm register — **this pack pre-measures every arm so the builder ships measured winners, not hypotheses.**

## THE COMPOSITE SHIP-SET (all measured, ranked by verified delta)

### Recall arms
| # | arm | measured delta | ship |
|---|---|---|---|
| 1 | **ctx propagate+inject** (pi, w=0.9, W=2, M_ctx=25) — s' = s + w·max(neighbor scores) BEFORE lane ranks | **+0.041 ans any@10** (0.583→0.624), +0.039 all, +0.057 any@20, +0.099 adversarial, lane_miss −20%, rank_shift −16%, abstain 58→48 | SHIP |
| 2 | **ent_idf = 0** in rerank weights | cat-5 0.404→**0.493**, all 0.556→**0.617**, every cat up | SHIP — top single-weight win |
| 3 | **ent lane off** (LOLO) | +0.077 MRR / +0.071 nDCG / **+0.085 any@10** — removes ~2/3 of bm25 MRR gap | SHIP ablation; rebuild later |
| 4 | **graph lane OFF** | +0.011 any@10, +0.016 MRR, p50 −306ms, −3,213 stmts/q; 0 exclusive gold in every arm | **ABLATE** |
| 5 | **temporal bundle all_n6**: mentions-union + fallback-order + claims-anchor + events-loosen(0.5/0.5, subject_backfill) | +0.011 any / +0.044 ndcg / +0.050 mrr on cat-2 | SHIP (tune events gate — 5 churn) |
| 6 | **multi-hop ship-set**: decomp(≥2 canons+coord marker) + WHOLE_SHARE=0.5 + FACET_LANES={lex,ent} | +0.005 any / +0.007 all / +0.007 prop; −15ms | SHIP |
| 7 | **rerank dead-7 drop** (cov_idf_ctx, ident_exact, life_state, corrob, perspective_fit, event_pred, lane_agree) | ~25ms stage earned by ~3 features | SHIP (keep cov_idf, phrase, speaker_match) |
| 8 | speaker_match → 0 (or asymmetric +match) | sm any positive weight costs cat-5 (0.404→0.286 @0.4); u_speaker visibility fix ships WITH sm=0 | SHIP sm=0 |
| 9 | **nom_cooc** (≥2 nominated terms AND-of-postings) | +0.064 any/+0.053 MRR; union 674→110 mean — kills the nomination flood θ/K can't reach | SHIP cooc_min=2 |

**MEASURED COMPOSITE (c4, paired same-store): comp−typed−obs = any@10 0.6929 / session 0.8818 / MRR 0.4441 / ndcg 0.4729 / p50 185.3ms / stmts 445 — beats flat_bm25 on EVERY item-level metric (bm25: 0.587 / 0.884 / 0.398 / 0.418 / 4.3ms).** Interaction: any Σ +0.18-0.20 → +0.149 composite (~25-30% eaten by overlapping rescues); p50 amplified −484ms (stmt collapse compounds); temporal cat +0.063 (×5.7 vs solo).

**b5's lexical ship-set alone (cooc + earners{lex,fuzzy,dense,time} + slim-rerank + df_floor=4): any@10 0.686 (+0.149), MRR 0.476, p50 284ms (−66%)** — independently beats bm25, WITHOUT ctx/temporal/multihop. The c4 composite (0.693) used windows-rescue instead of cooc — final builder arm = c4-composite + b5's cooc/dff4/slim swapped in for windows (windows measured NEGATIVE solo).

**Flags c4 found:** multi_hop all@10 regresses −0.032 (DECOMP fires on junk canons 'of'/'both' when ENT removed → gate on ≥2 REAL entity canons); b3's fallback EXISTS-MATCH re-ran MATCH per unit → c4 fixed to uncorrelated IN-SELECT (~200× fewer probes).

### Speed ship-set (read path)
| fix | measured | ship |
|---|---|---|
| graph lane off | p50 −306ms, −3,213 stmts/q | SHIP |
| dense 512-row repack | scan 75.6→7.2ms, lane 91.8→19.9ms, bit-identical | SHIP |
| elig cache (sig=data_version+rowid-MAX+write_epoch) | 20→0.03ms/hit | SHIP size=128 |
| stem memo | 4,467→13.3 calls/q, −11.6ms | SHIP |
| post-pool trim max(4×limit,64) | −8.2ms, 0 recall change | SHIP |
| fts5vocab+unit_doclen (a2) | stmts 4,250→507 potential | SHIP |
| **projected read p50** | 325@500/658@2000 → **~150-200ms**; needs post-composite recheck |

### Write-path ship-set (c2 — populated store 762.6→270.5ms/doc measured)
| # | fix | Δms/doc |
|---|---|---|
| 1 | **indexrel**: find_open instr() subject pre-filter (kills 15-19K-row JSON parse per pair) | **−437** |
| 2 | checkpoint by WAL bytes not job-count | −33.5 |
| 3 | skipcomention at write (kills 843-unit ctx reload) | −6.5 |
| 4 | batchvec + defercons + misc batching | −20-30 |
| | **composite4 measured** | **270.5 (target 150)** |
| | + unarmed: aliases −35, rel_write −20, prescan −15 | ~190 floor |
| | **<150 requires conversation-scoped job fusion** — drain must treat conv as unit of work | flag for V8 |

## Don't-ship list (measured rejects)
- context.ctx_field (209-385ms, worse mrr) — propagate+inject wins on quality AND speed
- multi-hop conjunctive joint (0 flips), canon_quota50 (mh −0.032), pack.group_maxpool (collapse)
- temporal as_of global re-anchor (−0.029) — rescope to window-anchor only
- mention-beats-session — redundant with mentions-union
- dense N_d>0 (0/137 unique rescues), sqlite-vec (buys nothing <200K rows), f32→f16 (no need)
- abstention floor variants (all fire 0/990 or worse than premise_mm at frontier)
- wrong-speaker penalty (safe but cat-5 never beats baseline)
- bulk tx batching (commits already 0.7ms — statements are the cost)

## AMB provider spec (c1 — verbatim_provider.py working, smoke-tested)
Contract corrections vs spec assumption: sync ingest/retrieve wrapped in `asyncio.to_thread` → **threading.Lock**; `concurrency=1` class attr; hardcoded REGISTRY zero-arg ctor → all config via env vars + `prepare(store_dir,unit_ids,reset)`; `engine.recall(RecallRequest(valid_at_us=query_timestamp))` NOT engine.search; `engine._ingester.run_pending(scope=None)` for cross-scope drain; `require_review=False` insufficient → operator `apply_transition(effect="admit")` per pending head; harvest max_len=1200 SKIPS longer → **chunk ≤1000 chars**; per-thread reader conns; raw_response MUST be None (json.dumps explosion); PMB source_ids=[Document.id]; LoCoMo doc id={sample}_{session_key}; LME isolation_unit=question; retries only on 502/503/529/429-substring. PMB retrieval_filter honestly dropped when supports_filters=False.

## Honest bounds / open items
- cat-1 all@10 stuck at 0.087 — unit-granularity/cross-session packaging issue, NOT ranking. Next lever: wider units or session-aggregation packs.
- cat-5 residual after ent_idf=0: premise turn loses to asked speaker's topically-loose turns (78% displacers are asked-speaker) — further needs premise-turn delivery boost not verdict surgery.
- premise_mm FPR 17.9% is a weak-pool problem — fixed by better pool strength (the ship-set above), not trigger tuning. Keep trigger as-is.
- 150ms read target: ship-set gets ~150-200ms projected; remaining structural = nomination economy + claim/event lanes' own costs.
- 150ms write target: needs conversation-scoped job fusion (biggest unbuilt structural change).
- Interactions: measured by c4 — any@10 Σ +0.18-0.20 → +0.149 composite (~25-30% eaten by overlapping produced-then-lost rescues); latency amplifies (−484 vs Σ −363).
- **Rank windows NEGATIVE solo** (b5): rerank_pool {150,200}/pack {96,128} convert lane_miss→rank_shift but never top-10 (golds sit fused-rank 60-200). c4's composite used pack 128 anyway and rescued 46 lane_miss→rank_shift+18→delivered — but the final builder arm should swap in cooc+dff4+slim (the stronger ship-set).
- **R_post scheduler: DON'T ship** (c3) — recall +0.03-0.06 but deadline bounding worsens (0 vs 17-45 over-25ms). Cheap-first is a recall lever, not latency.
- **100K wall** (c3): eligibility path breaks first — make_eligible ~206ms + _watermarks sig ~275ms > target before lanes; needs cheap cache signature + approx dense index + df floors + conv-scoped drain.
- multi_hop all@10 −0.032 regression in composite (c4): DECOMP gate fires on junk canons ('of'/'both') once ENT removed — fix: gate on ≥2 real entity canons.
