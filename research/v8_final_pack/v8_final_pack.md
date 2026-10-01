# V8 FINAL PACK — build agent spec (ship-set + AMB baseline + fix path)

Consolidates: (a) V8 overnight ship-set measurements, (b) the first full AMB locomo10
board run on the production engine, (c) the failure decomposition that tells the
builder WHERE the 65-point accuracy gap lives. Nothing committed to git.

Headline: **verbatim @ HEAD 1c86f4f scores 35.0% on AMB locomo10 (539/1540). The
cause is retrieval coverage, not the reader — 97.1% of wrong answers never had the
gold fact in context. The measured ship-set below is the fix.**

---

## 1. The board state — two honest baselines

### AMB locomo10 (vectorize-io/agent-memory-benchmark, production engine, FULL run)
- **accuracy 35.0% (539/1,540)** · retrieve mean 79.6ms / p50 65ms · ~494 ctx tokens/q · 0 hard errors
- by question_type: open-domain 41.4% (841q) · multi-hop 38.5% (96q) · single-hop 28.0% (282q) · temporal 23.4% (321q)
- reader+judge: `kilo/inclusionai/ling-3.0-flash-fin:free` on the user's OpenAI-compatible gateway
  (Nvidia-upstream models — incl. nemotron-3-ultra-550b — were 503ing; mimo-v2.6-flash also worked earlier)
- artifact: `eval/amb/locomo10_verbatim_ling-flash-fin.json` (merged per-query results + summary);
  shard outputs `eval/amb/amb_locomo10_shards.zip`
- Hindsight's published AMB rows for scale: ~2,565–2,968ms retrieve, ~23–36K ctx tokens.
  **Verbatim is already ~35× faster and ~50× cheaper — the entire open gap is accuracy.**

### Internal Track-R dev split (same HEAD): verbatim 0.546 any@10 / 0.852 session vs
flat_bm25 0.587 / 0.884 — consistent with the AMB number; the engine trails its own
lexical floor at HEAD, before the ship-set lands.

### Failure decomposition (the number that decides what to build)
- 97.1% of wrong answers: gold fact NOT in the retrieved context → **retriever failed**
- 1.7%: gold in context, reader answered wrong → reader error (negligible)
- Pack budget is NOT the limiter: TARGET_TOKENS=4096 cap, only ~494 delivered —
  ~20-item packs under-fill because lanes+packaging produce too few survivors at HEAD.
- Reader/judge model swap buys ~2 points max. **Retrieval coverage is the whole game.**

## 2. THE MEASURED SHIP-SET (all on the real dev store, 990q/2877 items/4143 units)

### Recall arms — ranked
| # | arm | measured delta | ship |
|---|---|---|---|
| 1 | **ctx propagate+inject** (pi, w=0.9, W=2, M_ctx=25): s' = s + w·max(neighbor) BEFORE lane ranks | **+0.041 ans any@10** (0.583→0.624), +0.099 adversarial, lane_miss −20% | SHIP — biggest single lever |
| 2 | **ent_idf = 0** in rerank weights | cat-5 0.404→**0.493**, all 0.556→0.617 | SHIP — top weight fix |
| 3 | **ent lane OFF** | +0.077 MRR / +0.085 any@10 — removes ~2/3 of bm25 MRR gap | SHIP ablation; rebuild later |
| 4 | **graph lane OFF** | +0.011 any@10, p50 −306ms, −3,213 stmts/q; 0 exclusive gold in every arm | ABLATE |
| 5 | **temporal bundle all_n6**: mentions-union + fallback-order + claims-anchor + events-loosen(0.5/0.5, subject_backfill) | +0.011 any / +0.050 mrr on cat-2 | SHIP (tune events gate — 5 churn) |
| 6 | **multi-hop ship-set**: decomp(≥2 REAL canons + coord marker) + WHOLE_SHARE=0.5 + FACET_LANES={lex,ent} | +0.005 any / +0.007 all | SHIP — gate on real canons, not junk |
| 7 | **rerank dead-7 drop** (cov_idf_ctx, ident_exact, life_state, corrob, perspective_fit, event_pred, lane_agree) | ~25ms stage earned | SHIP (keep cov_idf, phrase, speaker_match) |
| 8 | **speaker_match = 0** (or asymmetric +match) + u_speaker visibility fix | any positive sm costs cat-5 (0.404→0.286 @0.4) | SHIP sm=0 |
| 9 | **nom_cooc** (≥2 nominated terms AND-of-postings, cooc_min=2, fallback union) | +0.064 any / +0.053 MRR; union 674→110 mean | SHIP — kills nomination flood |
| 10 | **df_floor=4** | +0.024/+0.040 measured | SHIP |

### Composite (c4, paired same-store): comp−typed−obs
**any@10 0.6929 / session 0.8818 / MRR 0.4441 / ndcg 0.4729 / p50 185.3ms / stmts 445**
vs flat_bm25 0.587 / 0.884 / 0.398 / 0.418 / 4.3ms — **beats bm25 on every item-level
metric.** lane_miss 100→30, delivered 525→601.
- Interaction: any Σ +0.18-0.20 → +0.149 composite (~25-30% eaten by overlapping rescues); p50 amplified −484ms.
- b5 lexical ship-set alone (cooc + earners{lex,fuzzy,dense,time} + slim-rerank + df_floor=4):
  **any@10 0.686 (+0.149), MRR 0.476, p50 284ms** — independently beats bm25.
- **Final builder arm = c4 composite + b5's cooc/dff4/slim swapped in for windows-rescue
  (windows measured NEGATIVE solo).**

### Speed ship-set (read path)
| fix | measured | ship |
|---|---|---|
| graph lane off | p50 −306ms, −3,213 stmts/q | SHIP |
| dense 512-row repack | scan 75.6→7.2ms, bit-identical | SHIP |
| elig cache (sig=data_version+rowid-MAX+write_epoch) | 20→0.03ms/hit | SHIP size=128 |
| stem memo | 4,467→13.3 calls/q, −11.6ms | SHIP |
| post-pool trim max(4×limit,64) | −8.2ms, 0 recall change | SHIP |
| fts5vocab+unit_doclen | stmts 4,250→507 potential | SHIP |
| projected read p50 | 325@500 → **~150-200ms** — recheck post-composite | |

### Write-path (c2 — populated store, measured)
indexrel find_open instr() subject pre-filter **−437ms/doc** → composite4 **270.5ms/doc**
(762.6 baseline). <150 needs conversation-scoped job fusion (~190 floor w/ unarmed
fixes: aliases −35, rel_write −20, prescan −15). AMB ingestion measured ~1.38s/doc
end-to-end incl. operator admit pass — the write path is the second-biggest cost
after retrieval misses; spec target ≤150ms/doc.

## 3. Don't-ship (all measured rejects — do NOT let these back in)
- ent / graph / obs / typed lanes (net-harmful; −typed even stronger inside composite, ×1.7)
- rank windows (rerank_pool 150/200, pack 96/128 — convert lane_miss→rank_shift, never top-10)
- R_post two-phase scheduler (recall +0.03-0.06 but deadline bounding worsens 0→17-45 over-25ms)
- context.ctx_field (209-385ms, worse mrr) — propagate+inject wins
- multi-hop conjunctive joint (0 flips), canon_quota50 (mh −0.032), pack.group_maxpool
- temporal as_of global re-anchor (−0.029) — window-anchor only
- dense N_d>0 (0/137 unique rescues), sqlite-vec (<200K), f32→f16
- abstention floor variants (fire 0/990), wrong-speaker penalty, bulk tx batching
- lex_anchor, rescue_rows, single_match, claims-anchor standalone (inert by construction)

## 4. AMB-failure → ship-set mapping (what fixes each board weakness)
- **temporal 23.4%** — the biggest AMB hole: `occurred` fix + temporal bundle (arm 5)
  already measured +0.063 cat-2 inside composite (×5.7 vs solo). In-text date
  resolution anchored to occurred is the residual lever (~22/41 cat-2 misses).
- **single-hop 28.0%** — pure coverage: ctx-propagate+inject + cooc gating raise
  gold-in-pool; the reader is fine once context contains the fact.
- **multi-hop 38.5%** — already the best cat; arm 6 (decomp+slots) is measured but
  FLAGGED: gate DECOMP on ≥2 REAL entity canons — junk canons ('of','both') regress
  all@10 −0.032 once ENT lane removed.
- **open-domain 41.4%** — bulk of corpus; rides the same ship-set.
- **Premise/cat-5**: ent_idf=0 + sm=0 measured +0.09; premise_mm FPR 17.9% is a
  weak-pool symptom — pool strength fixes it, not trigger tuning.

## 5. Land order + verification protocol (for the build agent)
1. **Land in this order** (each earns its own Track-R rerun, Δ≥+0.010 per SPEC §ledger):
   earners-only lanes {lex,fuzzy,dense,time} → nom_cooc → slim-rerank (ent_idf=0, sm=0,
   dead-7 drop) → df_floor=4 → ctx propagate+inject → temporal bundle → decomp(≥2 real
   canons) → write-path indexrel + repack + caches.
2. **Track-R gate**: `VERBATIM_EVAL_LOCOMO=1 python3 -m eval.v7.track_r --dataset locomo
   --split dev --arms verbatim,flat_bm25,fts5 --workdir <dir>`; freeze `V7_PROBE_NOW_US`
   for paired determinism; expected post-ship-set ≈ composite 0.69+ any@10.
3. **AMB rerun** (the board that counts): same rig below; compare accuracy AND
   avg_context_tokens AND avg_retrieve_time_ms vs the 35.0%/494tok/79.6ms baseline.
   Per SPEC V8 the claim is ≥92.5% @ ≤4,500 ctx tokens, ≤100ms recall.
4. **Report honestly**: per-category deltas in `eval/v8/ledger.jsonl`; no superiority
   claims without a rerun (AGENTS.md rule).

## 6. AMB rig — replication recipe (harness lives in /tmp — EPHEMERAL, recreate it)
```bash
git clone https://github.com/vectorize-io/agent-memory-benchmark /tmp/agent-memory-benchmark
cd /tmp/agent-memory-benchmark && uv sync
```
Three file changes (all self-contained):
1. **`src/memory_bench/memory/verbatim.py`** — copy `eval/v7/overnight/verbatim_provider.py`
   from the pack; ensure the `AMB_VERBATIM_REPO` sys.path shim (default
   `/workspace/verbatim-new`); set `concurrency = int(os.environ.get("AMB_VERBATIM_CONCURRENCY", "4"))`.
2. **`src/memory_bench/memory/__init__.py`** — add `from .verbatim import VerbatimProvider`
   + `REGISTRY["verbatim"] = VerbatimProvider`.
3. **`src/memory_bench/llm/openai.py` + `runner.py`** — apply the robustness patch
   (`amb_patches.md` in this dir): strip `<|...|>` tokens, extract first JSON object
   via `raw_decode`, lazy response_format downgrade json_schema→json_object→prompt
   on 400 INVALID_REQUEST_BODY (kilo/ling models lack structured outputs), retry
   ValueError/429; runner: non-fatal query errors (record `query_error` result
   instead of crashing) + any-error retry 3×@5s.

Run (10 parallel unit shards; `--unit` per conversation; each writes its own
`outputs/` under its CWD):
```bash
GEMINI_API_KEY=dummy OPENAI_BASE_URL=<endpoint>/v1 OPENAI_API_KEY=<key> \
OMB_ANSWER_LLM=openai OMB_ANSWER_MODEL=<model> OMB_JUDGE_LLM=openai OMB_JUDGE_MODEL=<model> \
AMB_VERBATIM_CONCURRENCY=4 .venv/bin/amb run \
  --dataset locomo --split locomo10 --memory verbatim --unit conv-NN
# units: conv-26 30 41 42 43 44 47 48 49 50 → merge the 10 locomo10.json files
```
Working free models on the user's gateway (`https://OPERATOR_GATEWAY/v1`):
`kilo/inclusionai/ling-3.0-flash-fin:free` (fast, used for this baseline),
`mimo-v2.6-flash-free` (supports strict json_schema). Nvidia `kilo/*:free` models
returned upstream-503 during the run — retry when Nvidia recovers for a
stronger/judge-grade read. Leaderboard-comparable numbers still need the pinned
Gemini reader+judge once a key exists.

## 7. Honest bounds
- Composite is measured on the internal harness, not yet on AMB — treat as
  directional until the AMB rerun (expected large gain: coverage is the bottleneck
  and the ship-set adds +0.15 any@10 coverage internally).
- The AMB judge (ling-flash) is weaker than the pinned Gemini — absolute accuracy
  under-reports vs a stronger judge; relative deltas remain meaningful.
- cat-1 all@10 internal residual 0.087 — unit-granularity packaging, needs wider
  units/session-aggregation packs (not addressed by this ship-set).
- 100K wall (c3): eligibility path (~206ms make_eligible + ~275ms cache sig) breaks
  the read target before lanes at ~103K units — needs cheap cache signature +
  approx dense index; R_post scheduler stays on the don't-ship list.
- Read-p50 projected ~150-200ms post-ship-set vs the ≤100ms V8 target — remaining
  structural work flagged in c3 note.
```
```
Files in this pack: v8_overnight_pack.md · arm_register_v8.md · amb_patches.md ·
a1-a5/b1-b5/c1-c4 notes · verbatim_provider.py · forensic JSONs · b5_arm_results.tar.gz
```
