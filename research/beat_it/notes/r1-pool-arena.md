# Per-lane candidate caps vs shared candidate arena — research note

Question set: (a) external pool-sizing norms vs corpus scale and term df; (b) published guidance on
per-lane truncation below fusion vs a jointly-scored arena; (c) lane_cap/rerank_pool that preserves
any@64 vs uncapped BM25 given N=7020 units and entity-term dfs ~700; (d) Hindsight TEMPR + Mem0
pool sizes from code.

Environment note: this child VM has no `/tmp/v7-trackr-*/mem.db`; I rebuilt the store locally by
running `eval.v7.arms.VerbatimArm` over the LoCoMo dev split (`VERBATIM_EVAL_LOCOMO=1`,
`load_corpus('locomo', split='dev')` → 2,877 items, 990 tasks, 777 answerable — exactly the numbers
in the assignment). Result: `units=7020` (2,877 `turn` + 2,877 `session` + 1,266 `sentence_window`),
`unit_fts=7020`, `sources=2877`, `claims=7122`, `claim_evidence=7122`, `events=25865`;
`facts_fts=0`, `t2_facts=0`, `graph_edges=0`, `entities=0`, `observations=0` — the four dead lanes
reproduce verbatim [measured]. Divergence vs the parent's store: my clone (GitHub HEAD 892f2b8)
populates `entity_canon` (1,513) / `entity_postings` (8,780) / `entity_aliases_v7` (1,774) and writes
claims, so the entity lane has data on this build even though `entities`/`entities_v2` are absent.
Ingest cost: 920 s for 2,877 items (~320 ms/item, infer on) [measured].

## 0. The actual architecture (code, verbatim-new @ 892f2b8)

- `POOLS` table — `verbatim/core/types_v7.py:414-418`: `LOW(50,40,0,0,1,F)`, `MID(200,100,16,1,2,F)`,
  `HIGH(800,300,32,2,3,T)` for `(lane_cap, rerank_pool, ce_pool, neighbor_window, max_facets, prf)`.
- S2 slice allocation — `verbatim/retrieval/v7/deadline.py:130-153`: deadline split is
  cost-proportional over `LANE_COSTS_V1` (`policy.py:153-164`: lex 6, fuzzy 4, dense 6, ent 2,
  time 3, graph 6, typed 4, obs 2 ms) — a p95 *target* shared as a proportional weight, with a
  `SLICE_FLOOR_MS=0.5` floor. Same `cap=pool.lane_cap` on every slice (`deadline.py:101`).
- Per-lane admission cap — every lane ends `ordered[:cap]` or `order[:cap]` AFTER scoring the full
  posting union: lexical `lexical.py:1513-1522`, source `source.py:1585` (`final = ordered[:cap]`),
  scope `scope.py:466`, fuzzy `fuzzy.py:475`, entity `entity.py:627-629`. So **the lane pays to score
  every posting row regardless of cap**; the cap only bounds emitted `CandidateV7` objects.
- Lexical nomination — `lexical.py:991-1049` + docstring: a document enters the pool only via a
  *kept* query term (`nominate_terms_max=32`, identifiers first then content terms by ascending
  eligible df — rare terms win). `nominate_df_theta` (df > θ·N_E exclusion) exists but is disarmed by
  default — and arming it would be harmful here (see §4). Median query length on the dev split is 10
  terms, p95=16, max=25 [measured] — the 32-term budget essentially never binds on LoCoMo.
- Fusion — `fusion.py:rrf_fuse`: weighted RRF `Σ w_lane/(60 + lane_rank)`, `RRF_K=60`
  (`fusion.py:95`), ungated lanes all contribute; `limit=` exists but the pipeline does not pass it —
  the fused list is the full union (`pipeline.py:977-993`).
- S4 feature rerank — `rerank_features.py:score_candidates` scores the ENTIRE fused pool with a
  linear feature model; `pipeline.py:1045` then cuts `[:pool.rerank_pool]` post-scoring.
- S5 CE — `pipeline.py:1078`: reranks `scored[:pool.ce_pool]` only (16 MID / 32 HIGH).

So the effective funnel at MID: 8 lanes × ≤200 admitted → union (≤1600) → all feature-scored →
top-100 → CE top-16 → pack to limit. Admission caps are pre-fusion; rerank_pool is post-fusion-score;
ce_pool is the CE batch.

## (a) How external systems size pools vs corpus scale / term df

| System | Pool architecture | Numbers | Source |
|---|---|---|---|
| TREC pooling | Depth@k judged pool per run; k chosen by *judgment budget*, not df | k=100 standard; TREC-6: k=100 × 30 runs ≈ 60k judgments; Move-to-Front varies depth by run strength | Cormack/Palmer/Buckley SIGIR'98 (doi:10.1145/290941.291009); Losada et al. ICTIR'17 "Fixed budget pooling" (doi:10.1145/3019612.3019692) [paper] |
| Lucene | ONE index, ONE scorer; top-k via Block-Max WAND — provably identical to exhaustive disjunctive scoring | "safe" early termination returns *the same result*; truncation exists only at the final k | Ding & Suel SIGIR'11 (research.engineering.nyu.edu/~suel/papers/bmw.pdf); Lucene BMW story PMC7148045 [paper] |
| Vespa | Phased ranking: first-phase evaluates **all** retrieved hits; second-phase reranks top-N per node | `rerank-count`/`total-rerank-count` default **100 per node** | docs.vespa.ai/en/ranking/phased-ranking.html; schemas.html.md [vendor] |
| MS MARCO convention | BM25 top-1000 → neural reranker | top-1000 per query (Nogueira & Cho 2019, arXiv:1901.04085) | [paper] |
| Mem0 OSS | ONE dense arena is the only recall source; BM25/entity re-score it, cannot inject | `internal_limit = max(top_k*4, 60)` ≈ 80 at top_k=20; entity sidecar breadth top_k=500/entity × ≤8 entities; reranker sees only final ≤20 | `main.py:1641,1502,1747,1775,1778`, `scoring.py:111-119` (per research note d2-mem0-code.md) [code] |
| Hindsight TEMPR | Per-arm SQL LIMIT (real per-lane caps), then unweighted RRF union | arm LIMITs {low,mid,high}={100,300,1000}; adaptive `clamp(max_tokens×{0.025,0.075,0.25},20,2000)`; per-source pre-fusion cap exists but **off by default** (`DEFAULT_RECALL_MAX_CANDIDATES_PER_SOURCE=0`); CE pool=300 (batch 32); temporal entry pool=60/fact-type, 10 entry points, 8 buckets; graph lateral cap=200/entity; entity-resolution pool=200; BM25 query trimmed to **16 lowest-df terms** | `memory_engine.py:1453-1483`, `config.py:1291-1319,1818,1908-1917`, `fusion.py:8`, `retrieval.py:404,414-460`, `bm25_term_selection.py:82-118` (per notes a1/d1) [code] |

Takeaways: nobody sizes a pool from corpus N directly; the pattern is a *fixed budget per stage*
(scoring) and a *smaller fixed depth before each expensive stage* (rerank). Vespa and Lucene score a
single shared arena; Hindsight is the counter-example WITH per-arm caps ({100,300,1000}) but at
Postgres scale with a disabled-by-default per-source gate — i.e. even they don't truncate below the
fusion that needs it.

## (b) Published guidance: per-lane truncation below fusion vs shared arena

No paper says "strictly worse" — the honest guidance is a conditional-safety rule:

1. **Truncation is safe only if the truncated list still contains the fused top-k's support.**
   Metasearch/fusion literature (Aslam & Montague SIGIR'01 models; Cormack/Clarke/Büttcher SIGIR'09
   RRF, doi:10.1145/1571941.1572114) fuses already-truncated run lists (top-100/1000) — accepted
   because each run is a full-corpus ranker whose relevant mass is in its head. Under RRF k=60 a
   candidate at lane-rank r contributes `1/(60+r)`: rank-1 = 0.0164, rank-200 = 0.0038, rank-1000 =
   0.0009. To enter a fused top-64 that is typically won at ~rank-1-single-lane score, a lone
   deep-admitted candidate needs rank ≲ ~124 (1/(60+124)); deeper admission only counts via
   corroboration: m lanes at ~rank-200 give m·0.0038 — three lanes ≈ rank-1-alone [calculated].
   **Corollary: enlarging a single lane's cap past ~120-300 has sharply diminishing fused value
   unless other lanes vote too.** Verbatim has 4/8 lanes emitting zero — corroboration starves.
2. **Lucene's existence proof for "safe" truncation is single-scorer only.** BMW returns *the same*
   top-k as exhaustive; no equivalent bound exists for RRF-fused lists — a unit ranked >cap in every
   lane contributes 0 (hard loss), versus bounded loss in same-scorer truncation [paper reasoning].
3. **Pool-bias literature** (Buckley et al. "Bias and the limits of pooling" SIGIR'07; Zobel SIGIR'98):
   relevant docs DO sit beyond pool depth; variable-depth/move-to-front pooling exists precisely
   because fixed shallow caps systematically miss a relevance class — the eval-side analogue of
   Verbatim's lane_miss [paper].
4. **Vendor evidence splits both ways**: Mem0 solves it by construction (one dense arena — nothing to
   truncate below fusion); Hindsight keeps per-arm caps but leaves the per-source pre-fusion cap OFF
   and relies on `recall_budget` 300-1000 sized to their corpus. Both are consistent with "truncate at
   a stage boundary, not below the join" [code].
5. Measured mechanism where per-lane caps are *worse than equivalent arena depth*: when a lane's
   admit-order ≠ evidence-order for the tail. Entity/scope lanes admit by lane-local score over a
   df-700+ posting set — cap 200 keeps 28-49% of a common entity's postings with no guarantee the
   evidence sits in that head (measured §c). A shared arena scores the SAME 7020 units once with one
   ordering — truncation there at least loses tail items by a single comparable score [calculated].

So: not "strictly worse" in the literature, but the published safety conditions (same-scorer
equivalence for BMW; corroboration mass for RRF; pool-depth-bias mitigations) are exactly the
conditions Verbatim's current setup violates — per-lane cap(200) × dup-units × dead lanes.

## (c) Measured arithmetic — N=7020, what preserves any@64

Scorers rebuilt in-Python [measured]: `flat_bm25` = the arm's Okapi over 2,877 item docs
(k1=1.2,b=0.75, `\w+` tokens); `unit_bm25` = same over all 7,020 `unit_fts.text` rows — the closest
stand-in for "one shared arena over units".

Headline replication: flat_bm25 any@10/**any@64** = **0.589/0.757** (assignment cites 0.587 — match).
unit-space BM25: any@10/64 = **0.492/0.685** — the session+turn duplication + sentence_window
fragments cost ~10 pts at @10 vs the flat item space [measured].

**Admission curve** — P(some gold unit has unit-BM25 rank ≤ C), 777 answerable tasks [measured]:

| cap C | admission | | cap C | admission |
|---|---|---|---|---|
| 64 | 0.685 | | 700 | 0.878 |
| 100 | 0.727 | | 1000 | 0.924 |
| 128 | 0.746 | | 1500 | 0.960 |
| 200 | 0.792 | | 2000 | 0.983 |
| 256 | 0.807 | | 3000 | 0.997 |
| 300 | 0.820 | | 4000 | 0.999 |
| 500 | 0.847 | | ∞(7020) | 0.999 |

Key joint facts [measured]:
- **All 588 flat@64 hits are admitted at unit-cap=200 — 100%.** (532/588 at cap 64, 565 at 100.)
  ⇒ The verbatim-vs-flat any@64 gap is NOT a lexical-admission problem; it lives downstream:
  deadline-starved slices (parent: ~85 ms lex slice vs ~200 ms needed; 8 s budget recovers 72/120),
  eligibility/verdict drops, dup-unit crowding, and rerank/verdict order — not the 200 cap itself.
- Of the 189 flat@64 misses, 188 still score in the unit arena — **median best-gold rank 695,
  p75 = 1148** — the deep tail that cap 200 cannot admit and that a cap-1000 lane covers at ~92%.
- Per-category admission at cap 200 → 1000: open_domain **0.553 → 0.851** (largest headroom),
  multi_hop 0.722 → 0.905, single_hop 0.830 → 0.928, temporal 0.811 → 0.949.
- 1 task's gold has zero lexical score entirely — unreachable by any cap (true paraphrase gap).

Term-df picture on the store [measured]: `unit_fts.entities`-field head dfs: john 1078, nate 1072,
james 1030, joanna 991, evan 973, sam 951, jolene 928, deborah 863, jon 699, gina 667 — matches the
"~700" figure. `entity_canon.df_units` (the lane's own weight field): max 407, p99 231, 20 canons
>200. 30 entity terms exceed df 200; 11 exceed 700. So an entity/scope lane admitting a global
top-200 keeps **at most ~19-49% of a head entity's posting set** — and entity-lane scoring is
idf-weighted membership, not evidence relevance, so the truncated tail is near-random w.r.t. gold.

**Derived cap rules** [calculated from the above]:
1. *Flat-parity preservation* (any@64 = 0.757): needs every flat@64 gold admitted AND deliverable.
   Admission needs cap ≥ ~200 (measured 100% at 200; recommend 256 for BM25F-vs-BM25 rank wobble and
   dup crowding). Delivery needs `rerank_pool ≥ 64` strictly and ≥ ~4×64 for corroboration slack.
2. *Deep-tail capture* (the extra recall flat itself misses): admission cap by coverage target —
   0.85 → 500, 0.92 → 1000, 0.96 → 1500, 0.98 → 2000, 0.99 → ~3000 = effective uncap at this N.
   But fused-value bound: single-lane rank >~124 needs corroboration to reach a top-64 slot; with 4
   lanes dead, pushing cap past ~300 buys admission without delivery unless the feature reranker
   promotes on non-RRF signals (phrase/entity match exist as features, so it partially can).
3. *Entity/scope lanes*: replace the global cap with a **per-canon quota**
   `q = max(64, ⌈C_ent / n_query_canons⌉)` so a df≈700-1078 canon cannot starve a df≈20 co-canon
   (Hindsight's per-entity lateral cap 200 and Mem0's 500/entity are the same pattern) [code+inferred].
4. *Dup-unit tax*: session+turn units carry identical `unit_fts.text` — identical BM25 → adjacent
   ranks → each evidence consumes **2 admission slots and double-votes RRF (2/(60+r))**. Dedup to one
   canonical unit per source at admission (or collapse at fusion by source_id) doubles effective lane
   depth for free [measured — verified identical text rows at rowid 1/2 on this store].
5. *rerank_pool*: fused pool ≈ Σ lane caps (up to 8×cap); feature rerank scores all of it
   (mem0-measured ~0.033 ms/80 cands → ~8000 cands ≈ +3 ms — negligible); the cut is post-score, so
   deep candidates CAN surface on feature strength. rerank_pool bounds the deliverable: keep
   `≥ max(4·k_out, 256)` for k_out=64. Vendor depth ratios for rerank vs output: Vespa 100/node for
   ~10; Hindsight CE 300 for ~30-50; Nogueira-Cho 1000 for 10 — i.e. 4-10× minimum [vendor/paper].
6. *ce_pool* is purely CPU-bound: MiniLM-L6 ≈ 8-15 ms/pair on 4-core (d1 note) → MID 16 ≈ 130-240 ms
   at the ≤150 ms bound only with TinyBERT (~2-4 ms/pair → 32 ≈ 60-130 ms) [vendor/calculated].
7. **Cost asymmetry**: raising lane_cap is nearly free — lanes already score every posting row before
   `[:cap]`; the real cost was never the cap, it's the cost-proportional *deadline slice* (lex share ≈
   6/35 of remaining ms → ~85 ms at a 500 ms budget vs ~200 ms of posting work). Enlarging caps without
   lifting the deadline changes nothing — consistent with the parent's own 8 s-budget recovery.
   nomi­nate_df_theta: keep DISARMED — θ·N_E = 0.1×7020 = 702 would bar exactly the df~700 speaker/entity
   terms from nominating (jon 699 stays, john 1078 barred) — the opposite of what entity recall needs.

## (d) Hindsight TEMPR and Mem0 — actual pool sizes [code, via research notes a1/d1/d2]

Hindsight (vectorize-io/hindsight):
- Per-arm recall LIMIT (semantic, BM25, temporal, graph arms): fixed {100,300,1000} for
  budget {low,mid,high}; adaptive = clamp(max_tokens·{2.5%,7.5%,25%}, 20, 2000). Per-arm, per
  fact-type — so the effective pre-fusion pool is ~4× the budget.
- `cap_per_source` pre-fusion cap: exists, **default 0 (off)**.
- Reranker pool cap `DEFAULT_RERANKER_MAX_CANDIDATES = 300` — pool>300 is re-sorted by boosted RRF and
  truncated; per-budget overrides default to 0 → 300. The "32" is `RERANKER_LOCAL_BATCH_SIZE` (batch,
  not pool). Production sends 20-50 to the CE typically (ACL demo).
- Temporal arm: entry pool `_TEMPORAL_POOL_SIZE=60`/fact-type (window-overlap WHERE + sim ≥0.1),
  `_TEMPORAL_ENTRY_POINTS=10` picked over 8 time-buckets; bounded BFS ≤5 iters × 20.
- Graph arm: top-20 semantic seeds ≥0.3 per fact-type, strictly 1 hop, per-entity lateral cap 200.
- Entity resolution (write path): pg_trgm candidates `entity_resolution_max_candidates=200`.
- BM25 arm: query capped to **16 lowest-DF terms** via `pg_stats.most_common_elems` — df-selective
  term nomination (Verbatim's nomination keeps ascending-df order too — the same instinct).
- RRF k=60 unweighted; rank-space boosts only (score-space weighting measured to collapse
  recall@20 0.97→0.40 — `recall_boost.py:27-47`).

Mem0 OSS (mem0ai/mem0):
- `search.top_k` default 20; internal candidate pool `max(top_k*4, 60)` = 80 — **one dense arena**;
  sparse BM25 and entity similarity re-score the pool but cannot inject candidates.
- Entity expansion: ≤8 query entities, `top_k=500` each (breadth 4000, deduped), sim-gate 0.5.
- Reranker (off by default) reorders **only the final top_k ≤20**, not the internal pool.
- Platform memory-decay (docs only): widens pool to `top_k*3` floor 50 before rescoring.

## Recommended pool architecture

1. **Do not flatten the lanes into one scorer** — flat parity is already inside the lexical lane at
   cap 200. The fix sequence is: deadline budget → dedup units → per-canon entity quota → THEN cap
   sizing. A "shared arena" is what unit_fts already is; the failure is that admission, not scoring,
   is truncated mid-pool — while the pool is narrowed again by dup units and by deadline-PARTIAL lanes.
2. **Caps as stage boundaries, matching the literature**: (i) admission cap sized to the df tail you
   intend to cover, not a magic 200; (ii) a second, larger bound before expensive scoring
   (rerank_pool ≥ 4× deliverable); (iii) a small CE batch sized by the 150 ms budget, not by recall.
3. **Concretely** (proposed POOLS; "(lane_cap, rerank_pool, ce_pool)" — keep other fields):
   - LOW  `(128, 64, 0)`   — flat-head parity tier; admits 74.6% of gold pool at this N; ~1-3 ms.
   - MID  `(1000, 256, 16)` — the measured 92.4%-admission point (1500→96%); open_domain 85%; fused ≤8k
     cands ≈ +3 ms feature scoring; CE 16 ≈ ≤150 ms only with TinyBERT-class, else ce_pool=0.
   - HIGH `(3000, 512, 32)` — ~uncapped at N=7020 (99.7% admission); CE 32 = 60-130 ms TinyBERT.
   Scale rule for larger N: `lane_cap ≈ min(max(4·k_out, 256), ⌈p99(df of lane terms)⌉·n_terms)`
   — i.e. cap follows the df tail, not corpus N; entity lane uses per-canon quota `max(64, C/n_canons)`.
4. **Pair every cap raise with**: (a) request deadline scaled to corpus (or resumable lane scans —
   cap enlargement is wasted while slices starve); (b) session+turn dedup at admission/fusion (frees
   ~40% of every lane's slots and removes the double RRF vote); (c) filling the 4 dead lanes — cap
   arithmetic is moot for lanes that emit zero; corroboration is what makes deep admission pay off.
5. **Alternative worth prototyping** (scratch only): nominate-set arena — score the union of ALL lane
   posting sets under each lane's own score, admit by a single fused-ranked cut. At this N it is
   equivalent to cap≈3000 lanes; it becomes the better shape only when the dup/eligibility fixes land
   and the question is ordering, not admission.

## Verdicts

1. **Raise MID lane_cap 200 → ~1000** (per scoring-side admission curve: 79.2%→92.4% gold admission;
   open_domain 55%→85%). Cost ≈ +3 ms fused-pool Python work; lanes already score every posting.
   But expected any@10 lift is small-to-moderate on its own — deep-admitted candidates still need
   corroboration/feature promotion to reach top-64 — do it *with* the deadline fix. [measured]
2. **rerank_pool: MID 100 → 256** (≥4×64 + corroboration slack; vendor depth ratios 4-100×).
   Cost ~0. [calculated + vendor pattern]
3. **Deadline is the dominant knob, not the cap** — parent's own datum: 8 s budget surfaces 72/120
   gold into fused top-64; my arithmetic says cap=200 already admits 100% of flat@64 gold. Scale the
   request budget to corpus size (500 ms is tuned for ~590 turns, this corpus is 2877 items → 7020
   units) and/or give lexical/source a slice floor ≥ posting-scan time. [measured + parent datum]
4. **Dedup session+turn units before admission** (identical `unit_fts.text` → 2 slots + double RRF
   vote per evidence). Effectively doubles every lane's cap for free and removes a systematic fusion
   bias. [measured]
5. **Entity lane: per-canon quota** `max(64, ⌈cap/n_canons⌉)` instead of a global cap — a df~700-1078
   canon floods a global 200 to 19-49% coverage of its own postings. [measured df + code]
6. **ce_pool**: keep 0/16/32 (LOW/MID/HIGH) — Hindsight's 300 needs remote/TEI; at ≤150 ms only a
   TinyBERT-class CE fits (32 pairs ≈ 60-130 ms). The existing table values are already right.
   [vendor + calculated]
7. **Keep nominate_df_theta disarmed** — any θ≲0.1 bars the exact head-entity terms (df 600-1100)
   that entity-bearing questions need to nominate. [measured]
8. **Do not adopt a single shared scored arena yet** — measured unit-arena BM25 underperforms flat
   item-BM25 (0.492 vs 0.589 any@10) *because of* dup/sentence_window crowding; arena-without-dedup
   would move the bias, not fix it. Revisit after dedup. [measured]
9. **Priority order** (expected lift ÷ cost): deadline scaling ≫ unit dedup ≈ lane_cap 200→1000 +
   entity quota ≈ rerank_pool→256 > ce tuning. Filling obs/typed/facts/graph lanes unlocks the
   corroboration that makes deep caps worth anything — the measured median deep-gold rank is 695,
   and a lone rank-695 candidate contributes RRF 0.0013 ≈ nothing without a second lane voting.
   [measured + calculated]
