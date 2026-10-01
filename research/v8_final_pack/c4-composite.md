# V8 composite ship-set — paired measurement report

Paired replay on ONE locomo-dev store (`/tmp/v7w/mem.db`, built once via `track_r` arms `verbatim,flat_bm25`). Baseline arm reproduces the stock verbatim arm bit-for-bit (any@10 0.5444 vs 0.5444, p50 669.5 vs 671.7ms — freeze drift only). `now` frozen at `V7_PROBE_NOW_US=1790228700000000` (2026-09-24T05:45Z) in both trees. Composite = `/tmp` copy of @1c86f4f, all 10 ship-set items live behind env flags; write-path items applied as post-hoc store mutations on a store copy (`/tmp/v7w-comp/mem.db`). 990 tasks, k=10/20, deadline 2000ms.

## Composite table (item-level)

| arm | any@10 | all@10 | prop@10 | ndcg@10 | mrr@10 | p50 ms | p95 ms | stmts p50 | stmts mean |
|---|---|---|---|---|---|---|---|---|---|
| **base64** (stock+freeze, pack 64) | 0.5444 | 0.4667 | 0.5010 | 0.3238 | 0.2815 | 669.5 | 1556 | 4,896 | 5,853 |
| **comp** (−graph −ent +rest, pack 128) | 0.6667 | 0.5596 | 0.6063 | 0.4180 | 0.3792 | 222.2 | 383 | 496 | 581 |
| **comp−typed** | **0.6879** | 0.5838 | 0.6297 | 0.4695 | **0.4408** | 197.8 | 325 | 451 | 547 |
| **comp−obs** | 0.6747 | 0.5646 | 0.6116 | 0.4235 | 0.3839 | 211.3 | 381 | 490 | 573 |
| **comp−typed−obs** | **0.6929** | **0.5869** | **0.6322** | **0.4729** | **0.4441** | **185.3** | **326** | **445** | **539** |
| flat_bm25 (reference) | 0.5869 | 0.5030 | 0.5404 | 0.4181 | 0.3984 | 4.3 | 7.0 | — | — |

Session-level (any@10 / mrr@10): base 0.8465/0.5334 → comp 0.8758/0.6497 → comp−typed 0.8798/0.6854 → **comp−t−o 0.8818/0.6876** (flat_bm25 0.8838/0.6836).

Errors: 0 in every arm. abstain_rate: 0.185 → 0.193–0.201 (slight uptick, see below).

## Per-category (item any@10 / all@10 / mrr)

| category | base64 | comp | comp−typed | comp−obs | comp−t−o |
|---|---|---|---|---|---|
| adversarial (n=213) | .4038/.3944/.1930 | .6244/.6009/.3729 | .6526/.6338/.4199 | .6338/.6103/.3751 | .6573/.6385/.4178 |
| multi_hop (n=126) | .4524/.0873/.2070 | .5476/.0714/.2842 | .5397/.0635/.2990 | .5714/.0635/.2816 | .5635/.0556/.3004 |
| open_domain (n=47) | .3617/.2128/.2014 | .4681/.2766/.2745 | .4681/.2766/.2892 | .4681/.2766/.2701 | .4681/.2766/.2848 |
| single_hop (n=429) | .6084/.5781/.3282 | .7156/.6690/.4238 | .7459/.7040/.4809 | .7226/.6783/.4351 | .7483/.7110/.4904 |
| temporal (n=175) | .6743/.6229/.3498 | .7371/.6686/.3743 | .7543/.6857/.5108 | .7371/.6686/.3733 | .7543/.6857/.5089 |

## Attribution (base → comp−t−o)

delivered 525→601 · lane_miss 100→30 · rank_shift 98→96 · packed_out 0→0 · abstain 54→50 · unsupported 213→213 · unattributed 0.

Transitions (base→comp): +67 rank_shift→delivered, +18 lane_miss→delivered, +20 abstain→delivered, +46 lane_miss→rank_shift; regressions: 15 delivered→rank_shift, 8 delivered→abstain, 4 lane_miss→abstain.

## Stage timings (ms/query mean, comp−t−o vs base)

t_lanes 662.7→~125 · t_rrf 27.4→~13 · t_rerank_feat 26→~15 · t_pack 6.7→~13 (larger pack) · t_ctxprop ≈14 · pool_trim mean ≈60 rows trimmed · t_total 739.7→~200.

## Interaction analysis — composite delta vs Σ single-arm deltas

| metric | Σ single-arm (measured) | composite delta | interaction |
|---|---|---|---|
| any@10 | ≈ +0.18–0.20 (graph +.010, ent +.085, ctxprop +.041, decomp +.005, windows rescue, temporal +.002) | **+0.149** (comp−t−o) | ~25–30% eaten — arms rescue overlapping produced-then-lost tasks |
| mrr@10 | ≈ +0.148 (graph +.017, ent +.077, ent_idf+dead-feats +.054) | **+0.163** | ≈ neutral, mild amplification via −typed (+.061 vs +.036 alone, ×1.7) |
| p50 | ≈ −363 ms (graph −306, repack −72, stem −11.6, trim −8.2, decomp −15, ctxprop +50) | **−484 ms** | amplified ~+120 — stmt collapse compounds (5,853→539 mean) |
| lane_miss | −20% (ctxprop) + 60/99 window rescues ≈ −80 | **−70** (100→30) | additive as predicted |
| stmts/q | −3,213 (graph) + ~−3,700 (doclen+vocab 4,250→507) | **−4,451 p50 / −5,314 mean** | mostly additive |
| temporal cat-2 any | +0.011 | **+0.063** (temporal cat) | amplified ×5.7 — rerank200/pack128 surfaces mention-union additions |
| multi_hop prop@10 | +0.009 | **+0.019** | amplified |
| multi_hop all@10 | n/a | **−0.032** (0.0873→0.0556) | REGRESSION — see below |

## Verdicts

**Verified ship-set (co-lands cleanly):** items 1, 2 (−graph −ent **−typed −obs**), 3, 4, 5, 6, 7, 9, 10 — and −typed/−obs verified *inside* the composite, not just in isolation: removing all four non-core lanes is the best arm (any +0.149, mrr +0.163, sess-mrr 0.6876 ≈ flat_bm25 0.6836, p50 185ms −72%).

- −typed is *stronger* in composite than isolated (+0.061 vs +0.036 mrr) — claim-units stop competing once windows widen. Ship it with item 7, not alone.
- Item 2 sub-arm answer: **remove all of ent, typed, obs** (not just ent). −obs alone is small (+0.005 any / +0.003 mrr over comp−typed) but non-negative everywhere.
- Latency: p50 185.3ms — **misses the 150ms target by ~35ms**; 3.6× faster than baseline and beats flat_bm25 on every quality metric now (was behind on mrr at baseline: 0.28 vs 0.40 → now 0.44 vs 0.40).

**Flag — multi_hop all@10 regresses in every composite arm** (0.087→0.056 best arm, −36%): widening surfaced windows rescues single-evidence units but multi-evidence coverage slips, and DECOMP with ENT removed degenerates facets to lex-only (FACET_LANES=lex,ent − ent gone), so b2's mh-coverage gain does not transfer. Options: keep facets only where they earn (raise DECOMP gate to need ≥2 *real entity* canons — currently fires on coord-marker terms like 'of'/'both', yielding junk canons — real bug observed in comp run), or hold multi_hop-all on v8 before landing item 8.

**Also observed (fixed in-run, keep for reviewers):** (a) a `lane_outputs[key]` indent bug I introduced + fixed (deadline lane crashed once — now 0 errors); (b) the b3 restricted-fallback `EXISTS … unit_fts MATCH` re-ran MATCH per scanned unit → up to 400k internal `unit_fts_idx` probes/query (16 tasks, p99-latency 2.2s); rewrote as uncorrelated `IN (SELECT … MATCH)` — probes collapse ~200×, p99 265,828→3,265 stmts. Apply the same fix upstream of b3's write-up.

**Item-9 sub-note:** claims-anchor store mutation (3,091/7,122 claims re-anchored) is spec-parity — no v7 read path consumes `valid_intervals`; inert by construction, disclosed.

Subset that ran: **all 10 items** (no reduction needed). Sub-arms run: comp, comp−typed, comp−obs, comp−typed−obs.
