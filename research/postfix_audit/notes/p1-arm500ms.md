# Verbatim × LoCoMo dev — product-default 500ms vs eval-profile 2000ms deadline

Verification on `mosesman831/verbatim` @ HEAD `1c86f4f` (\"Beat-it retrieval plumbing fixes\"). Read-only clone, no commits; artifacts under `/tmp/v7w/` (attached).

## Setup & method
- Corpus: real LoCoMo dev split (VERBATIM_EVAL_LOCOMO=1, `research/v7_formula_search/locomo10.json`), registry dev partition → 5 conversations, digest `41df5b3a…`. 2,877 items, 990 tasks (777 answerable; 213 adversarial unsupported by design). Cats: single_hop 429, adversarial 213, temporal 175, multi_hop 126, open_domain 47.
- Ingest: one real `VerbatimArm` ingest (`Memory.add` + durable-queue drain + `wait_ready`) into `/tmp/v7w/mem.db` — 2,877/2,877 indexed, 0 add errors, 0 dedup collisions, settle ready, 15,753 jobs drained, ~17.3min.
- Queries: two scored passes through `eval.v7.track_r` machinery (k=10,20; surfaced pool probe `search(limit=64)` under each arm's own timeout). `verbatim@2000ms` = arm default (eval profile); `verbatim@500ms` = product `Memory.search` default — second arm instance sharing the same `Memory` + source-id maps, queries only, no second ingest.
- Caveat: 500ms pass ran second on a warm store/process — mildly favors its latency, not recall semantics.

## (a) verbatim @ 500ms — item granularity
| cat | n | any@10 | any@20 | prop@10 | ndcg@10 | mrr@10 | p50ms | p95ms |
|---|---|---|---|---|---|---|---|---|
| **all** | 990 | **0.547** | **0.658** | 0.503 | 0.356 | 0.325 | 325 | 613 |
| adversarial | 213 | 0.437 | 0.545 | 0.432 | 0.271 | — | 329 | 607 |
| multi_hop | 126 | 0.421 | 0.579 | 0.198 | 0.153 | — | 302 | 612 |
| open_domain | 47 | 0.234 | 0.426 | 0.158 | 0.099 | — | 318 | 703 |
| single_hop | 429 | 0.615 | 0.720 | 0.600 | 0.420 | — | 337 | 622 |
| temporal | 175 | 0.691 | 0.760 | 0.666 | 0.520 | — | 317 | 483 |

Session granularity: **any@10 0.842 / any@20 0.928** — per-cat any@10: adv 0.859, multi_hop 0.794, open 0.681, single 0.858, temporal 0.863. zero_rate 0; abstain 0.196; status 796 ready / 194 insufficient.

## (b) verbatim @ 2000ms — item granularity
| cat | n | any@10 | any@20 | prop@10 | ndcg@10 | mrr@10 | p50ms | p95ms |
|---|---|---|---|---|---|---|---|---|
| **all** | 990 | **0.543** | **0.649** | 0.497 | 0.326 | 0.285 | 635 | 1470 |
| adversarial | 213 | 0.423 | 0.521 | 0.420 | 0.249 | — | 645 | 1422 |
| multi_hop | 126 | 0.437 | 0.587 | 0.206 | 0.145 | — | 607 | 1556 |
| open_domain | 47 | 0.213 | 0.468 | 0.163 | 0.094 | — | 577 | 1570 |
| single_hop | 429 | 0.601 | 0.709 | 0.585 | 0.388 | — | 629 | 1490 |
| temporal | 175 | 0.714 | 0.754 | 0.675 | 0.462 | — | 658 | 898 |

Session granularity: **any@10 0.834 / any@20 0.926** — per-cat any@10: adv 0.859, multi_hop 0.754, open 0.638, single 0.844, temporal 0.891. abstain 0.186; status 806 ready / 184 insufficient.

## (c) Attribution split (item granularity, k=20)
| arm | delivered | lane_miss | rank_shift | packed_out | abstain | unsupported | unattributed |
|---|---|---|---|---|---|---|---|
| @2000ms | 532 | 97 | 90 | 1 | 57 | 213 | 0 |
| @500ms | 535 | 106 | 78 | 1 | 57 | 213 | 0 |
| **Δ** | **+3** | **+9** | **−12** | 0 | 0 | 0 | 0 |

- Answerable-task delivered rate: **68.9% @500ms vs 68.5% @2000ms**.
- Per-task flips are symmetric jitter, not a deadline trend (rank_shift→delivered 16 vs delivered→rank_shift 10; rank_shift→lane_miss 14 vs lane_miss→rank_shift 8; abstain\u2194delivered 6/5).
- The `surfaced` pool probe ran under each arm's own deadline → lane_miss\u2194rank_shift churn partly reflects the shallower 500ms probe pool, not delivered-recall movement.
- Per-cat attribution (delivered/lane_miss/rank_shift/abstain): single_hop 309/59/41/20 vs 304/58/46/21; temporal 133/16/13/13 vs 132/11/19/12; multi_hop 73/20/17/15 vs 74/19/18/15; open_domain 20/11/7/9 vs 22/9/7/9.
- insufficient verdicts rose 184→194 (single_hop +4, temporal +4, adversarial +2).

## (d) Latency
| arm | p50 | p95 | mean | min | max |
|---|---|---|---|---|---|
| @2000ms | 634.7 | 1470.4 | 735.7 | 452 | 1737 |
| @500ms | 324.8 | 613.3 | 371.6 | 222 | 758 |

~2.0× faster at p50, ~2.4× at p95; max ~758ms (deadline bounds lane stages; residual is fixed post-pipeline work).

## (e) Lane status @500ms post-eligibility-fix — YES, PARTIAL/DEADLINE persist
| lane | ok | partial | deadline | skipped | @2000ms |
|---|---|---|---|---|---|
| v7.lex | 983 | 7 | 0 | 0 | ok 990 |
| v7.dense | 977 | 13 | 0 | 0 | ok 990 |
| v7.ent | 943 | 44 | 3 | 0 | ok 976 / partial 14 |
| v7.fuzzy | 181 | 42 | 0 | 767 | identical (structural) |
| v7.time | 120 | 283 | 4 | 583 | ok 205 / partial 198 |
| v7.graph | 0 | 986 | 4 | 0 | partial 990 \u21d2 structural, not deadline-bound |
| v7.obs | 88 | 444 | 88 | 370 | ok 620 — heaviest deadline hit |
| v7.typed | 0 | 368 | 87 | 535 | ok 351 / partial 26 |

The deadline now lands where calibration intends: obs/typed lose tail contribution; lex/dense essentially complete. The lanes that get cut deliver no top-k gold — delivered recall rose, not fell.

## Caveats
- One shared ingest → identical store state; pass ordering mildly favors 500ms latency.
- Dev split only (990 tasks/5 conversations); ±0.02 per-category deltas are noise.
- Adversarial any@k measures premise-turn retrieval, not answer correctness.

## Verdicts
- **500ms-vs-2000ms recall delta ≈ zero, slightly positive for 500ms**: item any@10 +0.40pp, any@20 +0.81pp, session any@10 +0.81pp, ndcg@10 +3.0pp, mrr@10 +4.0pp. Delivered count +3/990.
- **The remaining recall gap is structural, not deadline-bound**: with 4× headroom the arm still misses 187 tasks (97 lane_miss + 90 rank_shift); 500ms only redistributes ~1% between those classes. Raising the deadline won't close it — lane coverage and ranking will.
- **What 500ms costs vs buys**: costs obs/typed/time/graph tail completions + ~10 insufficient verdicts — none delivered top-k gold; buys ~2× latency (p50 325 vs 635ms, p95 613 vs 1470ms).
- **Product default does not need to change on recall grounds.** 500ms is the right ship default: equal-or-better retrieval at half the latency. The 2000ms profile is justified only as a measurement tool for bounding lane potential, not as a product setting.
- Watch item (separate issue): `v7.graph` reports partial on ~all queries at BOTH deadlines — a lane-side structural cap worth its own investigation; obs/typed deadline misses at 500ms are honest degradation already absorbed by fusion.
