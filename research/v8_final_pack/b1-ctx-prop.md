# S2c context-propagation sweep — SPEC V8-06 §21.1 on the real dev store

Prototype: `verbatim/retrieval/v7/ctxprop.py` in the /tmp copy, env-gated
hook inserted between the S2 lane loop and the S3 eligibility re-check in
`pipeline.py` (propagation BEFORE lane ranks; RRF scores never rewritten —
`rrf_fuse` consumes `cand.rank`, so ranks are recomputed after the score
rewrite). Store: `/tmp/v7w/mem.db` snapshot — 2877 sources, 2877 turn
units, 1266 sentence_windows, generation 1, LOCOMO dev split (2877 items,
990 task views, 777 answerable). All 25 arms ran the identical
`run_search`+`attribute` loop on per-worker sqlite-backup copies of the
same store; the baseline row below is this driver's own no-ctx pass
(pairwise-comparable, differs ≤0.004 from the earlier standalone track_r
run from hash-order tie-breaks — disclosed).

## Paired baseline

- item any@10: all-990 0.535, answerable-777 0.583 (any@20 0.681, all@10 0.476, prop@10 0.522, ndcg@10 0.352, mrr@10 0.316)
- session any@10 0.848 (mrr@10 0.538)
- attribution {'delivered': 529, 'lane_miss': 99, 'rank_shift': 91, 'packed_out': 0, 'abstain': 58, 'unsupported': 213, 'unattributed': 0}
- latency p50 660ms / p95 1572ms; pack tokens mean 1206

**Disclosure:** `units.seq` is degenerate on this build — every per-turn `Memory.add` is one source, so seq=0 for all turns. The spec's 'seq within ±W' is realized as ±W positions over session members ordered by (recorded_at_us, unit_id), which equals dialogue order on this corpus (verified distinct+monotone). Production constant-transfer caveat below.

## Arm table (deltas vs paired baseline; any@10 = answerable item any@10)

| arm | ans any@10 | Δ | Δ all@10 | Δ sess | Δ mrr | Δ adv | lm | rs | delivered | t_ctx p50 | sql |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.583 | — | — | — | — | — | 99 | 91 | 529 | — | — |
| pi-w0.9-W2-M25 | 0.624 | +0.041 | +0.039 | +0.006 | +0.014 | +0.099 | 79 (-20) | 76 (-15) | 573 (+44) | 15.181 | 2 |
| pi-w0.9-W2-M50 | 0.620 | +0.037 | +0.040 | +0.001 | +0.017 | +0.089 | 77 (-22) | 77 (-14) | 575 (+46) | 15.736 | 2 |
| pi-w0.7-W2-M50 | 0.620 | +0.037 | +0.037 | +0.000 | +0.015 | +0.099 | 76 (-23) | 81 (-10) | 571 (+42) | 16.567 | 2 |
| pi-w0.7-W2-M25 | 0.619 | +0.036 | +0.030 | +0.002 | +0.012 | +0.085 | 78 (-21) | 80 (-11) | 569 (+40) | 15.078 | 2 |
| pi-w0.9-W1-M25 | 0.615 | +0.032 | +0.032 | +0.009 | +0.009 | +0.160 | 82 (-17) | 88 (-3) | 559 (+30) | 14.211 | 2 |
| pi-w0.5-W1-M50 | 0.613 | +0.030 | +0.028 | +0.009 | -0.003 | +0.103 | 84 (-15) | 86 (-5) | 556 (+27) | 14.582 | 2 |
| pi-w0.5-W1-M25 | 0.611 | +0.028 | +0.027 | +0.009 | +0.000 | +0.103 | 85 (-14) | 85 (-6) | 557 (+28) | 13.324 | 2 |
| pi-w0.9-W1-M50 | 0.611 | +0.028 | +0.030 | +0.013 | +0.006 | +0.169 | 79 (-20) | 87 (-4) | 563 (+34) | 15.098 | 2 |
| pi-w0.7-W1-M25 | 0.611 | +0.028 | +0.024 | +0.013 | +0.003 | +0.122 | 85 (-14) | 85 (-6) | 554 (+25) | 14.003 | 2 |
| i-w0.7-W2-M25 | 0.609 | +0.026 | +0.018 | +0.006 | +0.004 | +0.080 | 80 (-19) | 94 (+3) | 551 (+22) | 13.631 | 2 |
| pi-w0.7-W1-M50 | 0.607 | +0.024 | +0.024 | +0.010 | +0.002 | +0.136 | 83 (-16) | 88 (-3) | 556 (+27) | 14.978 | 2 |
| i-w0.7-W2-M50 | 0.607 | +0.024 | +0.017 | +0.009 | +0.003 | +0.089 | 81 (-18) | 97 (+6) | 552 (+23) | 15.101 | 2 |
| p-w0.9-W2 | 0.606 | +0.023 | +0.019 | +0.006 | +0.009 | +0.028 | 84 (-15) | 90 (-1) | 547 (+18) | 13.072 | 2 |
| ctxfield-W2 | 0.606 | +0.023 | +0.018 | +0.024 | -0.006 | +0.038 | 73 (-26) | 86 (-5) | 566 (+37) | 385.456 | 7 |
| pi-w0.5-W2-M50 | 0.606 | +0.023 | +0.019 | -0.003 | +0.010 | +0.066 | 80 (-19) | 83 (-8) | 562 (+33) | 16.457 | 2 |
| p-w0.7-W2 | 0.605 | +0.022 | +0.019 | +0.003 | +0.008 | +0.023 | 86 (-13) | 87 (-4) | 548 (+19) | 13.155 | 2 |
| pi-w0.5-W2-M25 | 0.602 | +0.019 | +0.024 | -0.003 | +0.006 | +0.070 | 81 (-18) | 81 (-10) | 562 (+33) | 14.697 | 2 |
| i-w0.7-W1-M50 | 0.601 | +0.018 | +0.017 | +0.008 | +0.001 | +0.075 | 89 (-10) | 89 (-2) | 550 (+21) | 14.06 | 2 |
| ctxfield-W1 | 0.600 | +0.017 | +0.012 | +0.018 | -0.013 | +0.056 | 83 (-16) | 87 (-4) | 559 (+30) | 209.448 | 7 |
| i-w0.7-W1-M25 | 0.597 | +0.014 | +0.013 | +0.008 | +0.003 | +0.080 | 88 (-11) | 90 (-1) | 548 (+19) | 13.4 | 2 |
| p-w0.5-W1 | 0.596 | +0.013 | +0.018 | +0.009 | +0.000 | +0.052 | 89 (-10) | 89 (-2) | 548 (+19) | 12.82 | 2 |
| p-w0.9-W1 | 0.593 | +0.010 | +0.014 | +0.006 | +0.000 | +0.056 | 92 (-7) | 87 (-4) | 548 (+19) | 12.818 | 2 |
| p-w0.5-W2 | 0.592 | +0.009 | +0.017 | +0.001 | +0.007 | +0.023 | 85 (-14) | 92 (+1) | 547 (+18) | 13.099 | 2 |
| p-w0.7-W1 | 0.591 | +0.008 | +0.013 | +0.010 | -0.001 | +0.070 | 91 (-8) | 84 (-7) | 550 (+21) | 12.817 | 2 |

`p-*` = propagate-only, `i-*` = inject-only, `pi-*` = propagate+inject,
`ctxfield-*` = V8-06.03 expanded-BM25F neighbor-text scoring (w_ctx=0.5).

## Task 2 — mode decomposition (at w=0.7, W=2, M_ctx=25)

- propagate-only: Δany +0.022, lm 86, rs 87 — boosts already-nominated units only
- inject-only: Δany +0.026, lm 80, rs 94 — surfaces never-nominated neighbors
- propagate+inject: Δany +0.036, lm 78, rs 80 — sub-additive but strictly better than either alone

Injection is the primary lever (fixes lane_miss AND rank_shift: never-
nominated golds enter the pool at w·max-src); propagation alone only
rescues rank_shift. Combined wins because propagation also re-ranks the
injected unit's lane neighbors.

## Task 4 — ctxfield arm (V8-06.03 approximation)

- ctxfield-W1: Δany +0.017, Δsess +0.018, Δmrr -0.013, lm 83, t_ctx p50 209.448ms, sql 7
- ctxfield-W2: Δany +0.023, Δsess +0.024, Δmrr -0.006, lm 73, t_ctx p50 385.456ms, sql 7

ctxfield gets the best lane_miss (73 @W2) and session any@10 (+0.024) —
neighbor text does surface more pool recall — but loses on the primary
metric (+0.023 vs +0.041) and mrr (−0.006): content-word overlap with a
bare reply is a weaker signal than 'its turn-neighbor scored'. And it
costs ~200-385ms/query (per-candidate text fetch + token probes) vs ~15ms
for s'+w·max. **Verdict: s'+w·max dominates on quality and 14-25× on
latency; do not build the index-time ctx field.**

## Task 3 — contamination audit

Mechanism: both batched SELECTs filter `scope_id = ctx.scope_id` +
`generation <= ctx.generation` + `kind='turn'` (newest in-fence
generation per unit_id wins), so cross-scope/cross-generation leakage and
sentence_window participation are excluded by construction. Runtime audit
(VERBATIM_CTX_AUDIT dumps every inject/boost edge) verified on 100 tasks:

- W=1 (40 tasks): 38,483 edges — 0 violations
- W=2 (60 tasks): 110,587 edges — 0 violations
- every edge: same session, |position delta| ≤ W, both endpoints pass
  `ctx.eligible` (pinned EligibilityV7 unit_ids); eligibility is
  fail-closed (None elig → empty set → no ctx contribution).

## Per-category deltas — pi-w0.9-W2-M25

| category | n | baseline any@10 | arm any@10 | Δ |
|---|---|---|---|---|
| adversarial | 213 | 0.362 | 0.460 | +0.099 |
| multi_hop | 126 | 0.468 | 0.468 | +0.000 |
| open_domain | 47 | 0.298 | 0.362 | +0.064 |
| single_hop | 429 | 0.599 | 0.655 | +0.056 |
| temporal | 175 | 0.703 | 0.731 | +0.029 |
## Task 5 — what ctx does NOT fix (pi-w0.9-W2-M25 vs baseline)

Fixed 60 baseline misses (lane_miss/rank_shift→delivered); regressed 16 (delivered→miss — injected crowding); abstain 58→48; net delivered 529→573 (+44).

Fixed breakdown: single_hop 36 (6 lm + 30 rs), temporal 5 rs,
open_domain 2 rs, multi_hop 1 lm + 2 rs. **The bare-reply shape is gone**
— of the 155 still-missed answerable tasks only 1 (temporal) has a bare
gold; the rest are substantive-text golds:

- single_hop 72 (53 lm + 19 rs) — vocabulary mismatch beyond neighbor
  reach: query terms match neither the gold turn nor its ±2 neighbors;
  this is a §08 coverage/lexical problem, not context propagation.
- multi_hop 35 (17 lm + 18 rs), unchanged any@10 — golds span sessions;
  intra-session ctx can't bridge them. §10 fusion/pack composition.
- temporal 20, open_domain 17 — same coverage shape; temporal residuals
  are session-window questions where evidence sits outside the nominated
  turn's neighborhood entirely.

## Latency / cost

- t_ctx p50 15.2ms / p95 19.6ms / max 254ms; SQL statements p50 2 / max 3 (2 = nominated-session lookup + member fetch)
- e2e p50 660→710ms (+50), p95 1572→1630ms — the ~35ms beyond t_ctx is extra candidates through fusion/packing
- injected/query p50 224, boosted p50 455 (8 lanes share the one lookup)
- pack tokens mean +14 (~1%)

The ≤3ms target is unreachable on this store *because seq is degenerate*: positions must be materialized from a full member fetch (~9.3ms SQL + ~5ms Python, amortized over 8 lanes). On a store with real seq, stmt 2 becomes `session_id=? AND seq BETWEEN lo-W AND hi+W` range probes
(batched per session over the nominated-min/max, ~2-6 rows each) → well
under 3ms and still 2 statements. The arm numbers therefore transfer; the
latency constant does not — re-measure t_ctx on a seq-bearing build.

## Verdicts

**Recommended: mode=propagate_inject, w=0.9, W=2, M_ctx=25.**

- Δ answerable any@10 **+0.041** (0.583→0.624); Δ all@10 +0.039; Δ any@20 +0.057; Δ ndcg@10 +0.019; Δ mrr@10 +0.014; Δ session any@10 +0.006 (+0.035 session mrr); Δ cat-5 adversarial **+0.099**
- lane_miss 99→79 (−20%), rank_shift 91→76 (−16%), abstain 58→48, delivered +44 net (60 fixed / 16 regressed)
- cost: +15ms ctx stage (p50), +50ms e2e p50, 2 SQL statements per
  request, ~1% pack growth; zero contamination on 149k audited edges
- runner-up pi-w0.7-W2-M50 (+0.037, best lane_miss 76) trades a little
  precision for recall; pi-w0.9-W1-M50 wins cat-5 (+0.169) if premise
  questions are prioritized, at −0.013 overall any@10
- ctxfield (V8-06.03) rejected: +0.023 any@10 at 209-385ms/query, worse
  mrr; keep it out of the build
- inject cap 2·M_ctx never binds adversarially (injection p50 ≤447/query  across all 8 lanes)