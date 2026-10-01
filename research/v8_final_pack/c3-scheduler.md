# V8-14.03 two-phase scheduler + SB8-10 100K twin — measurement report

Repo: verbatim @ 1c86f4f (read-only; all prototype code in `/tmp/vt/`). Store rebuilt via
`eval.v7.track_r` → `/tmp/v7w/mem.db`; probes run against a `VACUUM INTO` snapshot
(`/tmp/v7w/mem_snapshot.db`, 4,143 units) and a SQL-replicated 100K twin
(`/tmp/v7w/mem100k.db`, 103,575 units = 4,143×25, 203 × ≤512-row vector blocks,
3.63M edges, 342K mentions, 2.9GB).

**Replication fidelity caveats** (direct-SQL, not re-ingest): replicated units share
`source_id` (sources stay 2,877, so per-source fanout is 25× higher than organic growth);
token-suffix jitter keeps dedup from collapsing rows, so df inflation reflects real
distinct-term growth only partially — replicated entities keep the same entity ids, which
slightly understates ent-lane breadth; edge/mention counts scale 25× linearly.
Vector blocks were re-packed into the a3 512-row layout, so dense numbers are faithful
to post-a3 production layout.

## Task A — three scheduler arms × deadline 500/1000/2000 (dev scale, n=124)

elapsed = wall ms inside `run_search` (lanes + S3-S8 tail). `over25` = queries with
elapsed > deadline+25ms; `any10` = any gold in delivered top-10.

| cell | p50 | p95 | p99 | over25 | overshoot p95 | any10 |
|------|-----|-----|-----|--------|---------------|-------|
| a@500  | 424 | 587  | 657  | 21 | +87  | 0.548 |
| b@500  | 443 | 592  | 701  | 30 | +92  | 0.581 |
| c@500  | 481 | 670  | 704  | 45 | +170 | 0.597 |
| a@1000 | 447 | 738  | 825  | 0  | −262 | 0.573 |
| b@1000 | 454 | 1084 | 1175 | 17 | +84  | 0.573 |
| c@1000 | 479 | 1149 | 1248 | 18 | +149 | 0.589 |
| a@2000 | 456 | 1118 | 1192 | 0  | −882 | 0.573 |
| b@2000 | 461 | 1174 | 1226 | 0  | −826 | 0.605 |
| c@2000 | 476 | 1228 | 1262 | 0  | −772 | 0.597 |

Arms: a = production per-slice `deadline.allocate`; b = cheap-first phase-1 (lex,time
run to completion) + expensive lanes dealt the remainder via a fresh `allocate`;
c = b but expensive lanes each get `deadline = full remaining` (shared hard deadline)
+ chunked dense scan (256-row preemption) installed.

### Mechanism (what actually drives the numbers)

- Dev-scale unbounded lane costs: **graph ~230ms, dense ~70-230ms, lex ~38-43ms**,
  ent ~7ms, fuzzy ~3.5ms, time/typed/obs <1-11ms. Lanes alone ≈ 380-420ms p50
  (`t_lanes` column); the S3-S8 tail adds ~70-100ms. **The 500ms envelope is already
  ~150% consumed by lane work before scheduling matters** — cheap-first ordering
  cannot fix intra-lane cost; it only decides which lanes get preempted.
- any10: b/c ≥ a at every budget (+0.03 to +0.06). Cheap-first removes the
  sequential-ordering penalty where late lanes (typed/obs) get starved only *after*
  expensive lanes ate the budget; dealing the remainder to dense/graph buys recall
  (graph `ok`: c@500 103/124 vs b 0/124 where allocated slices cut it mid-scan).
- Overshoot: a@1000/2000 has **zero** over-25ms queries; b/c at 1000 have 17-18.
  Cause: with the whole remainder dealt to each expensive lane, lanes that preempt
  on `slice.deadline` run right up to the global wall — graph's 64-iteration check
  granularity (~230ms/iter-batch) then overshoots the global deadline. c@500 worst
  (45, p95 +170ms) — the shared deadline lets every expensive lane start *and* run
  to the wall, so the tail lands on the deadline instead of an earlier slice bound.
- Chunked dense scan (c) barely matters at dev scale: dense is ~70ms, fits its
  slice anyway. At 100K it still can't help (see B(a)).
- Lane-side preemption inside graph (existing `expired()` checks) is what produces
  `partial` statuses — b@2000 graph partial 96/124, c@2000 graph ok 124/124.

### Harness note (for reviewers)

An earlier run of arms b/c was corrupted by a splice bug — my `_s2_sched` replacement
matched only the S2 header comment (S2 contains an *inner* `# ---` divider), so the
production lane loop remained after the splice and every b/c query ran **both**
schedules. Fixed by extracting through the S3 divider + asserting the production loop
text is gone. Numbers above are from the clean rerun; the corrupted artifact is
retained at `task_a_polluted.json` for provenance.

## Task B — 100K twin (SB8-10)

| probe | dev scale | @103,575 units | verdict vs target |
|-------|-----------|----------------|-------------------|
| B(a) dense scan, full | ~70ms | **p50 769ms, p95 812ms** | ✗ lane <30ms impossible |
| B(a) dense under 30ms slice | ~70ms | **~600-615ms then `partial`** | ✗ preemption can't help |
| B(b) lex lane | ~38ms | **p50 250ms, max 558ms** | ✗ ~7× slower |
| B(b) lex nomination | ~1-2K | **mean 17,779, max 44,225 of 71,925 eligible** | ✗ df inflation; cap-200 output still covers |
| B(c) `make_eligible` | ~0.03ms (cached) | **~206ms fresh** | ✗ breaks 150ms alone |
| B(c) `_watermarks` sig | ~ms | **~275ms** | ✗ the a4 cache *check* alone >150ms |
| B(d) ingest `add` | ~60ms/doc | add p50 21.9ms | ✓ enqueue cheap |
| B(d) drain (infer+jobs) | — | **35.6s/30docs ≈ 1,211ms/doc** | ✗ write-path ~20× degradation |
| B(e) e2e @2000 deadline | p50 456 | **p50 2,273ms, p95 2,692ms** | ✗ 15× over 150ms target |

Lane p50s inside e2e @100K: dense 762, graph 609, lex 299, ent 247, fuzzy 115,
tail(S3-S8) ~67-99ms. 58/60 queries overshoot the 2000ms deadline.

### Which stage breaks first at 100K

**The eligibility predicate construction — before any lane runs.** `make_eligible`
(~206ms) and the a4 cache signature `_watermarks` (~275ms, COUNT/MAX UNION digest over
3.6M-edge + 103K-unit tables) each independently exceed the 150ms e2e target. Even if
eligibility is served from cache, the *signature check* that validates the cache is
itself >150ms at this scale. Behind it, dense (769ms — linear f32 scan, ~7.7ms/1K rows,
no approximate index) and graph (609ms) make lanes alone ~2s.

Dense under a 30ms slice still burns ~600ms: the per-block deadline check only runs
between 512-row blocks, but at 100K the vector-matrix gather + first blocks' matmul
dominates — chunking can't reduce the fixed cost of touching 103K vectors. a3's
"f32 extends brute-force to ~200K" claim refers to *throughput feasibility*, not the
30ms lane budget: brute-force f32 at 103K is ~770ms per query — technically scannable,
but 25× the lane budget.

Lex nomination at 100K: mean 17.8K docs nominated per query (vs ~1-2K at 4K) — df
inflation on replicated speaker/content terms. The cap-200 output still produces 200
candidates, but scoring ~18K docs costs ~250ms inside the lane.

Ingest: `add` enqueue is fine (22ms/doc), but the drain side — harvest/infer/edges —
totals ~1.2s/doc, ~20× the dev-scale write path. Not quadratic in measurement, but
materially degraded; edge-adjacency work per doc scales with corpus size.

## Verdicts:

1. **R_post recommendation: do not implement R_post as specced.** The two-phase
   cheap-first schedule buys a *recall* bump (any10 +0.03/+0.06) at dev scale but
   *worsens* deadline bounding: a@1000/2000 overshoots 0 queries; b/c overshoot 17-45.
   The regression is structural — the expensive lanes (graph ~230ms, dense ~70-230ms)
   exceed any plausible share of a ≤500ms envelope, so giving them "the remainder"
   just relocates the wall, and graph's 64-iter preemption granularity (+~230ms
   worst-case) lands *past* the global deadline. If R_post ships at all, it needs
   lane-side preemption as a *hard* contract (per-64-row or timer checks in graph,
   not just dense/lex) AND a global-wall guard, not just slice realloc.
2. **Scheduler ship/skip: skip for latency; consider it only as a recall lever.** The
   measurable win is any10 (+0.03-0.06), not p95 headroom — p95 got worse under b/c
   (1084-1228ms vs 738-1118ms @1000/2000) because expensive lanes consume the full
   remainder instead of early-skipping under per-slice budgets. At dev scale the
   dominant problem is intra-lane cost (graph 230ms + dense 230ms ≈ the entire 500ms
   envelope), which ordering cannot fix. Ship criteria not met: "true ≤timeout+25ms"
   fails worst under the new schedule.
3. **100K readiness gaps, ranked:**
   1. **Eligibility path** — `make_eligible` ~206ms + `_watermarks` ~275ms *before*
      lanes run; both exceed the 150ms target alone. Needs a cheaper cache signature
      (e.g. PRAGMA data_version only, or a written generation counter) and a bounded
      eligible-set materialization.
   2. **Dense lane** — 769ms linear f32 scan; no amount of slicing/chunking keeps it
      <30ms because the fixed gather cost dominates. Needs an approximate index
      (HNSW/IVF) or a precomputed candidate subset — a3's brute-force bound does not
      extend to 100K for a lane budget.
   3. **Graph lane** — 609ms and its preemption granularity (~64 iters) is too coarse
      to bound; either delete the lane (a1 already showed OFF dominates ON) or add a
      real deadline check per iteration.
   4. **Lex nomination** — ~18K docs scored/query at 100K (df inflation); cap-200
      covers output but the in-lane scoring cost (~250ms) doesn't scale. Needs df
      floors / min-df pruning or an early-terminate top-k.
   5. **Ingest drain** — ~1.2s/doc (20× dev) — jobs/edges per doc scale with corpus;
      affects write throughput, not recall latency, but will throttle bulk ingest.
