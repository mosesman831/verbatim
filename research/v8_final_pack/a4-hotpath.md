# V8 hot-path wins — measured on LoCoMo dev store (`/tmp/v7w/mem.db`, 4143 units / 2877 sources)

Tree: verbatim @1c86f4f, prototype at `/tmp/vv_proto` (uncommitted). Harness: `/tmp/measure.py` — 100 evenly-spaced dev queries for walls (`mem.search(q, limit=20, timeout_ms)`), 60 for stage attribution (`_search_v7(limit=64)`), 40 cProfiled calls. Both runs un-contended. Wall figures are `mem.search`; stage figures are `PipelineResult.stage.fields`.

Baseline: wall p50 **625.6 / p95 1481.1 ms** @timeout 2000; t_total p50 599.4. Proto: **611.3 / 1461.1**; t_total 572.1.

### Item 1 — stem memoization (bounded process LRU 65,536)

`stem_memo` (OrderedDict, `move_to_end` + `popitem(last=False)` beyond 65,536) added to `lexical.py`; `_stem_of`, the query-term stem at ~line 994, and fuzzy respell candidate stems route through it.

- **Calls/q**: `4,466.5 → 13.3` per query (−99.7%; the `porter_stem` primitive is now reached only on true misses; all `_stem_of` traffic hits the LRU). `_stem_of` ncalls drop out of the profile entirely (was 1.32M/40 calls, 2.11s).

- **Lex-lane delta**: `51.25 → 39.63` p50 ms (−11.6 ms, −23%); p95 `131.25 → 104.33`. Wall p50 improves −14 ms end-to-end.
- **Memory cost**: ~4,251 distinct tokens after 50 queries ≈ 0.98 MB (avg key+stem ~110 B + node overhead). Worst case at the 65,536 cap ≈ **14.4 MB** — bounded, acceptable.
- Failure mode: none — memoization is pure (stem is a pure function); eviction only costs a recompute.

### Item 2 — eligibility snapshot cache

Process LRU(16) in `eligibility.py`; `authorize(...)` still executes on every call — the cache skips only the inventory `_load`. Key:

```
(scope_id, principal_id, purpose, projection_generation,
 current_epoch(scope),                 # grants epoch
 PRAGMA data_version,                  # bumps on ANY committed write the snapshot sees — covers UPDATE-style payload/disposition/grant-revoke writes that never move a rowid
 store._write_epoch (when exposed),    # writer's own commit count
 _inventory_sig)                       # MAX(rowid) UNION ALL over units/sources/source_revisions/source_state/quarantine/purges/purge_targets/spans/source_envelopes/delegations — covers INSERT/DELETE
```

- **Cold vs hit** (same read snapshot, `make_eligible` direct): `19.97 ms → 0.024 ms` (~830×). In-pipeline: `make_eligible` falls out of the cProfile top-18 (was 44.7 ms cum/call — profile-inflated; micro ~20 ms).
- **Invalidation verified**: warm hit → commit a write tx → next read **miss** (22.8 ms rebuild) → following read **hit** again.
- **No payload bytes held**: cached `loaded` dict carries only eligible id sets/pairs + stats — ~0.3–0.5 MB/entry, ≤16 entries → ≤8 MB bound.
- **Fail-closed**: any exception building a key component (missing table, unreadable watermark, epoch query failure) → `key=None` → uncached path. Cache never substitutes for `authorize`.

### Item 3 — slim post-lane pool (`max(4*limit, 64)`)

`limit_cap = max(4*manifest.limit, 64)` applied to `fused` right after RRF (manifest limit is `_MAX_LIMIT=64` → cap 256). Measured fused pool is **median 612, max 798** on the dev store — the trim binds hard.

- `t_rerank_feat` p50 `14.99 → 6.80 ms`; direct decomposition (`score_candidates` on the same fused list): 669 cand → `16.2 ms`, 256 cand → `6.4 ms`, 100 → `2.7 ms`. The −9.8 ms is the trim, not memoization.
- `t_verdict`/`t_pack` unchanged (they already operated on `rerank_pool`/`limit`).
- **Recall/MRR**: pack diff vs baseline — 48/60 queries identical; 1 order-only swap; 11 queries swapped 1–3 items at positions ≥36 (one at pos 11). Recall@8 `25/49 → 25/49`, @20 `34/49 → 34/49`, @64 `38/49 → 38/49`, zero gold-item flips — RRF-rank >256 items rerank would have promoted land inside pack-tail slots that never covered gold on this sample. Statuses identical (83 ready / 17 insufficient).

### Item 4 — deadline-bounded post stages

`_POST_FLOOR_MS = 110` (calibrated post-lane p95). When `remaining_ms() < floor` at S4: RRF-order passthrough replaces `score_candidates`, `_structural_verdict` replaces classification below 40% of floor, `explain` payload trimmed — each move recorded in `coverage.security` (`deadline_post_cheap`, `explain_trimmed`, `fused_pool_trim`).

- **Overshoot** (`_search_v7` wall minus `deadline_ms`):
  - @500 ms: base max **+171 ms** → proto max **+150 ms** (p50 both under budget; p95 overshoot 98 → 88 ms).
  - @200 ms (40 queries): base p50 **+67.5 / max +125.4** → proto p50 **+28.1 / max +69.3** — overshoot roughly halved.
- **≤timeout+25 target: not met** — and cannot be met by post-stage bounding alone: a lane launched with remaining>0 runs its SQL to its own slice (graph ~290 ms p50) with no mid-lane preemption; by the time S4 runs, the wall is already over. The cheap path correctly fires (verified @300/120 ms) and bounds the *controllable* tail (−40 ms p50 overshoot @200).
- p95@500: `597.7 → 588.0` ms wall.

### cProfile top-10 (cumtime, 40 `_search_v7` calls)

Baseline:
```
ncalls    cumtime  function
40        49.72s   facade._search_v7
40        47.34s   pipeline.run_search
412       41.16s   lanes_base.run_one / _invoke
54        17.57s   graph.lane_graph
2756      13.21s   {sqlite3.Cursor.fetchall}
54        12.61s   dense.lane_dense  (matrix.scan 11.42s)
54         6.51s   lexical.lane_lexical (_score_candidates 5.56s)
320        5.28s   pipeline._stage_call
27569      3.59s   {sqlite3.Connection.execute}
40         2.15s   fusion.rrf_fuse
1317246    2.11s   lexical._stem_of          ← item 1 target
40         1.79s   eligibility.make_eligible ← item 2 target
40         1.70s   rerank_features.score_candidates
```

Proto:
```
ncalls    cumtime  function
40        49.79s   facade._search_v7
40        49.14s   pipeline.run_search
409       43.69s   lanes_base.run_one / _invoke
54        19.10s   graph.lane_graph
54        13.82s   dense.lane_dense  (matrix.scan 12.49s)
2602      13.74s   {sqlite3.Cursor.fetchall}
54         5.35s   lexical.lane_lexical (_score_candidates 4.21s)
301        4.56s   pipeline._stage_call
26808      3.83s   {sqlite3.Connection.execute}
40         2.27s   fusion.rrf_fuse
40         1.69s   obs.lane_obs
```
`_stem_of` (1.32M calls) and `make_eligible` both fell below the top-18 cutoff — the two caches took them off the hot path. `lane_lexical` cumtime 6.51→5.35s (−18%).

### Item 5 — aggregate projection toward SB8-09 p50 ≤120 ms

| stage | base p50 | proto p50 | projected (a1 ext + a2 doclen + a3 repack) |
|---|---|---|---|
| lane dispatch/facet overhead | ~40 | ~40 | ~40 |
| graph | 291.2 | 289.5 | ~10–20 (a1, external) |
| dense | 121.0 | 119.8 | ~120 — **residual** |
| lex | 51.3 | 39.6 | ~25 (a2 doclen est.) |
| obs | 13.7 | 14.3 | ~14 |
| ent | 8.7 | 8.5 | ~9 |
| fuzzy | 4.6 | 4.6 | ~5 |
| typed+time | 0.3 | 0.3 | ~0.3 |
| union | 3.1 | 3.0 | ~3 |
| rrf | 15.2 | 15.8 | ~16 |
| rerank_feat | 15.0 | 6.8 | ~7 |
| rerank_ce | 0.1 | 0.1 | ~0.1 |
| boost+currency | 1.8 | 1.5 | ~1.5 |
| verdict | 6.4 | 6.3 | ~6 |
| pack | 4.7 | 4.7 | ~2 (a3 est.) |
| eligibility | ~20 | ~0.03 | ~0.03 |
| **t_total p50** | **599.4** | **572.1** | **~230–245** |

- Even granting a1 (graph→~15), a2 (−15 lex), a3 (−3 pack): **~235 ms p50 — ≈2× over the 120 target**.
- **Residual blocker: the dense lane.** `embeddings.matrix.scan` is a pure-Python row scan over 4143 units × 384-dim (profile tottime leader: 10.2 s/54 calls). At ~120 ms p50 it alone consumes the entire SB8-09 budget. Needs a vectorized path (numpy `np.dot`/sqlite-vec) or index — est. −80–100 ms with numpy. Secondary residual: `rrf_fuse` ~16 ms p50 over ~612-candidate pools (p95 102–114); RRF-side capping or early-terminate would reclaim ~10 ms.

## Verdicts

1. **Stem memoization — SHIP.** −11.6 ms lex p50 / −14 ms wall; 4,467→13.3 porter calls/q; ≤14.4 MB bounded. Spec: `_STEM_LRU` OrderedDict cap 65,536 keyed on exact token; `_stem_of`, query-term stemming, fuzzy respell all route through `stem_memo`.
2. **Eligibility cache — SHIP.** ~20 ms→~0.03 ms/query; zero recall/verdict change (authorize still per-call). Key = `(scope, principal, purpose, generation, grant_epoch, data_version, write_epoch?, rowid_sig)`; `data_version` is load-bearing (catches UPDATE writes); wholesale invalidation = key mismatch — no explicit invalidation hooks needed. Fail-closed on any unreadable component. Note: OrderedDict is not thread-safe — pipeline calls are serialized by `Memory._lock` today; add a small lock if that ever changes.
3. **Post-pool trim — SHIP.** −8.2 ms rerank p50 (612→256 candidates); recall@8/20/64 identical on sample; coverage note logs trims. Honest cost: items at RRF-rank >256 lose their rerank chance — observed only at pack positions ≥36/never covering gold.
4. **Deadline post-bounding — SHIP with caveat.** Overshoot halved (@200: +67.5→+28.1 p50, +125→+69 max) but **≤+25 unattainable** without lane-side preemption — lanes started before exhaustion run uninterruptible SQL slices (graph ~290 ms). Recommend: extend the floor concept into lanes (`run_one` early-exit checks) or a hard "last-lane starts only if remaining > lane p95" gate.
5. **Aggregate — target misses.** Projected p50 ~235 ms after a1+a2+a3+my items vs 120 target. **Residual: dense lane ~120 ms** (`matrix.scan` — needs vectorization/sharding); secondary: `rrf_fuse` ~16 ms on 612-candidate pools.
