# Beat-it post-fix verification — LoCoMo dev split latency + statement profile

**Repo**: `github.com/mosesman831/verbatim` @ `1c86f4f` (\"Beat-it retrieval plumbing fixes\"), read-only clone at `/tmp/vscratch/verbatim`, zero commits/pushes, all scratch under `/tmp`.
**Store**: rebuilt per spec — `VERBATIM_EVAL_LOCOMO=1 python3 -m eval.v7.track_r --dataset locomo --split dev --arms verbatim --workdir /tmp/v7w` → `/tmp/v7w/mem.db`. **2,877 sources, 4,143 units** (baseline had 7,020 — `UNITS_DERIVER_VERSION=v1.1` deduped session twins), 2,877 `unit_vectors_block` rows (~1.44 units/block), 4,143 `unit_fts` rows, **145,310 `graph_edges`** (empty at baseline — newly populated by `project_units_v7` running `build_edges` in-tx), 13,683 `entity_mentions`, 853 `entity_canon`, 38 `observations_v7`, 1,248 `events_v7`, 156 `state_facts`, 219 `preferences`.
**Track-R eval itself** (`/tmp/v7w/track_r_dev.json`, all 990 tasks, timeout_ms=2000): wall p50 **669.2ms**, p95 1529.2, mean 779.6; item any@10 **0.549**, any@20 0.661, all@10 0.461, ndcg@10 0.324, abstain 0.184, zero_rate 0.000; ingest 910s, 15,753 jobs, 0 failures.
**Probe driver**: `/tmp/probe_v7.py` — loads `mem.db` via `Memory(path=..., user_id=\"track-r\", worker=\"external\", encoder=\"hashing\")` (matching the arm), calls `search(query, limit=64, timeout_ms)`; wraps `facade._v7_run_search`/`_v7_make_eligible`/`dense._fetch_unit_rows`/`dense._KeyEligibility.__init__`; `conn.set_trace_callback` on the thread-local reader; classifies FTS5-internal shadow statements (the `-- ` prefixed ones) by real table. n=97 queries per budget (20/category × 5 cats, strided). Runs: `/tmp/probe2000b.json`, `/tmp/probe500.json`, cProfiles at `*.cprofile.txt`.

**Measurement caveat vs baseline**: r2 measured ~8,932 stmts/query at timeout_ms=8000 with `pool.max_facets=2` *firing* (16 lane invocations/query). On this HEAD the facade's `_v7_query_view` never populates `QueryViewV7.facets` → **8 lane invocations/query, zero facet re-runs**. Part of the post-fix statement/latency drop is the facet path not triggering, not only the eligibility batching — the (d) evidence below isolates the eligibility win specifically.

## (a) SQL statements per query — post-fix census

| | pre-fix (r2, @8s) | post-fix @2000ms | post-fix @500ms |
|---|---|---|---|
| statements p50 | **~8,932** (mean 10,792) | **4,529** (mean 5,648, p95 10,211, p99 13,881, max 16,931) | **4,188** (p95 9,460, p99 10,604, max 15,872) |

**−49% at p50.** Per-query mean census @2000ms (corrected FTS5 shadow attribution):

| shape | /q @2000 | /q @500 | who |
|---|---|---|---|
| `fts5: unit_fts_docsize` | **3,783** | 3,726 | FTS5-internal bm25 length probes (lex lane) |
| `fts5: unit_fts_content` | 706 | 706 | FTS5 external-content row fetches |
| `units WHERE unit_id=` (per-unit) | **486** | 71 | **graph lane N+1** — `graph.py::unit_row` per visited node |
| `fts5: unit_fts_idx` | 328 | 326 | FTS5-internal |
| `fts5: unit_fts_tri_idx` | 202 | 202 | FTS5-internal (trigram) |
| `fts5: unit_fts_stem_idx` | 87 | 87 | FTS5-internal (stem channel) |
| `PRAGMA data_version` (internal) | 30 | 30 | sqlite auto |
| `PRAGMA ?` | 17 | 17 | misc |
| `units` scans | 19 | 19 | eligibility `_load` + lexical `_universe` |
| `unit_fts MATCH` / `stem MATCH` / `tri MATCH` | 14 / 10 / 1.6 | same | outer MATCH calls |
| `lex_df` | 12 | 12 | lexical df loads |
| `sqlite_master` (has_table) | 12 | 11 | `_table_exists` probes |
| `units … unit_id IN(…)` | 7.5 | 5.5 | dense post-scan reverify + lex/ent batches |
| `entity_mentions`, `unit_vectors_block`, `graph_edges` | 7.0 / 5.0 / 3.8 | 3.8 / 5.0 / 3.6 | |
| `BEGIN`/`COMMIT` | 4.0 / 4.0 | same | tx wrappers |
| eligibility `_load` set (scopes/principals/grants/quarantine/purges/source_revisions/source_state/meta/jobs/delegations…) | ~20 total | ~20 | one bounded scan each |

**What dominates now**: FTS5-internal machinery = **~5,191/q @2000 (~90% of all statements)** — generated inside the ~24 outer `MATCH` calls by the FTS5 engine itself. Second is the graph lane's per-unit `SELECT … FROM units WHERE unit_id=?` (486/q — a *new* N+1 that didn't exist at baseline because `graph_edges` was empty). Eligibility's per-key `units IN` path is gone: dense-side `units` statements fell ~2,877 → ~8/q.

## (b) Wall latency

| timeout_ms | p50 | p95 | p99 | mean | max |
|---|---|---|---|---|---|
| 2000 | **600.0** | 1392.4 | 1511.9 | 713.7 | 1568.4 |
| 500 | **333.3** | 581.8 | 597.7 | 366.8 | 665.5 |
| baseline @8000 (r2) | 1818 | — | — | 2120 | 3029 |

-67% p50 vs baseline (1818→600; the arm's own 990-task run says 669). At 2000 the pipeline finishes inside the deadline everywhere (p99 1512 < 2000). At 500 the median fits (333) but **p95 exceeds the budget (582 > 500)** — tail queries still truncate.

## (c) Lane slice completion

Slice math (8 lanes sliced at S2 entry; `share = 0.5 + headroom·cost/\u03a3540`): @2000 → lex/dense **739.8**, fuzzy/typed 92.9, ent 74.4, time/obs 55.9, graph **148.3**. @500 → lex/dense **184.3**, fuzzy/typed 23.5, ent 18.9, time/obs 14.3, graph **37.2**.

Completion among lanes that attempted work (ok / attempted):

| lane | @2000ms | @500ms | t_lane p50/p95 @2000 | t_lane p50/p95 @500 | verdict |
|---|---|---|---|---|---|
| lex | 97/97 ok | 97/97 ok | 48.6 / 118.6 | 48.1 / 118.4 | fits |
| dense | **97/97 ok** (baseline: PARTIAL 32/32 @500-class budgets) | **97/97 ok** | 84.7 / 254.4 | 85.6 / 237.7 | **fits — eligibility batching worked** |
| ent | 96/97 | 90/97 | 9.4 / 29.3 | 9.2 / 31.1 | mostly fits |
| fuzzy | 23 ok / 31 attempted (66 no_oov_terms skip) | same (8 partial = no_vocab_table data issue) | 4.4 / 10.3 | 4.4 / 10.2 | fits |
| time | 19 ok / 41 (11 deadline + 11 scan_bound) | **9/41** (32 deadline) | 0.2 / 99.4 | 0.2 / 26.2 | starved @500 (14ms slice) |
| graph | **0/97 — partial/deadline on every query** | **0/97** | **293.8 / 842.2** | 51.1 / 144.2 | **never fits** (148ms/37ms slices) |
| typed | 35/40 | **0/46** (38 deadline + 8 deadline_exhausted) | 0.1 / 61.5 | 0.05 / 25.7 | starved @500 |
| obs | 57/59 | 8/59 (43 deadline + 8 deadline_exhausted) | 15.0 / 53.9 | 9.8 / 45.4 | starved @500 |

(`t_lane` is pipeline wall around the lane incl. any facet sub-runs — that's why dense shows 237 > its ~184ms slice yet reports `ok`: the *main* run finished in-slice; facet passes report their own status.)

Answer to (c): recalibrated LANE_COSTS + batched eligibility **let the heavy recall lanes (lex, dense, ent) finish inside slices at both budgets** — dense went from 32/32 PARTIAL to 97/97 ok. But the small lanes' 14–23ms shares at 500ms starve time/typed/obs, and **graph never completes at any budget** — its real cost is ~300ms+ p50 (up to ~840) vs a 37–148ms grant, so it burns deadline-partial work on every query.

## (d) Eligibility path — set-mode took over

- `keyelig_modes` = **`['set']` on every dense invocation** in both runs — `_KeyEligibility` resolves `_Eligible.unit_ids` (lazy frozenset) and `filter_keys` is a pure in-memory `{k for k in keys if k in self._set}` during `matrix.scan`'s per-block eligibility pass.
- `_fetch_unit_rows` calls: **~1.24/query** with key counts ≤200 (max 167) — that's only the post-scan reverify of produced hits (`dense.py:545`), not the per-block path. Per-block mode would be ~2,877 calls of ~2 keys.
- `elig_stats.calls = 0`, `checked_pairs=2877`, `withheld_pairs=0`, `units_fenced=4143` — **zero per-key SQL in the scan loop** (r2's check passes).
- `units … unit_id IN(…)` = 7.5/q total (baseline ~2,877 for dense eligibility alone, ~3,600 `SELECT * FROM units` overall).
- `make_eligible`/`_load` itself: **p50 17.9ms** (@2000), 18.4 (@500) — one bounded scan set inside the pinned snapshot; modest vs baseline's 23ms but now *shared* by all lanes.

## (e) Top CPU/SQL cost now — cProfile @2000ms, 12 queries (10.15s total, ~846ms/q)

| cum ms/q | symbol | note |
|---|---|---|
| 846 | `facade.search` → `_search_v7` 835 → `run_search` 796 | S2 lane loop = 676 (80% of wall) |
| **326** | **`graph.py:351 lane_graph`** | tottime only 21 — nearly all inside sqlite fetchall; **new #1 lane** |
| 263 | `sqlite3.Cursor.fetchall` (721 calls) | fat IN-list results: graph_edges edges + lexical paged fetches + block rows |
| 149 | `dense.py:368 lane_dense` | `matrix.scan` 125 (tottime 59 — numpy IS present (2.2.6) and engaged, but 2,877 blocks × ~1.44 units defeats vectorization) |
| 113 | `lexical.py:1445 lane_lexical` | `_score_candidates` 98; `_stem_of` **24,725 calls/q** (41ms) → `porter_stem` 2,184/q (33ms) |
| 71 | `sqlite3.execute` (543 calls/q) | was 63% of CPU at baseline — now ~8% |
| 40 | `obs.lane_obs` | |
| 39 | `fusion.rrf_fuse` | |
| 34 | `rerank_features.score_candidates` | |
| 33 | `str.join` (44,969 calls) | SQL/serialization glue |

Post-lane stages on a representative @500 rec: t_union 7.4 + t_rrf 22.1 + t_rerank_feat 24.6 + t_boost 2.8 + t_currency 0.6 + t_verdict 17.4 + t_pack 10.8 ≈ **86ms** — no longer \"<30ms\"; on a reduced budget the fixed post-pipeline is ~25% of wall.

## (f) The 150ms question — honest math @4,143 units

Measured @500ms p50 333 decomposes ≈ elig 18 + \u03a3 lane p50s ~209 + post ~86 + facade ~20. What each lever buys:

- **Cheap-lanes-first**: nearly worthless here — slices are computed once on the full budget (`allocate()` at S2 entry, never re-derated), and ordering only decides which lanes run before the *global* deadline, which isn't binding at p50 anyway (333 < 500).
- **Expansion gating (drop/short graph+typed+obs+time when union is already fat)**: saves graph's ~51ms partial waste and the starved lanes' ~35ms — gets p50 ≈ 245ms. Necessary, not sufficient.
- Then: elig 18 + lex 48 + dense 86 + ent 9 + fuzzy 4 + post ~86 ≈ **~245ms p50 floor** with recall lanes intact. **>150.**
- To actually reach 150: dense needs block repack (2,877 blocks × 1.44 units → ~512-row blocks so numpy vectorization engages; ~86→~20ms plausible), lex needs stem memoization / FTS-native bm25 prefilter (~48→~25), eligibility caching across queries (~18→~3), and a slimmed post-pipeline pool (~86→~35). Sum ≈ **95–120ms** — reachable only with all four structural changes *and* graph held off/gated. Any of them skipped → >150.

Extrapolation to the 7k-unit corpus the question assumes: current build is 4,143 units post-dedup; scan lanes are ~linear in units, so the same architecture at 7k lands ~p50 450–550ms @500 budget — further from 150, not closer.

## Verdicts

- **SQL/query**: p50 4,529 vs 8,932 baseline (−49%) — and the composition changed: the eligibility per-key N+1 is dead (dense `units` statements ~2,877→~8/q, `elig_stats.calls=0`, `_fetch_unit_rows` now only a ≤200-key post-scan reverify). What's left is engine-internal FTS5 shadow churn (~5,191/q ≈ 90%, inside ~24 MATCH calls — irreducible per-statement, reducible only by fewer/batched MATCHes) plus a **new** graph-lane N+1 (`unit_row` per visited node, 486/q @2000).
- **Latency**: p50 600ms @2000 / 333ms @500 vs 1818ms @8000 baseline (−67%); the arm's full 990-task eval confirms 669ms p50. Dense went from 1,480ms PARTIAL-everywhere to 86ms p50, 97/97 ok.
- **Deadline-fit**: plausible at 2000ms (p99 1512 < budget, all lanes complete except graph). **Not at 500ms** — p50 fits but p95 misses (582), and 4 of 8 lanes (time/graph/typed/obs) degrade to partial/deadline whenever they have real work. Recalibrated costs fixed the *heavy* lanes; small lanes now starve on 14–23ms shares.
- **Remaining bottleneck**: graph lane — 145,310 real edges now exist and its BFS needs ~300–840ms vs a 37–148ms slice: per-node `unit_row` probes (~486 stmts), fat `_latest_edges` fetchall's, and coarse per-round deadline checks → 97/97 deadline-partial at both budgets. Second-tier: lex's per-candidate `porter_stem`/`_stem_of` churn (~25k calls/q) and the 2,877 micro-blocks that keep `matrix.scan` Python-bound.
- **150ms quality budget**: not reachable by scheduling alone. Cheap-lanes-first is nearly free to skip (slices are pre-computed, not sequential) and expansion gating only removes lanes already starving — the floor with lex+dense+post intact is ~245ms p50. 150ms requires block repack + stem caching + elig caching + slim pools (≈95–120ms projected), i.e. structural work, not tuning.
