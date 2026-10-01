# V7 query-latency profile — where 382ms–1.8s per query goes

## Setup & method (provenance)

- The parent's `/tmp/v7-trackr-ezeh8hhn/mem.db` and `eval/v7/lane_miss_diagnosis.md` do not exist on this box (child sessions are separate VMs/filesystems). I rebuilt an equivalent store: cloned `mosesman831/verbatim` @ `892f2b8`, ran the real `VerbatimArm` ingest (`eval/v7/arms.py:684-788`) over `load_corpus('locomo', split='dev')` — 2,877 items / 990 tasks, sha256-pinned corpus (verified `79fa87e9...`). Result: `/tmp/v7-trackr-local/mem.db` — **15,753 jobs drained, 0 failures, `settle_state=ready`; 7,020 units, 2,877 sources, 2,877 `unit_vectors_block` rows, 7,020 `unit_fts` rows** (measured via sqlite).
- Instrumentation: no repo edits. `run_search` already self-times every stage into `PipelineResult.stage.fields` + `stage.t_lane` (pipeline.py:914–926, :1240–1258); I captured each `PipelineResult` by wrapping the facade's module-global `_v7_run_search` (import at facade.py:165, call at :2929). Additional timed wrappers around `eligibility._load`/`make_eligible`, `Memory._v7_query_view`/`_search_v7`/`_v7_hits` (facade.py:2810/2858/2937), `pipeline._materialize_details` (:555)/`_apply_source_currency`/`_build_explain`, lexical internals (`_universe` 498, `_resolve_eligibility` 587, `_eligible_corpus_stats` 683, `_maintained_df` 785, `_collect_postings` 1052, `_score_candidates` 1119, `_phrase_flags` 1304), `matrix.scan` (embeddings/matrix.py:672), and a `conn.set_trace_callback` statement log on the thread-local reader (`Store._reader`, store.py:838–857). Separate `cProfile` pass over 6 queries.
- Sample: 32 queries, ~7 per LoCoMo category strided across the 990-task dev split (`single_hop` 429, `adversarial` 213, `temporal` 175, `multi_hop` 126, `open_domain` 47 — measured counts), `limit=10`, `timeout_ms=8000`, warm.
- Caveat on absolute numbers: walls are higher than the parent's 382ms p50 because the Track-R harness used the default `timeout_ms=500` — under 500ms lanes get deadline-truncated (PARTIAL) and return early. At 8s the pipeline spends what it *wants*: ~1.8s p50. That gap IS the diagnosis — the pipeline is ~4× over budget before truncation. My wrappers add ~5–10%.

## Headline numbers (measured)

- Wall p50 = **1,818ms**, mean 2,120, max 3,029; every query `ready`, 10 items.
- **~8,932 SQL statements per query (p50; mean 10,792)** — the engine is statement-bound.
- `sqlite3.Connection.execute` tottime = **10.33s of the 16.4s cProfile run = 63% of all search CPU**; `lane_dense` = 12.2s = 74%.

## Stage-latency table (ms/query, n=32)

| Stage | p50 | mean | max | share | code site |
|---|---|---|---|---|---|
| facade `_search_v7` total | 1,813 | 2,115 | 3,023 | ~99% | facade.py:2858 |
| \u251c S0 eligibility build (`make_eligible`\u2192`_load`) | 23.0 | 23.9 | 46.6 | 1.3% | eligibility.py:683, :491 |
| \u251c S1 query view (`_v7_query_view`) | 5.2 | 7.9 | 18.0 | 0.3% | facade.py:2810\u2192query_view.py:220 |
| \u251c `run_search` S2–S8 (`t_total`) | 1,785 | 2,082 | 2,985 | ~97% | pipeline.py:797 |
| \u2502  S2 lane loop (`t_lanes`) | **1,730** | 2,045 | 2,953 | **~95%** | pipeline.py:859–926 |
| \u2502  S3 union recheck (`t_union`) | 2 | 2 | 3 | — | pipeline.py:931–954 |
| \u2502  S3 RRF fuse (`t_rrf`) | 8 | 14 | 176 | 0.5% | fusion.py:322 |
| \u2502  S4 feature rerank (`t_rerank_feat`) | 9 | 10 | 13 | 0.5% | rerank_features.py:747 |
| \u2502  S5 CE rerank (`t_rerank_ce`) | 0 | 0 | 0 | — | no CE hook wired (pipeline.py:1065) |
| \u2502  S6 boosts + S6b currency | 1.4 | 1.4 | 1.7 | — | pipeline.py:1092–1116 |
| \u2502  S7 verdict (`t_verdict`) | 4 | 4 | 6 | — | pipeline.py:1121–1162 |
| \u2502  S8 materialize+pack (`t_pack`; `_materialize_details` alone 2.2) | 5 | 5 | 5 | — | pipeline.py:555, :1187–1225 |
| \u2514 post (`_v7_hits` units IN-join) | 0.9 | 0.9 | 1.0 | — | facade.py:2937–2968 |

**Per-lane p50 (inside `t_lanes`, from `stage.t_lane`):**

| lane | p50 | max | candidates | status |
|---|---|---|---|---|
| dense | **1,480** | 2,593 | 200 (capped) | **PARTIAL/deadline 32/32** — `blocks_scanned 2,074/2,877` |
| lex | 143 | 389 | 200 (capped) | OK |
| graph | 16 | 38 | 0 | OK-empty (entities/graph_edges unpopulated per diagnosis) |
| ent | 7 | 42 | 200 | OK |
| fuzzy | 6 | 21 | 13 | OK |
| time | 0 | 145 | 0 | OK-empty |
| typed | 0 | 116 | 0 | OK-empty (t2_facts unpopulated) |
| obs | 0 | 1 | 0 | OK-empty (obs_v7 unpopulated) |

Note `pool.max_facets=2` (types_v7.py:416) means most queries run **every lane twice** — cProfile counted 96 `run_one` calls / 6 queries = 16 invocations per query (facet subdivision, pipeline.py:877–912). The dense N+1 below happens *per invocation*.

## Top-3 hot spots (evidence)

### 1. Dense lane eligibility N+1 — ~80% of the wall — THE hot spot

Chain: `lane_dense` builds `_KeyEligibility(conn, ctx.eligible, ...)` (dense.py:482) — callable mode since `ctx.eligible` is the `_Eligible` predicate. `matrix.scan` iterates **all 2,877 block rows** via `_block_rows` (matrix.py:774\u2192486: one SELECT streaming every block's `rowmap_blob`+`data_blob`). Per block it calls `_eligible_keys(eligibility, keys)` (matrix.py:786, defined :639) \u2192 `filter_keys(list(keys))` \u2192 `missing` keys \u2192 `_fetch_unit_rows` issues **`SELECT * FROM units WHERE scope_id=? AND generation<=? AND unit_id IN (...)` per block — ~2–3 keys each** (dense.py:142–157, :160–190). `_LOOKUP_CHUNK=400` (dense.py:73) can't batch because each call only sees one block's rowmap.

Why ~3 keys/block: blocks are minted **per source_embed job** — 2,877 sources \u2192 2,877 blocks of ~2.4 vectors (measured `(2877 blocks, 7020 rows)`). \"Per-block\" == \"per-source\".

Evidence:
- Direct replay of the exact filter path (same conn, same keys): **1,768–1,830ms** for 2,877 statements (~0.6ms each — `SELECT *` + `ORDER BY unit_id, generation DESC` over the IN list + dict materialization in `_fetch_unit_rows`).
- cProfile: `_fetch_unit_rows` 16,425 calls / **10.26s of 16.4s total**; `filter_keys` itself 16,414 calls / 10.55s cumulative.
- Lane stats: `partial=true, stop_reason=\"deadline\"` on **all 32 queries** — even at an 8s request budget the lane burns its ~1,454ms cost-weighted slice and exits at 2,074/2,877 blocks. At the production `timeout_ms=500`, the slice allocator (deadline.py:131–157; costs lex 6 / fuzzy 4 / dense 6 / ent 2 / time 3 / graph 6 / typed 4 / obs 2 = 33) grants dense ≈95ms \u2192 it inspects ~150/2,877 blocks and returns a truncated top-200 — exactly the \"deadline truncation\" the parent's diagnosis measured.

**Verified fix (same box, same conn):**
- `block_row_keys` (matrix.py:531 — rowmap decode only, no data blobs; already used by `_coverage_lag`): 8.6ms for all 7,020 keys.
- One chunked `_fetch_unit_rows` over the whole key union (18 queries of 400): 53.0ms.
- `eligible(row)` eval per row: 10.7ms. **Total: 72.5ms vs 1,768ms \u2192 24×.**
- `scan(eligible_keys=<frozenset>)` then costs zero SQL; pure cosine top-200 over 7,020 vectors = **64–73ms** (numpy path, hashing:subword-ngram:v1:hdr:v1 space). Dense lane net ≈ **145ms vs 1,480ms \u2192 ~10×**, and `partial` disappears \u2192 *recall improves too*.

### 2. Lexical lane — Python BM25F + FTS5-internal per-row probes — ~8% of wall

`lane_lexical` p50 143ms, max 389 (lexical.py:1397). Measured internals per query:
- `_score_candidates` 75–91ms (max 246) — per nominated doc: `_fetch_fields` bytes + `_field_lens` docsize (paged 256/400 — lexical.py:238–290), `_tokenize` regex + `porter_stem` (memoized `_SC_MEMO`) per text token, `bm25f_score` per term (loop 1168–1270). cProfile: 13 calls / 1.94s cumulative.
- `_universe` 32–55ms — second full `units` scan per query after `_load` (lexical.py:535–537); `_UNI_MEMO` dedupes only within one ctx.
- `_resolve_eligibility` 14–134ms — loops the universe calling `elig(u)` per row (605–622); `_Eligible.__call__`/`_decide` invoked ~216k times / 6 queries overall (~36k/query, ~90ms CPU total across lex+union+dense).
- `_collect_postings` 8–160ms (per-term `unit_fts MATCH` + `unit_fts_stem MATCH`, 1084–1090); `_phrase_flags` ~3ms.
- Trace: ~2,058 `unit_fts_content` + ~6,100 `unit_fts_docsize` internal statements per query — FTS5 external-content/docsize shadow probes (`-- SELECT` trace comments) under the paged fetches. The per-statement overhead, not the row count, is the cost.

### 3. Whole-scope rescans repeated every query — ~90–110ms of cacheable work

- `eligibility._load` 23ms p50 — scans ALL units `WHERE scope_id=? AND generation<=?` + `source_revisions`×`sources` + `source_state` + `quarantine` + `purges`/`purge_targets` + taint spans (eligibility.py:504–645).
- `lex _universe` 32–55ms — same `units` table again.
- dense `_coverage_lag` live-set = **correlated `MAX(u2.generation)` subquery per row** (dense.py:287–296): 19.8ms measured; a `JOIN (SELECT source_id,revision,MAX(generation) ... GROUP BY)` rewrite = **10.4ms** (verified, same result set). Runs ~2×/query.
- `has_table`/`_table_exists` \u2192 `SELECT 1 FROM sqlite_master` ≈ 2.3ms × ~114 calls/query (~9ms/query) — no schema cache anywhere.
- `standing_queries` scope scan ≈ 8ms/query.
- `_v7_known_canons` `SELECT canon ... LIMIT 2000` (facade.py:2844–2856) inside query-view build — small here (~2k rows).

## Statement census (per query, from the trace log — approx; time-to-next-event attribution)

| Statement shape | ~count/query | ~ms/query | owner |
|---|---|---|---|
| `units WHERE unit_id IN (≤3)` | ~2,877 | ~1,400 | dense `_fetch_unit_rows` (dense.py:179) |
| `unit_fts_docsize id=?` (FTS5 internal) | ~6,100 | ~12 | lex `_field_lens` pages |
| `unit_fts_content fts_row_id=?` (FTS5 internal) | ~2,058 | ~12 | lex `_fetch_fields` pages |
| full `units` scans ×3 | 3 | ~90 | `_load`, `_universe`, `_coverage_lag` |
| `sqlite_master` probes | ~114 | ~9 | `has_table` |
| `unit_fts(MATCH)/stem MATCH` postings | ~30–50 | ~10–30 | lex `_collect_postings` |
| everything else (blocks rowmap scan, canons, standing_queries, pack joins\u2026) | <50 | ~20 | misc |

## Where the 382ms–1.8s actually lives (synthesis)

~97% of wall is the S2 lane loop; ~80% is dense alone; ~63% of *all* search CPU is `sqlite3.execute` — dominated by the per-block `units IN (...)` eligibility lookups. The pipeline runs ~9k statements where ~50 would suffice. Post-lane stages (fuse/rerank/verdict/pack/hits) sum to <30ms — the tail is already fine, and `_materialize_details`+`pack` (the ask's S8 suspicion) is ~5ms — a non-issue at this scale.

## Cheapest staged-retrieval design \u2192 p95 ≤ 150ms @7k, scaling to 100k

- **C1 — dense eligibility pre-resolution (~10 lines in `lane_dense`, zero semantics).** `all_keys = block_row_keys(conn, encoder_id, scope_id, snap_generation)` \u2192 `ok = eligibility.filter_keys(all_keys)` \u2192 `scan(..., eligible_keys=ok)` (dense.py:482–501). Measured: **1,480\u2192~145ms**, kills the PARTIAL truncation. Expected wall p50 after C1 alone: **~430ms** (1,730 lane loop \u2212 ~1,335 dense savings + small residuals).
- **C1b — materialized eligible set (small interface seam).** `_Eligible` already loaded the live unit map in `_load`; expose a `keys()`/snapshot frozenset on `LaneContextV7.eligible` \u2192 dense ≈ **82ms** (scan 73 + rowkeys 9), `_resolve_eligibility` and union recheck become set ops (~0). Keeps the callable for fail-closed safety.
- **C2 — one inventory per (scope, generation, epoch), shared.** `_load` + `_universe` + `_coverage_lag` live set + dense key fetch all re-read the same `units` slice (~100–130ms/query at 7k, all O(units) — ~1.5s at 100k). Cache keyed `(scope_id, projection_generation, epoch)`; also cache `has_table` per conn.
- **C3 — cheap-lanes-first staging + expansion gate.** Measured lane costs (p50 ms): fuzzy 6, ent 7, time ~0–145, graph 16, lex 143, dense 1,480\u2192145; typed/obs skip ~free while their tables are empty (reorder dynamically as they fill). Proposed S2 order: **[ent, fuzzy, time, lex] \u2192 fuse probe \u2192 gate \u2192 [dense, graph] \u2192 [typed, obs]**. Gate on `len(union) >= 4×limit` AND top-RRF margin \u2265 \u03c4; lanes already return honest `LaneOutput`s — the gate is loop order + early exit inside pipeline.py:865–925, reusing the cost-weighted slice allocator (deadline.py:131–157) on *remaining* budget. Estimated: **p50 ≈ 60–100ms, p95 ≈ 150–200ms @7k** after C1+C2.
- **C4 — lexical scoring via FTS5-native bm25 (optional).** `unit_fts MATCH`+`bm25(unit_fts)` ranks in SQL — the harness's own `FTS5Arm` ran ~0.5–3ms p50 at this scale vs `lane_lexical`'s 143ms. Keep Python `_score_candidates` for the merged top-N (bm25f multi-field + stem channel differ — treat as a scoring-form change to validate on the dev split) or as FTS-prefilter + Python-rescore. Lex ≈ 20–40ms.
- **C5 — 100k headroom.** Pure cosine scan is O(#vectors): ~65ms/7k \u21d2 ~0.9s/100k; per-source blocks make fetch overhead O(#sources) — repack `unit_vectors_block` into ~512-row blocks (~200× fewer fetches) or land an ANN (sqlite-vec/hnswlib) + eligible-set post-filter; eligibility evals \u2192 bitset/generation-set rather than per-row dict calls. The expansion gate bounds p95 only if dense is optional-or-pruned.

## Caveats / limitations

- Store equivalence: rebuilt via the same `VerbatimArm` path on the same pinned corpus — unit/source counts match the diagnosis (~7k, session+turn duplicates present: 7,020 units / 2,877 sources). I could not diff against the parent's actual `mem.db` (not on this VM).
- Trace attribution (`sql_ms`) bounds each statement by time-to-next-event — counts are exact, ms are approximate; the wrapper timers (`extra_ms`, `stage.fields`, cProfile) are authoritative.
- Numbers are warm-cache, single-threaded, this VM. Parent's box may differ in absolute ms but the attribution ranking is structural (statement counts don't move).
- `t_rrf` max=176ms on one query — a fused-pool outlier when lanes overfill (pool 200×8 lanes pre-dedup); irrelevant once C1+C3 shrink candidate churn, but worth a glance if p99 stragglers appear.

## Verdicts

| # | Recommendation | Expected lift (measured basis) | Cost / risk |
|---|---|---|---|
| 1 | **Batch-resolve dense eligibility in `lane_dense`** — `block_row_keys` + one `filter_keys(all)` + `scan(eligible_keys=set)` (dense.py:482–501; matrix.py:531). Verified on this store: filter 1,768\u219272.5ms, lane 1,480\u2192~145ms (10×); dense stops returning PARTIAL (32/32 truncated today) — should also recover recall the 500ms-deadline truncation was losing | wall p50 ~1,820\u2192~430ms | ~10 lines; zero semantic change; needs no interface work |
| 2 | **Materialize the eligible unit set once per search** (expose off `_Eligible`/`LaneContextV7`; set-mode `filter_keys` = zero SQL everywhere — dense ≈82ms, lex eligibility + union recheck ≈0) | \u221260–90ms/query beyond #1 | small seam; keep callable fallback for fail-closed |
| 3 | **Shared per-generation inventory cache** — `_load` + `lex._universe` + `_coverage_lag` all rescan `units` (eligibility.py:504, lexical.py:535, dense.py:287); rewrite `_coverage_lag`'s correlated `MAX()` as GROUP BY JOIN (measured 19.8\u219210.4ms) | \u2212100–130ms/query; **required** at 100k (all O(units)/query) | cache key (scope, generation, epoch); low risk |
| 4 | **Cheap-lanes-first staging + expansion gate** in the S2 loop — ent/fuzzy/time/lex \u2192 fuse probe \u2192 dense/graph only when union <4×limit or RRF margin low (reuse cost-weighted slices on remaining budget) | p50 ≈ 60–100ms, p95 ≈ 150–200ms @7k | moderate (pipeline.py:865–925 reorder+gate); validate any@10 on the 990q split — dense is needed for the 38% paraphrase gap, so keep it as the always-available expansion, just batched |
| 5 | **FTS5-native scoring** for lex (`bm25(unit_fts)` or FTS-prefilter+Python-rescore) | lex 143\u2192~20–40ms | scoring-form change — dev-split quality check required |
| 6 | **Kill statement churn** — cache `has_table` per conn (sqlite_master ~2.3ms × ~114/q), page/batch the `unit_fts_docsize`/`content` internal probes (~8k/q), stop facet re-runs of expensive lanes (`max_facets` divides the slice — dense pays the whole cost twice today) | \u221230–60ms/query + removes dense×2 | trivial guards |
| 7 | **100k scaling** — repack blocks ~512 rows/block (~200× fewer fetches) or ANN + eligible-set post-filter; eligibility \u2192 bitset/generation set | keeps p95 ≈150ms at 100k under the gate | larger: encoder-manifest/index change |

Bottom line: this is **not** a \"lanes are slow\" problem — it's one statement-count pathology (dense per-block eligibility, ~63% of all CPU) plus redundant whole-scope rescans. Fixes 1–3 get p95 ≈ 250–350ms at 7k with a ~15-line footprint; adding the staged gate (4) hits ≤150ms without touching scoring semantics; (5)–(7) buy the 100k headroom.",
  "timestamp": "2026-09-23T18:31:04.223000Z"
}