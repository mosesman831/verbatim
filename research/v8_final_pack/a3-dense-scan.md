# V8-08 dense-scan speed — measured findings

Repo `mosesman831/verbatim` @ `1c86f4f` (read-only; all prototypes in `/tmp` copies, nothing committed). Env: pyenv Python 3.12.13, numpy **2.5.3** (auto-probe engaged — see below), cryptography 50.0.1, `VERBATIM_EVAL_LOCOMO=1`. Store: `/tmp/v7w/mem.db` rebuilt by `eval.v7.track_r --dataset locomo --split dev --arms verbatim`; probes ran against a WAL-checkpointed `VACUUM INTO` snapshot (`/tmp/v7w-base/mem.db`) taken after all 2,877 `source_embed` jobs landed (the remaining ~424 queued jobs were admit/review jobs that don't touch `unit_vectors_block`). The `track_r` driver itself was still writing its report at probe time; the vector space was complete and frozen.

Store shape (snapshot): one block space `enc=hashing:subword-ngram:v1:hdr:v1, scope=ns_3b837db217b54467bd20d6004db0813d, gen=1` — **2,877 blocks / 4,143 rows / 4,143 units / 2,877 sources**, all `quant='f32'`, mean 1.44 rows/block (histogram: 2,251 sources→1 unit, 612→3, 14→4).

## Task 3 — numpy engagement + where the ~86ms goes

numpy **is** engaged: `verbatim.embeddings.matrix._NP = numpy 2.5.3` (auto-probe path); `stats.numpy=True` on every measured scan; rescore never fired (`rescore="none_needed"` — all-f32 space, `INT8_ROW_THRESHOLD` not crossed). The problem is not "numpy missing" — it is **one tiny matmul per block across 2,877 blocks**.

Raw `matrix.scan` over all 990 dev queries (k=64, deadline 2s):

| metric | value |
|---|---|
| scan p50 / p95 / mean | **75.57 / 96.64 / 79.56 ms** |
| blocks_scanned | 2,848,230 = 990 × 2,877 (every query walks every block) |
| dims-mismatch / skipped-quant / dup keys | 0 / 0 / 0 |

Lane-level (n=990, `lane_dense` incl. encode+scan+`_coverage_lag`+emission, excludes ctx-build):

| metric | baseline | 
|---|---|
| lane_dense p50 / p95 | **91.77 / 104.75 ms** |
| ctx-build (eligible+qv+policy) p50 | 24.52 ms |
| `eligibility_evaluated` | 0 — set-mode `_KeyEligibility` engaged (V7 eligibility fix verified live) |

cProfile (n=40 queries, 6.79s total): `matrix.scan` cumtime = 5.69s (**84%** of lane time), `_apply_lag`/`_coverage_lag` = 0.82s (**~12%** — it re-decodes every rowmap via `block_row_keys` + correlated live-units CTE after the scan already decoded them), encode ≈1%. Inside scan, the cost is per-block Python iteration, not math — per 40 queries:

| callee | calls | cumtime |
|---|---|---|
| `_unpack_rowmap` | 230,160 | 0.634s |
| `_block_rows` gen (SQL row iter) | 115,120 | 0.321s |
| np `.sum`/`.reduce`/`.asarray`/`frombuffer`/`astype`/`arange` | ~690K | ~1.15s |
| `_decode_scales`+`_decode_f32`+`errstate` | ~345K | ~0.35s |
| `_eligible_keys` (set filter) | 115,080 | 0.207s |
| `_heap_push` | 165,720 | 0.123s |

i.e. ~86ms ≈ per-block loop overhead (~85%) + coverage-lag tail (~12%) + real FLOPs (<5%).

## Task 1 — REPACK prototype [measured]

Repack (`/tmp/repack.py`): copy store → `wal_checkpoint(TRUNCATE)` → per (enc,scope,gen): `_block_rows`→unpack rowmap→slice `data_blob` rows → sort by unit_id → dedup → `DELETE` old blocks → `write_block` contiguous chunks at fresh `block_no` 0..n in **one tx** → VACUUM. f32-only (raises on int8). Result: 2,877 blocks → **9** (512) / **5** (1024) / **1** (8192), 0 dup keys, **~460ms** wall each.

Scan + lane after repack (same 990 queries, same eligibility, k=64):

| bound | blocks | scan p50 | scan p95 | lane p50 | lane p95 |
|---|---|---|---|---|---|
| baseline | 2,877 | 75.57ms | 96.64ms | 91.77ms | 104.75ms |
| **512** | **9** | **7.17ms** | **8.42ms** | **19.93ms** | **23.62ms** |
| 1024 | 5 | 9.33ms | 10.33ms | — | — |
| 8192 | 1 | 11.64ms | 23.57ms | — | — |

- numpy engagement post-repack: **one matmul per block = 9 matmuls/query** of ~460×384 rows (`blocks_scanned` 8,910 = 990×9, `numpy=True`). Target ≤20ms lane p50 **met at bound=512** (19.93ms).
- **Top-64 candidate identity: bit-identical at every bound — 990/990 tasks, order AND set.** (Same f32 bytes, `fsum` norms, `(-score,key)` order-independent sort.) No quality regression.
- 512 beats larger bounds: at 8192 the whole space is one 6.4MB-f32 blob → `astype(f64)` makes a 12.7MB copy per query (bandwidth-bound); 512-row chunks fit cache and keep `argpartition` small.

## Task 2 — Coalesced-write analysis [measured + model]

Today `source_jobs._commit_unit_vectors` writes **one block per embed job** (chunks at `MAX_BLOCK_ROWS=8192`, far above the 1–4 units a job carries): 2,877 jobs → 2,877 blocks. Two per-job costs scale with total rows: `block_row_keys()` present-check decodes **every rowmap in the space** — measured **8.4ms** on the 2,877-block end-state vs 4.2ms on the 9-block store (per-key decode bound); called once per job ⇒ ~O(N²) across ingest, ≈**12s** of rowmap decode over this ingest — plus `SUM/MAX` block query, `write_block` INSERT, oracle rows.

A **512-row batch bound** (coalesce: append into the last open block until ≥512 rows, then seal) would have produced **9 blocks** on this ingest (4,143/512) — the same shape the repack yields. Write-path change is small and local to `_commit_unit_vectors`: instead of unconditional `block_no=MAX+1`, read the newest block when `n_rows < BOUND`, concat `(key,blob)` items into it, and `UPDATE` its three blobs in place (≤bound rows rewritten, O(bound) not O(N)); when full, allocate `block_no+1`. Present-check unchanged semantically (still `block_row_keys` or, better, the oracle table — cheaper: `SELECT unit_key FROM unit_vectors_oracle`). Deferred-repack variant: keep per-job blocks, repack in the admit/consolidate job — same 460ms one-tx cost, zero write-path risk. Either way the scan sees ~320× fewer blocks.

## Task 4 — Alternatives vs brute force [measured micro-benchmarks]

Simulated repacked inner loop on this VM (scipy-openblas, Haswell): blob→`frombuffer`→matvec→norm→`argpartition`, one consolidated block per 512 rows, all-eligible (best case for the index-less path):

| rows N | current int8→f64 path | f32-native matvec |
|---|---|---|
| 4,143 | 2.87ms | 0.23ms |
| 10K | 6.97ms | 0.57ms |
| 50K | 34.37ms | 2.80ms |
| 100K | 70.19ms | 5.60ms |
| 200K | 146.35ms | 16.82ms |
| 500K | 359.71ms | 58.85ms |

Findings: (a) the `astype(np.float64)` inside `scan()` is bandwidth-bound — ~12–35× the f32-native cost; switching the q-vector to f32 (or storing f32 and scoring in f32 — numerically acceptable since blocks are already f32/int8) pushes the 20ms line from ~30K rows to **~230K rows**. (b) **sqlite-vec is not an index** — its `vec0` KNN is C-level brute force; it buys nothing the repacked numpy scan doesn't already do, and loses the eligible-set filter it would need as a post-filter. (c) A real ANN (hnswlib-style) queries ~1–3ms at 100K–1M rows but must post-filter against the eligible set — at low eligible fractions the over-fetch factor (k × 1/frac) eats the gain; it also adds index storage, rebuild-on-repack coupling, and recall <1.0 vs our exact scan.

**Honest crossover:** with the current f64 path an index wins ≥~30–50K rows; with an f32-native scan brute force wins until **~200K+ rows** (500K×384 int8 ≈ 190MB storage also pressures RAM/blob fetch). At this store's 4.1K–100K range, contiguous-block brute force dominates; int8 storage + f32 scoring would extend that to ~500K.

## Task 5 — Dense-slot rescue value, N_d ∈ {5,10,20} [measured]

Full dev gold set, live `run_search` (all lanes), gold = `evidence_ids` turn refs mapped via `source_revisions.payload`↔`item_document` (2,877/2,877 mapped; byte-identical dedup collisions kept per arm convention):

- **777 answerable** tasks (213 adversarial unanswerable excluded); **137 item-misses** (delivered pack contained no gold source).
- Of those misses, gold present in **dense** top-N: 3 @ N≤10, 6 @ N=20.
- Gold **absent from every other lane's candidate set** in those cases: **0** at N=5, **0** at N=10, **0** at N=20.

i.e. whenever dense top-N happens to contain the missed gold, at least one other lane already surfaced it in its candidate pool — dense's **unique rescue value on this corpus is exactly zero**. (Sparse/graph lanes carry the recall; dense is fully redundant at candidate level.)

## Verdicts

1. **Repack benefit — measured, large, correctness-safe.** 512-row repack: scan p50 75.6→**7.2ms (10.5×)**, lane p50 91.8→**19.9ms** (meets ≤20ms), top-64 sets bit-identical 990/990. Repack itself is a ~460ms one-tx operation — cheap enough to run in a maintenance job. Remaining lane time is now dominated by `_coverage_lag` + ctx-build (~25ms) — next target after repack.
2. **MAX_BLOCK_ROWS / batch bound: use ~512 rows as the effective block bound.** It is optimal on the measured curve (512: 7.2ms < 1024: 9.3ms < 8192: 11.6ms), keeps `astype(f64)` copies cache-resident, and bounds worst-case block reads. Implement via coalesced write (append-to-open-block in `_commit_unit_vectors`) — drops this ingest from 2,877 blocks+INSERTs to ~9, removes the O(N²) `block_row_keys` decode, and keeps `MAX_BLOCK_ROWS=8192` as the hard ceiling while 512 becomes the fill target. Repack stays as the one-time migration + periodic maintenance.
3. **Dense-slot N_d recommendation: N_d = 0 — do not fund a dense-slot arm (V8-08.04).** Unique rescue = 0/137 misses at N=5/10/20 on the dev gold set; every dense-gold hit is already covered by another lane's candidates. If a dense slot is still wanted for insurance, the minimal N_d=5 is justified (all rescuable gold appeared by rank 10; ranks 10–20 added nothing unique). Spend the latency recovered by repack on `_coverage_lag`/ctx-build instead; revisit an ANN index only when the space exceeds ~200K rows (and then only after switching scoring to f32-native, which alone moves the brute-force viability ceiling ~7×).
