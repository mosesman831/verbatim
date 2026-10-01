# Verbatim write-path ingest profile — V8-13.05 / W5 write arms

Setup: verbatim @ `1c86f4f`, `VERBATIM_EVAL_LOCOMO=1`, LoCoMo-10 corpus. Python 3.11, 8-core box, WAL mode. Every number below is measured, `sqlite3.set_trace_callback` statement attribution + monotonic-clock spans; ~5-10% profiler overhead on all runs (relative comparisons unaffected). Two regimes:

- **Fresh engine** — empty store, conv-47, 689 items.
- **Populated store** — copy of the rebuilt dev-split store (2,877 sources / 4,143 units / 145,073 graph_edges / 7,122 claims / **14,813 open reviews**), then conv-41 (663 items) on top. This is the regime that matters: per-item cost grows with scope state — early→late quartiles ramp 634→847ms/doc within one conversation.

Reference baseline (full dev-split rebuild, 2,877 items): 929,275ms = **323ms/doc end-to-end** (includes add+drain+settle), consistent with the populated per-doc band below.

## Task 1 — per-stage profile (exclusive ms/doc, populated conv-41; wall 762.6ms/doc, ~2,930 stmts/doc, ~19 commits/doc)

| ms/doc | n/doc | stmt/doc | stage | where |
|---:|---:|---:|---|---|
| 450.3 | 2.77 | 212.9 | **admit.prop_unstructured** | `ingest.py` ADMIT → `relations.py:208` `propose_unstructured_relations` |
| 47.8 | 1.05 | 0.0 | tx.checkpoint (passive) | `ingest.py:1240-1249,1280` — every 8 jobs + per drain pass; ~45ms each on populated WAL |
| 37.8 | 1.00 | 19.5 | v7.aliases | `units_jobs.py:1295` `_write_aliases` reloads ALL `entity_canon` for the scope per job |
| 32.4 | 2.77 | 78.4 | admit.apply | claim revision application per admit |
| 30.9 | 1.00 | 1.0 | edges.total (pair-merge loop) | `graph_jobs.py:675` `build_edges` |
| 21.0 | 1.00 | 18.1 | sp.prescan | scope-rows prescan per SOURCE_PROJECT |
| 20.1 | 1.00 | 62.5 | add.capture_tx | envelope persist tx (`facade.py` add path) |
| 13.6 | 19.3 | 19.3 | tx.commit | ~0.7ms/commit avg — commits are NOT the bottleneck |
| 13.0 | 5.77 | 49.0 | job.commit_effects | per-job effects-commit wrapper |
| 12.6 | 2.00 | 44.0 | sp.decl_oblig | declared-obligation rows per project |
| 9.1 | 1.00 | 5.5 | se.commit_vectors | `source_jobs.py:1897-2007` `_commit_unit_vectors` — full `block_row_keys` reload per job |
| 8.4 | 1.00 | 2.0 | cons.closure_sweep | `units_jobs.py:405` `consolidate_scope_v7` — scans all observations_v7+profiles_v7 per job |
| 6.7 | 2.00 | 3.5 | edges.load_units | co-canon context hydration (graph_jobs.py) |
| 5.3 | 1.00 | 13.0 | sp.dedup | |
| 1.8 | 3.77 | 55.9 | job.settle_receipts | `ingest.py:666` `_settle_receipt_stages` — per-item eligibility/session-barrier receipts |
| 1.6 | 7.77 | 59.9 | queue.lease | `queue.py:451-560` `ORDER BY lane,… LIMIT` per job — O(pending queue depth) |
| 0.9 | 1.00 | 70.1 | v7.ensure_schema | `units_jobs.py:141` `ensure_v7_additive` ≈70 CREATE-IF-NOT-EXISTS + sqlite_master scans **per job** |

Job mix per doc: ~5.7 dispatched jobs (1× SOURCE_PROJECT, 1× SOURCE_EMBED, ~1× HARVEST, ~2.77× ADMIT + completions). Fresh-engine contrast (conv-47, 123.8ms/doc): drain.report 112.8, job.ADMIT 71.2, SOURCE_PROJECT 26.9, edges 6.4, checkpoint 4.6; ramp 59→193 within one conversation — the growth term is visible even fresh.

## Task 2 — batch-transaction arm

**Not the lever.** ~19 fenced write txs/doc but commits run 0.26ms fresh / 0.7ms populated (tx.commit = 13.6ms/doc total). `synchronous=FULL` (`store.py:400,598`) + WAL already; fsync is cheap on this box — **the 2,930 statements/doc dominate, not fsync**. Measured: `bulk_add` chunking (facade.py:1829-1949 — one `store.tx()` per ≤512 chunk, SAVEPOINT per item) lands within noise of per-item adds: bulk64 121.4 / bulk512 122.2 fresh; bulk256 populated 691.7 vs e2e 762.6 baseline — all within the ~1-9% range. The tx boundaries worth cutting are the **per-JOB** ones: 5.7 jobs × (lease+commit_effects+settle+fence) ≈ ~30ms/doc of pure machinery.

## Task 3 — edge-build arm

145K `graph_edges` incrementally, per SOURCE_PROJECT job:

- Insert itself is cheap: `edges.insert` = 1.2ms/doc at 57.2 stmts/doc populated — `_insert_edges` (`graph_jobs.py:629-648`) is a per-edge `INSERT OR REPLACE` loop; an `executemany` arm saved ~2ms/doc (noise). **Not worth shipping for speed.**
- The real cost is **context reload**: `_canon_map` (`graph_jobs.py:458-496`) pulls every `entity_mentions` row for all canons shared by the new units, `_session_mates` loads whole sessions, `_load_units` hydrates every co-canon unit — **~843 ctx units/job at this scale**, 12.9ms/doc of loads; per-job O(scope_units) ⇒ O(N²) over an ingest.
- `co_mention` emission is banded (`CO_MENTION_NEIGHBORS_V1`, graph_jobs.py:772-800) — O(df×neighbors), not pair-quadratic; the quadratic lives in the canon-membership context load that feeds it. `skipcomention` at write: **−6.5ms/doc** populated *and* collapses ctx loads 843→41 units/job. Given a1's "net-harmful at query" verdict — **ship it at write**; it also shrinks every later co-canon load.
- `session_fam` pair scan is O(session_members²) per job but sessions are ~70 units → ~0.1ms/doc.

## Task 4 — dense-embed arm

Claims land `pending` (review-gated) → EMBED jobs never fire (`ingest.py:1951` gates on `Lifecycle.ACTIVE`): `job.EMBED n=0`, `embeddings` empty. All dense cost is **SOURCE_EMBED** (hashing encoder over payloads): `se.body` 15.6ms/doc, of which `se.commit_vectors` 9.1ms — `_commit_unit_vectors` (`source_jobs.py:1897-2007`) does a `block_row_keys` SELECT reloading all `rowmap_blobs` for (encoder, scope, generation) **per job** plus `write_oracle_row` per unit — the matrix-repack quadratic a3 flagged. `batchvec` arm (memoize keyset per drain, coalesce writes): folded into composite4; isolated delta ≈ −8-10ms/doc.

## Task 5 — index/incremental arm

- `PRAGMA synchronous=FULL` at `store.py:400,598`; `journal_mode=WAL`; `foreign_keys=ON`. `syncnormal` arm: −4ms/doc — small because commits are already cheap; include only if the checkpoint discipline tolerates NORMAL.
- `ensure_v7_additive` (`schema_v7.py:886-906`, fired per job `units_jobs.py:141`): ~70 stmts/doc of CREATE-IF-NOT-EXISTS — gate once per drain/process.
- `v7.aliases` (37.8ms/doc — **#2 stage after indexrel**): `_write_aliases` reloads all `entity_canon` for the scope every SOURCE_PROJECT job; needs an incremental canon-delta pass (O(new canons), not O(scope)).
- `update_stats`/`corpus_stats` (`units_jobs.py:380-382`): ~49 stmts/doc — batch per conversation.
- Receipts: `job.settle_receipts` 3.77 calls × 55.9 stmts/doc — batch to one settle per drain (~−3ms/doc).
- `v7.analyze` per project ~0.8ms/doc — fine as-is.

## Task 6 — idle-job drain arm

The eval already drains ONCE per conversation (`arms.py:690-774`: per-item `add` enqueues only, then one `drain_memory` + `wait_ready` ≤2s). The drain itself holds ~729ms/doc of the 763 wall. Inside it:

- `tx.checkpoint` 47.8ms/doc: `checkpoint_passive` every 8 jobs (`ingest.py:1240-1249`) + per pass end (1280); each ~45ms on the populated WAL. `drain-every 64` arm: **−33.5ms/doc → 729.1**. Ship: cadence by WAL bytes (~4-8MB) not fixed job count.
- `queue.lease` O(pending): 1.6→4.7ms/job as the backlog deepens in drain-once mode — `LIMIT k` batch lease fixes the second-matching latency too (quality/latency note requested).
- e2e vs perdoc totals identical — queue-depth growth offsets checkpoint savings, consistent.

## The populated-scale hog — `admit.prop_unstructured`, 450ms/doc (59% of wall)

Decomposition of `propose_unstructured_relations` (`relations.py:208-317`): candidate scan (LIMIT 32, `:83`), `_claim_text` per candidate (full `source_revisions.payload` + HMAC, `supersession.py:265-300`), then per surviving pair (`_PAIR_BUDGET=4`, `:82`) `EdgesRepo.add` + `ReviewsRepo.create(dedup=True)` inside `store.tx()`.

- **`ReviewsRepo.find_open`** (`repos.py:1307-1345`, reached via `create(dedup=True)` :1363): `SELECT review_id, proposed_effect_json FROM reviews WHERE scope_id=? AND state='open'` — **fetches + `json.loads`-parses every open review in the scope** (~15-19K rows at this scale) to compare subject claim-ids in Python. Open reviews are never resolved during ingest → scan grows linearly → the whole ADMIT proposal path is O(open_reviews) per pair = the primary quadratic. ~2.4 creates/doc × ~15-19K row materializations+JSON parses ≈ the 450ms/doc (consistent: `skiprel` ceiling 306.6; warm isolated call only 7.6ms — pure find_open cost).
- `cachetext` arm (memoize `_claim_text`): stmts fell 213→122/doc, **wall unchanged (769.2)** — the cost is row materialization + JSON parsing in find_open, not statement count.
- **Fix arm `indexrel`**: `find_open` pre-filter `AND instr(proposed_effect_json, ?) > 0` for every subject id (any equivalent review must contain all subject ids as substrings → zero false negatives; the exact Python compare still runs on survivors). 100-item slice 279.6; **full conv-41: 326.0ms/doc (−436.6, −57%)** — within 20ms of the skip-everything ceiling (306.6) with semantics intact. Even better for prod: materialize `(scope_id, state, subject_key)` dedup columns and index them.

## Per-arm results

Fresh (conv-47, 689 items; base 123.8ms/doc):

| arm | ms/doc | Δ |
|---|---:|---:|
| composite (edgesbatch+skipco+dedupoff+defercons+syncnormal) | 115.1 | −8.7 |
| de64 | 116.9 | −6.9 |
| syncnormal | 120.0 | −3.8 |
| skipcomention | 119.6 | −4.2 |
| bulk64 | 121.4 | −2.4 |
| bulk512 | 122.2 | −1.6 |
| e2e | 123.7 | −0.1 |
| dedupoff | 125.6 | +1.8 (noise) |
| defercons | 126.5 | +2.7 (noise) |
| edgesbatch | 134.2 | +10.4 (noise) |

Populated (conv-41, 663 items; base 762.6/765.9 two runs):

| arm | ms/doc | Δ vs 762.6 |
|---|---:|---:|
| skiprel — skip prop_unstructured entirely (ceiling probe) | 306.6 | −456.0 |
| **indexrel — find_open instr() subject pre-filter** | **326.0** | **−436.6** |
| **composite4 — indexrel+skipco+dedupoff+defercons+syncnormal+batchvec, drain-every-64 (e2e)** | **270.5** | **−492.1** |
| **composite4b — same arms via bulk_add chunk 256** | **269.9** | **−492.7** |
| de64 | 729.1 | −33.5 |
| composite (old arms, no indexrel) | 690.8 | −71.8 |
| composite2 +cachetext | 689.1 | −73.5 |
| composite2b bulk256 | 687.3 | −75.3 |
| composite_b bulk256 old-arms | 691.7 | −70.9 |
| skipco | 756.1 | −6.5 |
| cachetext | 769.2 | +6.6 (no help) |

batchvec/composite3 solo-arm runs still draining at report time (marginal — batchvec's effect is inside composite4).

## Verdicts

**Ship set** (ordered by measured delta, populated):

1. **indexrel** — dedup-index `ReviewsRepo.find_open` (instr pre-filter now; materialized `(scope,state,subject_key)` columns later): **−437ms/doc**, semantics preserved (exact compare still runs on pre-filtered rows). The must-ship fix — prop_unstructured goes 450→~30ms/doc.
2. **Checkpoint cadence by WAL bytes** (not every-8-jobs): −33.5ms/doc.
3. **skipcomention** at write: −6.5ms/doc + kills the 843-unit ctx reload (a1 already flags it net-harmful at query).
4. **batchvec** — memoize `block_row_keys` per drain / coalesce `write_oracle_row`: ~−9ms/doc.
5. **defercons** — `consolidate_scope_v7` once per drain instead of per job: ~−10-15ms/doc; profile/observation materialization deferred to drain end (no semantics change mid-drain).
6. Minor: once-per-drain `ensure_v7_additive`/`update_stats` (~−5-8), batched `settle_receipts` (~−3), `syncnormal` (−4, decision needed).
7. **Flagged, unarmed**: incremental `_write_aliases` canon-delta — 37.8ms/doc is now the #2 residual; and `admit.rel_write` (35.6ms/doc — edges.add dedup SELECT + reviews insert per pair) can batch pair writes per admit.

**Expected composite**: **270ms/doc measured** (composite4 e2e / 269.9 bulk) vs the 150ms target → **~1.8× over**. With the flagged-but-unarmed fixes (aliases −35, rel_write batching −20, prescan −15, decl_oblig −8) the modeled floor is ~**190ms/doc** — still over.

**Irreducible stage**: the **per-item job fan-out** — 5.7 jobs/doc, each carrying lease + fenced tx + commit_effects + settle (~30ms/doc of machinery at any scale) wrapped around bodies that are themselves per-item O(scope) scans (`sp.prescan` 19, `v7.aliases` 38, edges ctx 29, `cons.closure_sweep` 8, `admit.apply` 30, `rel_write` 36). No single stage dominates post-indexrel — the wall is now a flat ~10-40ms spread across ~15 stages. Getting under 150 requires **conversation-scoped job fusion**: one SOURCE_PROJECT over N items (one canon/ctx load, one prescan, one consolidate), one ADMIT pass over the item batch, one vector-commit, one settle — i.e. the drain must see the conversation as the unit of work.

**Task 7 / AMB granularity**: `Memory.bulk_add(list[Document], chunk_size≤512)` (facade.py:1829-1949) already exists, keeps per-item SAVEPOINT isolation inside one tx per chunk, and measured equal to per-item adds (composite4b 269.9 ≈ e2e 270.5) — the API granularity verbatim needs is one-shot conversation lists through bulk_add; what it lacks is the drain treating that batch as one job unit. Fresh-scale composite ≈ 115-120ms/doc is already under target — the gap is purely populated-state scan amplification, so the fusion work should target the scope-scan functions listed above.

**Second-matching latency & quality limits** (requested): `queue.lease`'s per-job `ORDER BY lane,… LIMIT 1` scales with pending depth — 2.3→4.7ms/job when draining a 663-item backlog; a `LIMIT k` batch lease amortizes it (also improves lease latency under big drains generally). Quality side: `indexrel` is exact (substring superset then full compare — no false negatives); `skipco` is safe only if a1's net-harmful verdict holds; `defercons` defers materialization to drain end — mid-ingest readers see pre-consolidated state; `dedupoff`/`edgesbatch` measured as noise and are NOT in the ship set.
