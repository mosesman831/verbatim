# lane_miss forensic diagnosis — locomo dev (2026-09-23)

Measured on the persisted track_r store (`/tmp/v7-trackr-ezeh8hhn/mem.db`,
dev split: 4 convs, 2877 items, 990 tasks; verbatim ingest via real
`Memory.add`, infer=True, encoder=hashing, worker=external).

## Headline numbers (track_r v7-a, unpinned)

| arm | any@10 item | any@10 session | p50ms | p95ms |
|---|---|---|---|---|
| verbatim | 0.195 | 0.512 | 382 | 613 |
| flat_bm25 | 0.587 | 0.884 | 3.1 | 5.1 |
| fts5 | 0.578 | 0.876 | 5.0 | 7.9 |

verbatim attribution: delivered 222, **lane_miss 444**, rank_shift 100,
packed_out 11, unsupported 213 (cat-5 adversarial gold withheld).

## Mechanism decomposition (own measurements)

1. **~62% of lane_miss is pool-cap + deadline truncation, not recall.**
   Re-ran first 120 lane_miss tasks with `POOLS[MID].lane_cap` 200→800 and
   timeout 500→8000ms: **75/120 (62.5%) now surface gold in fused top-64;
   64/120 (53%) in top-20.** The fused ranking works when lanes aren't
   truncated. Lane cap is 200 (POOLS MID) while common-entity terms have
   df≈700 ('jon'=699, 'gina'=667 postings) — gold units die below the cap
   before fusion. 'destress' df=3 floods top-200 with 3 strong
   false-positives.

2. **Deadline starvation:** default `search` timeout_ms=500 while pipeline
   p50≈382ms and reaches ~1.8s with deadline lifted — the 500ms budget
   cuts lanes mid-scan (dense reports "partial"). Slow lanes lose
   candidates silently. Latency is the second-order miss cause.

3. **4/8 routed lanes emit zero candidates on real dialogue** (measured
   on the destress query): fuzzy, temporal, graph, (obs). Root cause:
   projected structures are EMPTY — `observations_v7`=0, `t2_facts`=0,
   `facts_fts`=0, `graph_edges`=0, `entities`=0, `observations`=0. What
   ingest DID produce: `units`=7020, `unit_vectors`=7020, `edges`=7690,
   `entity_canon`=1513, `state_facts`=156, `valid_intervals`=7122. The
   v7 semantic/observation/typed/graph lanes starve on raw-turn corpora
   because the write path doesn't materialize their inputs.
   (Note: `graph` lane reads `edges` — non-empty — but still returned 0;
   its seeds = top-10 eligible of ent∪lex, so seed-gated.)

4. **Unit duplication:** every source turn mints BOTH a `session` unit
   and a `turn` unit (2877 each) + sentence_window units (1266). Session
   units carry the same turn text — doubles df/tf mass and wastes lane
   cap slots. Any unit-kind × gold-granularity interaction needs audit.

5. **True recall gap ≈ 38% of lane_miss (45/120).** Query-term coverage
   in gold: misses where bm25 also failed = 0.152 vs 0.40 where bm25 hit.
   The residual is paraphrase bridging ('destress' ↔ 'stress relief',
   'mentorship' ↔ 'got mentored') + multi-turn join evidence — the
   hashing-embed lane can't bridge paraphrase; obs/facts lanes that
   would hold distilled paraphrastic facts are empty.

## Failure-class inventory (fix order)

A. Pool arena: per-lane cap 200 vs shared/global candidate arena —
   flat_bm25 ranks the whole corpus and wins precisely there.
B. Latency: pipeline 382ms–1.8s vs the 500ms timeout; needs staged
   retrieval + cheap-lane-first ordering to fit p95≤150ms.
C. Empty projections: obs/typed/facts/graph never materialize on raw
   turns — write-path gap, not a ranking gap.
D. Paraphrase bridging without LLM: the residual ~38%.
E. Unit granularity audit: session+turn duplication.
F. Cat-5 adversarial (213 withheld gold = premise turns): needs premise
   mode, not more recall.

## Appendix B — factorial correction: deadline dominates cap (measured post-writeup)

Same 120 misses, `Memory.search(limit=64)` with patched `POOLS[MID]`:

| config | gold in fused top-64 | p50 lat |
|---|---|---|
| cap 200 / timeout 500ms (as-run) | 15/120 | 322ms |
| cap 200 / timeout 8000ms | **72/120** | 1758ms |
| cap 800 / timeout 500ms | **10/120** (worse) | 375ms |
| cap 800 / timeout 8000ms | 75/120 | ~1.6s |

**Deadline is the primary killer, not the pool cap.** Enlarging the cap
*without* the deadline makes things worse — lanes nominate bigger pools
and fewer lane outputs land inside 500ms. With an 8s deadline the
default cap already surfaces 72/120 gold into the fused top-64 (the
earlier 62% estimate conflated the two knobs). Cost of the recall:
~1.7s p50 — so the fix is either a quality-profile deadline (~2s) or
making lanes finish inside 500ms, NOT blindly raising lane_cap.

## Appendix C — why the deadline bites: slice budget vs real lane cost

Mechanism traced through the code (verbatim/retrieval/v7/):

1. `Memory.search(timeout_ms=500)` → `run_search` → S2 computes every
   lane's `LaneSlice` **once at entry** via `deadline.allocate`
   (landed module — the equal-share fallback never fires).
2. Slices are proportional to `LANE_COSTS_V1` declared p95 contracts:
   lex 6ms, dense 6ms, graph 6ms, fuzzy 4ms, typed 4ms, time 3ms,
   ent/obs 2ms, default 2ms — Σ≈35ms total declared work.
   Under a 500ms request each lane's slice is floor(0.5)+headroom·cost/Σ,
   i.e. lex/dense/graph ≈ 85ms, typed ≈ 57ms, time ≈ 43ms, ent/obs ≈ 29ms.
3. Real work on this store (7020 units, df floods like 'jon' df=699):
   lane_lexical alone needs ~200ms+ — universe scan + eligibility +
   stats + posting fetch + per-doc bm25f rescore — so it expires mid-scan
   and returns `PARTIAL`/`DEADLINE` with few or zero candidates. The same
   hits every heavy lane; that is why instrumentation showed 4/8 lanes
   emitting 0 candidates. Lanes run **sequentially**; each lane's real
   wall-clock can exceed its slice only until its own slice fires, but
   post-S2 stages and later lanes still pay the request-deadline
   (`remaining_ms() <= 0` gate → `deadline_exhausted`).
4. `nominate_df_theta` — the policy knob that bars high-df flood terms
   from nomination — is **disarmed by default** (`_nomination_knobs`
   returns θ=None). Arming it (e.g. θ=0.5) is a config-level fix for the
   'jon'/'gina' floods that also shrinks lane work.
5. `track_r` latency vs attribution confirms the deadline is uniform
   across outcomes: lane_miss p50 376ms vs delivered p50 392ms; 134/444
   lane_miss queries exceeded 500ms outright (vs 50/222 delivered).

So the durable fix is not "raise the cap" or even "raise the timeout"
alone — it is making lane work actually fit inside slices (arm
df_theta, leaner universe scan, unit dedup) plus a larger budget for
the quality profile. The declared §04.2 costs are ~35ms of contract for
~1.5s of measured work on a 7k-unit corpus.
