# Beat-the-baseline plan — LoCoMo dev (measured, unresearched items marked)

## POST-FIX STATE (wave-1 landed, verified 2026-09-23)

Commit `1c86f4f` implemented the F0-F3 fix spec. Verified by an
independent 3-arm track_r rerun + 4 audit agents on the real LoCoMo
dev split:

| metric | pre-fix | post-fix | flat_bm25 |
|---|---|---|---|
| item any@10 | 0.195 | **0.546** | 0.587 |
| session any@10 | 0.512 | **0.852** | 0.884 |
| delivered | 222 | **527** | 507 |
| lane_miss | 444 | **104** | 1 |
| p50ms | 382 | 615 (@2s) / **325 (@500ms)** | 3.1 |
| SQL stmts/q | ~8,932 | **4,529** (−49%) | — |
| correct_refusal | ~0.042 | **0.211** | n/a |

- **verbatim now BEATS bm25 on cat-2 temporal (0.703/0.891 vs
  0.640/0.880) and cat-1 multi-hop (0.460/0.825 vs 0.413/0.786)**.
- **@500ms ≈ @2000ms recall** (0.547 vs 0.546 item — deadline is no
  longer the knob; product default stays 500ms, p1 measured).
- occurred NULL 100%→0%; units 7020→4143 (twins gone); when-field
  1 junk string→126 real dates; graph_edges 0→145,042; entity_canon
  1513→853 ('jon gina' fusions eliminated); lex/dense ok 990/990.
- Remaining: 104 lane_miss + 84 rank_shift (structural), cat-5 item
  gap (0.408 vs 0.577), premise-trigger FPR 17.5% on answerable
  (62/136 refusals had gold in pool — recall cost).

## WAVE-3 priorities (post-fix audits, ordered)

1. **Graph-lane N+1** — the live lane introduced its own: `unit_row`
   per visited node = 486 stmts/q, BFS needs ~300-840ms vs 37-148ms
   slice → deadline-partial 97/97. Batch node-row fetches + bound
   `_latest_edges` fetchall + finer deadline checks (~30-50 LoC).
2. **Premise-trigger precision** — fires 21% on both populations
   (near-zero discrimination; tracks thin coverage not
   unanswerability). Needs a presupposition-vs-fact signal: verify
   asked relation against structured state, or speaker-mismatch
   WITHIN the top group — not a coverage floor (p3 measured: no floor
   separates; delivered junk is *supported*-labeled near-miss).
3. **Write-time in-text temporal resolution** anchored to
   unit.occurred — resolves "yesterday/last week" in turn text at
   ingest → ~22/41 cat-2 residual misses (p2).
4. **recorded_fallback occurred-ordering** — still ingest-ordered
   4096-cap, truncates 96/105 cat-2 fallbacks (p2).
5. **claims-path event_us** — pass turn `when` into `_extract_valid`
   so relative_day/week anchors move off ingest-2026 (210 rows; p2).
6. **events-first loosening** — 59.5% NULL subjects + strict AND →
   lane contributes on 43/175 only (p2).
7. **150ms budget needs structural work**: block repack (2877
   micro-blocks→~512-row, numpy engages 86→~20ms), stem memoization,
   elig cross-query cache, slim post-pool → projected ~95-120ms (p4).
   Scheduling alone floors at ~245ms.
8. Facet path never fires (QueryViewV7.facets never populated →
   8 not 16 lane invocations) — dead code or wiring gap? (p4)
9. Session-ndcg normalization bug (values >1.0 — p2).

---

## ORIGINAL DIAGNOSIS (pre-fix, kept for provenance)

Goal: close verbatim's 0.195 → ≥0.587 item any@10 gap vs flat_bm25 on
LoCoMo dev (990q, 2877 items) without breaking the 150ms quality budget.
All numbers below are measured on this box (track_r 3-arm run +
factorial probes + sqlite forensics); diagnosis in
`eval/v7/lane_miss_diagnosis.md` (Appendices A–C).

## The gap decomposed (measured)

| attribution | n | share of 990 | class |
|---|---|---|---|
| lane_miss | 444 | 45% | gold never surfaced by any lane |
| delivered | 222 | 22% | correct delivery |
| rank_shift | 100 | 10% | surfaced, lost on rank |
| unsupported | 213 | 22% | cat-5 premise-mode withheld |
| packed_out | 11 | 1% | surfaced, dropped by pack cap |

lane_miss decomposes (120-sample + code trace):
- ~62% deadline-arena: 500ms request budget vs ~1.5s real lane work
  (Appendix B factorial: cap200/t8000 → 72/120 gold in fused top-64;
  cap800/t500 → 10/120, worse — bigger pools burn the deadline faster).
- ~38% true recall: paraphrase ('destress'↔'stress relief') + join
  evidence — obs/facts/graph write-path structures are EMPTY
  (observations_v7=0, t2_facts=0, facts_fts=0, graph_edges=0).
- Unit duplication: each turn mints a session+turn pair with identical
  text (2877+2877+1266 units on 2877 sources) — doubles df, wastes
  slice time and cap slots.

## TLDR — ordered fix list by measured leverage

1. **Fence `_universe` to kind='turn' (+windows)** — ONE line, no
   reindex: +15-24% gold-in-pool, −18% median lane latency, removes
   ~59% of scored docs that are non-turn duplicates (r5 measured).
2. **Populate `occurred` from `metadata.when` at write** (~20-40 LoC
   + reindex): repairs `when` FTS junk (df=7020 tokens), window scan,
   events occurred, claim valid anchoring, recorded_fallback order —
   ~5 defects share this one root (r8). +9-14 cat-2 tasks.
3. **Fix the eligibility N+1** — share resolved verdict/unit-row store
   across facets+lanes (request-scoped): p50 2332→577ms measured with
   the dedup patch; ~3600 SELECTs→~1 scan; then materialize the
   eligible set once per request (zero SQL) (3 verifications, r2).
4. **Timeout 500ms→~2s + LANE_COSTS recalibration + cheap-lanes-first**
   — deadline is the dominant recall knob (cap200/t8000: 72/120 gold
   in fused top-64 vs cap800/t500: 10/120); staged lanes →
   p50≈60-100/p95≈150-200ms@7k (r2).
5. **Pre-fetch df gating** — `_collect_postings` MATCHes every term
   (stopwords+date-junk) before nomination; gate on maintained lex_df /
   absolute cut ~df>1400 — removes the dominant slice burner (r2+r8:
   60/61 never-surfaced misses are FTS-reachable).
6. **Wire orphan write-path producers** — build_edges into
   project_units_v7 (~10 LoC), consolidate_scope_v7 post-projection
   (~20 LoC), require_review=False flag → lights graph/obs/facts lanes
   (r4).
7. **±1-turn same-session context propagation w≈0.7** — +0.115 any@10
   on flat-BM25 proxy, no re-index (r3); then potion-8M int8 dense
   +0.040 marginal → 0.800 any@10 proxy.
8. **Entity-joint lane** — em self-join HAVING COUNT(DISTINCT canon)≥2
   cuts 290→78 units at ~0.15ms; conjunctive retrieval neither Zep nor
   Hindsight has (r6). Plus `_cap_spans` `:`-merge bug fix.
9. **Cat-5 premise mode** — INSUFFICIENT status + still deliver
   premise items; speaker-authored-0-units check detects 75% at ~4%
   FPR → correct_refusal ~0.7-0.8 (r7).
10. **Facet slot reservation + per-canon quota** — facet candidates
    die in `merged[:capv]` today (r6); one df~700-1078 canon floods
    its own postings (r1).

Composite forecast (r3+r8+r5 sums, before interaction effects):
kind-fence + occurred + df-gating + timeout ≈ bm25 parity on cats
1-4 retrieval (~0.55-0.65 item any@10); ±1-turn ctx + dense + entity
joint + cat-5 mode ≈ parity+ on the residual categories.

## Fix levers, in order

### F0 — Budget/scheduling (unblocks the rest; config-only)
1. `nominate_df_theta` — CONTESTED (measured both ways): the df table
   shows floods are stopwords + **date-junk stamped into every unit**
   ('wednesday'/'2026-09-23' df=7020=100%, 'on'=6524=93%). θ≈0.2
   (df>1404) kills that band while keeping entity names (nate 1072,
   jon 699) — BUT θ fires on `n_eligible`, which shrinks on
   scope-restricted queries and would then bar those same entity terms
   (r1's warning; they recommend keeping it disarmed). Safer shape:
   pre-fetch gating on maintained `lex_df` with an *absolute* df cut
   (~df>1400 & content-only), or exempt entity-channel terms. Treat as
   a tuning experiment, not a default.
2. Timeout profile: 500ms → ~2s for the quality profile (measured
   72/120 gold reach fused top-64 at 8s; slices scale ×16 — every lane
   completes; r2 profiled the untruncated wall at p50=1,818ms).
   Cheap profile can keep 500ms.
3. LANE_COSTS_V1 recalibration + cheap-lanes-first staging (r2
   measured): declared costs (lex 6ms) are ~30× below measured;
   ent/fuzzy/time/lex → fuse probe → dense/graph only when union
   <4×limit or RRF margin low → r2-estimated p50≈60-100ms,
   p95≈150-200ms @7k. Also stop facet re-runs of expensive lanes
   (max_facets=2 divides slices — dense pays its whole cost twice).

### F1 — Lane speed (make work fit the slice)
0. **Per-key eligibility is re-evaluated per block × per facet — the
   single dominant cost** (three independent verifications: my cProfile
   `filter_keys`→`_fetch_unit_rows` = ~3600 calls/2.2s of 2.9s; my
   monkeypatched verdict-store dedup → **p50 2332→577ms**; r2's
   independent measure ~2,900 SELECTs/scan, dense 1,480→145ms).
   Mechanism: `_KeyEligibility` is constructed fresh per lane call,
   and `filter_keys` is invoked per block (~2.4 keys/call) →
   `_fetch_unit_rows` does `SELECT * FROM units ORDER BY` per call.
   Fix = share the resolved unit-row/verdict store across facets/lanes
   (request-scoped key), or batch `filter_keys(block_row_keys)` once
   per call and pass a set to `matrix.scan` — ~10-20 lines, zero
   semantic change. Then materialize the eligible unit set once per
   request (set-mode = zero SQL; −60-90ms more) + shared per-generation
   inventory cache (`_load`+`_universe`+`_coverage_lag` rescan units;
   −100-130ms). Engine is statement-bound: ~8,932 SQL/query p50 —
   also cache `has_table` (~114×2.3ms) and page the docsize probes.
4. **`lane_lexical` pays the flood at fetch time** (traced in code):
   `_collect_postings` runs `unit_fts MATCH` for EVERY term — query
   terms, identifiers, stopwords, date-junk — 1–2 FTS queries each
   returning up to ~6.5k rowids, then Python-set eligibility fences.
   `_select_nomination` runs AFTER collection, so dropping a term from
   nomination saves zero fetch cost. A 10-term query with 5 stopwords +
   a name fetches ~20–30k rowids → the ~85ms slice expires → PARTIAL/0.
   Fix: gate the MATCH fetch on maintained `lex_df` (already read into
   stats, unused for gating) or apply df_theta pre-fetch — small code
   change, removes the dominant slice burner. Date-junk ('wednesday' et
   al. at df=100%) additionally belongs to index-time meta-terms or the
   temporal lane, not bm25f postings.
5. Unit dedup — **fence lexical `_universe` to `kind='turn'` (+
   windows): ONE-LINE change, no reindex, r5 MEASURED +15-24%
   gold-in-pool (116→134 @cap200/t8000; 130→149 @cap800/t90ms) and
   −18% median/−27% p90 lane latency; ~59% of scored docs are
   non-turn duplicates.** Confirmed defect in code: `units_v7.py:1004`
   mints a session for ANY session_hint; lone-message carve-out
   (1044-1048) only fires for unlabeled messages → all 2877 sources
   mint byte-identical session twins (1-member aggregation over all 5
   FTS fields). Source fix: emit session units only for
   len(members)>1 (the episode dedup rule already in the docstring).
   Windows: index-only, deliver parent turn (`parent_unit_id` exists;
   windows never `_collapse` — substring signature). r1 adds: dup
   units double-vote RRF (~40% of lane cap); flattened unit-arena
   BM25 = 0.492 < flat 0.589 due to dup crowding — dedup BEFORE any
   arena flattening. And fix `when`-field ingest-date fallback
   (units_jobs.py:701-754): 5 tokens at df=7020 pollute every dated
   query's postings — root cause of the date-junk floods.

### F2 — Write path (the 38% paraphrase/structure gap) — r4 verdicts
6. Dead lanes are dead because producers are **orphans + gated**:
   `build_edges`/`consolidate_scope_v7`/`propose_facts`/`commit_facts`
   have ZERO production callers (live repro: calling them directly
   wrote 52 edges cleanly); facts_fts/entities/claim_entities gate on
   claim state ACTIVE while `admission.require_review=True` parks
   every harvested claim PENDING. Minimal deterministic write-path:
   - P0 wire `build_edges` at tail of `project_units_v7` (~10 lines) →
     lights graph lane
   - P0 trigger `consolidate_scope_v7` post-projection (~20 lines) →
     lights obs lane
   - P1 `require_review=False` or review-drain for the eval arm
     (1 flag) → restores facts_fts/entities/claim_entities
   - P1 t2-lite deterministic proposer reusing extractor pins →
     `verify_quotes`+`commit_facts` (~100 lines) → verified t2_facts
   - P2 suppress byte-identical session+turn units (~15 lines)
   - P2 widen deterministic SPO/time/quantity extractors
   - P2 wire EntitiesRepo into admit (~30 lines)
   NOT the cause: worker=external, infer flag, unit-kind filtering.
7. **±1-turn same-session document context = +0.115 any@10** — the
   dominant paraphrase lever (r3 measured on flat-BM25 proxy, 777
   answerable; raw 0.589→0.705; score-propagation w0.7 gives +0.079
   with NO re-index — wire `neighbor_window` from pack-time decoration
   into candidate scoring). Then **potion-base-8M int8 dense lane
   score-fused α≈1.2: +0.040 marginal** (stem+ctx+cos → 0.800 any@10
   vs flat 0.589; 7.6MB table, <1ms). Give dense top-10 an
   unconditional pool slot — zero-overlap golds die at fused-rank~47
   under pure score-sum. REJECTED by measurement: RM3/PRF (0/123
   gap recovery, `prf_round` flag is dead code), char n-gram
   (redundant after porter), WordNet (negative), potion-retrieval-32M
   (0.439 < 8M 0.564 on dialogue — update b9 recommendation).
   Residual 159 misses are multi-hop/temporal/premise — not paraphrase.

### F3 — Rank/allocation (post-surfacing)
8. rank_shift (100q) + arena sizing (r1 measured): cap=200 already
   admits 100% of flat_bm25's any@64 gold — admission isn't the gap,
   downstream is. Deep-tail curve: cap 200→79.2%, 1000→92.4%,
   2000→98.3% admission. Recommended POOLS: LOW(128,64,0),
   MID(1000,256,16), HIGH(3000,512,32) — gated on F0 deadline scaling
   first (bigger caps inside 500ms make things WORSE: 10/120).
   Entity lane needs a per-canon quota — one df~700-1078 canon floods
   its own 200-cap postings to 19-49% coverage.
8b. Entity-joint lane for cat-4 "both X and Y" (r6 measured):
    **em self-join `GROUP BY unit_id HAVING COUNT(DISTINCT canon)>=2`
    cuts union 290→78 units at ~0.15-0.24ms** (vs union 0.136ms) —
    conjunctive entity retrieval neither Zep nor Hindsight has.
    Implement as same-lane joint-units-first + `signals["joint"]`, or
    an `ent-joint` pseudo-lane. Cheaper variant: fielded FTS AND
    `entities:jon AND entities:gina` = 0.072ms. Speaker-pair probe
    `(speaker:jean OR speaker:john) AND entities:paris` = 0.062ms —
    split-evidence join for "both visited Paris". Reserve per-facet
    slots in the facet merge (`pipeline.py:894-906`): append+truncate
    starves facet candidates whenever whole-query output ≥ cap —
    verified real-analyzer facets get correct single-canon postings
    and die in `merged[:capv]` today. **Also a real write-path bug:
    `_cap_spans` merges across `:` minting `jon gina`-class junk
    canons that steal per-canon postings — add `:` to
    `_SENT_BOUNDARY`.** Defer materialized entity_pairs table.
9. Cat-5 premise mode (r7 measured): official LoCoMo scoring credits
   premise-turn evidence on retrieval recall — so correct cat-5
   behavior = **status INSUFFICIENT + still deliver premise items**
   (harness already double-scores correct_refusal AND any@k; pure
   abstention forfeits premise gold on 21.5% of corpus). Detection:
   75% of cat-5 (334/446) is one zero-SQL check — asked speaker
   authored zero supporting units in the group (speaker_canon +
   q_speaker already resolved; answerable queries have asked speaker
   among gold speakers 96% → ~4% FPR). Emit
   `premise_mismatch=speaker` as negative_evidence → INSUFFICIENT;
   expected correct_refusal 0.03-0.13 → ~0.7-0.8. Mem0/Hindsight skip
   cat-5 entirely — cats 1-4 is the de-facto headline convention.
10. Cat-2 temporal — root cause is `occurred` NEVER populated (r8
    measured): `metadata.when` carries '10:14 am on 9 May, 2023' for
    all 134 sessions but `_norm_occurred` reads only
    `occurred`/`occurred_*` keys → occurred NULL on 100% of 7020
    units → `_when_tokens` falls back to ingest-date → identical
    '2026-09-23 … wednesday' df=7020 tokens in EVERY `when` field
    (the date-junk floods) → events_v7 occurred NULL on all 1248 →
    claim valid_intervals anchor to ingest-2026 (212 rows) →
    `occurred_* IS NOT NULL` window scan vacuous → recorded_fallback
    scans 4096 ingest-ordered rows per no-window temporal query.
    FIX = `when`→`occurred_at` alias + "H:MM am/pm on D Month, YYYY"
    parser (~20-40 LoC + reindex): un-breaks `when` FTS tokens,
    window scan, events occurred, claim valid anchoring, and
    recorded_fallback ordering all at once. **+9-14 cat-2 tasks,
    highest leverage per LoC.** Then: query anchor precedence
    q_ts>session_ts>wall (fixes 4 wrongly-anchored 2026 windows:
    'visit in May'→May 2026, 'Cyberpunk 2077'→year 2077); loosen
    events-first emission (59.7% NULL subjects — OR-match + coref to
    speaker_canon); event-time extraction `resolve(text,
    anchor=occurred)` at write (103/120 misses' gold resolves; also
    enables the 40 year-answers = 23% of cat-2). Caveat r8
    established: only 26/175 cat-2 produce a parseable window — the
    temporal composite applies to ~15%, the gap is dominated by the
    occurred defect + posting floods; fixing both → ~0.55-0.65 item
    any@10 (bm25 parity) before the composite adds the rest.

## What to verify next (experiments)
- Re-run track_r with timeout_ms=2000 (arm change) → expected
  lane_miss drop ~60% at ~1.7s p50 — quantifies F0.2 exactly.
- Re-run with df_theta=0.2 armed (df>1404 cut) → measure lane latency
  delta + gold; and/or pre-fetch df gating patch.
- Index-time hygiene: drop date/meta tokens from unit_fts fields so
  'wednesday'-class terms never reach postings.
- Static-encoder swap (sentence-transformers potion-8M offline) →
  paraphrase coverage delta on the 45/120 residual.
