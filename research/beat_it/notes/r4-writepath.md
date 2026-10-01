# Write-path gap: producer map, starve mechanism, and the minimal extraction spec

**Scope.** Why `observations_v7` / `t2_facts` / `facts_fts` / `graph_edges` / `entities` are empty after real `Memory.add` of raw dialogue turns, which producers should populate them, and the minimum deterministic (no-LLM) write-path that would light the dead lanes. Read-only audit; no repo edits.

**Provenance discipline.** `[code]` = file:line in checkout `892f2b8` of `mosesman831/verbatim` (this session's clone — the stated paths `/workspace/verbatim-new` and `/tmp/v7-trackr-ezeh8hhn/mem.db` do not exist on this machine; they live on the parent session's box). `[measured]` = live sqlite census on a scratch repro store `/tmp/repro-mem/mem.db` built this session (8 synthetic dialogue turns via `Memory.add` + drain). `[parent]` = numbers from the parent session's diagnosis, not re-verified against their store. `[vendor]`/`[paper]` = external claims with URL.

---

## 1. The mechanism in one sentence

Every table the dead lanes read is written by a producer that either **(i) is never invoked by any production code path** (orphan producer — `build_edges`, `consolidate_scope_v7`, `propose_facts`/`commit_facts`, `write_supplied_edges`), or **(ii) is invoked but gated on claim lifecycle `ACTIVE`**, and every harvested claim lands `PENDING` under the default `admission.require_review=True` (`verbatim/config.py:29`) — so `facts_fts`, `entities`, `claim_entities` are unreachable through a bare `Memory.add` in this configuration. Worker mode (`external` vs `managed`) is **not** the cause: `source_project`/`harvest`/`admit` jobs enqueue synchronously inside the add transaction and `Ingester.drain_report` runs them all.

## 2. (a) Producer map — who SHOULD write each starved table

The only wired producer on the `Memory.add` path is `source_project` → `project_source_v7` → `project_units_v7` (`verbatim/jobs/units_jobs.py:115`), enqueued by `enqueue_source_jobs` (`verbatim/jobs/source_jobs.py:2744`, which enqueues ONLY `SOURCE_PROJECT`+`SOURCE_EMBED`, `:2826-2827`) from `Memory.add`'s commit path (`verbatim/memory/facade.py:1995` → `_source_jobs`). That one producer fills: `units`, `unit_fts_rows`/`unit_fts_content`/mirror FTS (`units_jobs.py:274-282`), `entity_mentions`+`entity_canon` (`:315`, turn-only gate `if u["kind"] != "turn": continue` at `:299`), `entity_aliases_v7`, `events_v7` (`_extract_events` at `:355` → `enrichment/events.py:1188 extract_events`), `state_facts`/`preferences` (`_write_prefs_state` at `:367` → `enrichment/prefs_state.py:2641`/`:2562`, gated on `_load_prefs_state` at `:210`), `lex_stats`. In parallel the `harvest`→`admit` chain (`envelopes.py:498 _enqueue_harvest` → `policy.py:731 _admit_apply`) writes `claims`/`claim_revisions`/`claim_evidence`.

| Starved table | Producer | Invocation mechanism | Status |
|---|---|---|---|
| `graph_edges` | `verbatim/jobs/graph_jobs.py:675 build_edges` (+`:651 write_supplied_edges`) | **None — zero production callers.** Not in `JobKind` (`core/types.py:192-222`); only tests call it. | ORPHAN |
| `observations_v7` | `verbatim/observations/consolidate_v7.py:1349 consolidate_scope_v7` | **None — zero production callers.** `JobKind.CONSOLIDATE` routes to v3 `observations/handlers.py:85 handle_consolidate` → `consolidate`/`consolidate_windowed` (writes v3 `observations`), never v7. | ORPHAN |
| `observations` (v3) | `observations/consolidate.py` via `handle_consolidate` | `JobKind.CONSOLIDATE` enqueued ONLY by `verbatim/refresh.py:963,981` (background lane, `jobs/queue.py:87`), called from `profiles/service.py:1773`, `repair.py:1102,1236`, `eval/v5/consolidation.py:204` — **never by the add path**. | UNTRIGGERED (+ input starved: reads ACTIVE claims) |
| `t2_facts` | `verbatim/extraction/t2.py:444 propose_facts` (needs injected `model_fn` — an LLM callable) → `:701 verify_quotes` (deterministic byte-verify) → `:820 commit_facts` (`:872 INSERT`) | **None — zero production callers.** No JobKind; LLM-shaped proposal stage. | ORPHAN + LLM-GATED |
| `facts_fts`, `fts_rows` | `verbatim/storage/repos.py:1737 FtsRepo.index` via `ingest.py:1825 _index_claim_text_tx` | Wired — but `ingest.py:1821` gates on `outcome.state == Lifecycle.ACTIVE`; all claims PENDING. | GATED (pending claims) |
| `entities`, `claim_entities` | `verbatim/storage/repos.py:958,1016 EntitiesRepo.find_or_create`, `:1050 link_claim` | **Zero production callers** outside `export.py:1389` and tests — v2 claim-side entity linking is write-dead in this tree. | ORPHAN |
| `state_facts`, `preferences` | `enrichment/prefs_state.py` via `units_jobs` | Wired — but narrow lexeme/key rules → thin coverage (measured: 0 + 1 rows). | THIN, not dead |

Retrieval side (`verbatim/retrieval/v7/`): `lane_graph` reads `graph_edges` (`graph.py:351`), `lane_obs` reads `observations_v7` (`obs.py:536`), `lane_typed` scans `state_facts`/`preferences`/`events_v7`/`t2_facts` (`typed.py:92-95`), `lane_entity` reads `entity_mentions`/`entity_canon` (`entity.py:429`). Ten lanes registered (`types_v7.py:93-104`: LEX, FUZZY, DENSE, ENT, TIME, GRAPH, TYPED, OBS, EXACT_ID, SOURCE); `search(retrieval="auto")` picks V7 when the store carries V7 projections (`facade.py:2205-2209`). The parent's "4/8 lanes (obs,typed,facts,graph)" maps to OBS/TYPED/GRAPH plus the claim-side FTS lane — this tree has no lane literally named `facts`; `facts_fts` is the v6/claim-side FTS (`facts_fts` + `fts_rows`, written only for ACTIVE claims).

## 3. (b) Why they didn't fire — the actual mechanism

**Mechanism 1 — orphan producers (structural).** `grep` over `verbatim/` for `build_edges|consolidate_scope_v7|propose_facts|commit_facts|write_supplied_edges` finds **zero callers outside their own modules** (verified). `JobKind` (`types.py:192-222`, 29 kinds) has no graph/t2/v7-consolidate kind. The modules are spec-complete (edge families, budgets, incremental rebuild — `graph_jobs.py:1-56`) and were exercised only by their test suites (`tests/retrieval/v7/test_graph.py` mirror-DDL note, `graph_jobs.py:51-55`).

**Mechanism 2 — admission gate (config).** `_admit_apply` (`core/policy.py:731`): sensitive/condition-false/judge-not-durable → PENDING; then `ctx.cfg.admission.require_review` at `:838` → PENDING `reason="review_required"`; `is_explicit_remember` → ACTIVE; eff_slot+ASSERTED+predicate whitelist → ACTIVE; else PENDING `abstained`. Default `require_review=True` (`config.py:29`) short-circuits every harvested claim to PENDING before the whitelist branch can matter for non-remember prose. FTS+embed for the claim is indexed **only** `if outcome.claim_id and outcome.state == Lifecycle.ACTIVE` (`ingest.py:1821-1825`). Eval arm used defaults → `[measured]` 14/14 claims `state='pending'` (`review_required`), 34 open reviews, `facts_fts`=0.

**Mechanism 3 — consolidation trigger absent (scheduling).** v3 `consolidate` exists and is routed by `handle_consolidate`, but CONSOLIDATE jobs are enqueued only from the refresh scheduler (`refresh.py:963,981`); nothing on `Memory.add`→`drain` schedules refresh/consolidation, and `consolidate_scope_v7` is never called at all.

**Eval ingest contract note.** `VerbatimArm` does `Memory(path, worker="external", infer=True, encoder="hashing")` (`eval/v7/arms.py:578`) then `add_with_retry(mem, doc, infer=True, metadata=item_metadata(item))` + `drain_memory` (`eval/v5/harness.py:108,131` → `Ingester.drain_report`, `ingest.py:1102`). `item_metadata` (`arms.py:220`) packs speaker/session_id/when/kind into `metadata` — NOT the units_v7 add-args turn contract — but `derive_units` (`projections/units_v7.py:826`) still mints turn units, and each add also emits a session unit whose `_agg_span` (`units_v7.py:1108`, emitted ~`:1211-1243`) covers exactly the turn's bytes → **byte-identical session+turn duplicates** (`[measured]` 16 units for 8 adds), each with its own `unit_fts` row → 2× lexical pool pressure feeding the pool-cap truncation the parent measured.

**Ambiguity flag.** If the parent's "entities" meant `entity_mentions`/`entity_canon`, that contradicts this checkout: they populate via the turn units (`[measured]` 23 mentions, 13 canons). If it meant the v2 `entities`/`claim_entities` tables, they're write-dead (orphan `EntitiesRepo`). Worth one `SELECT COUNT(*)` on `entity_mentions` in `/tmp/v7-trackr-ezeh8hhn/mem.db` to disambiguate.

## 4. Live repro census (this session, scratch db — 8 dialogue turns)

```
units=16 (8 session + 8 turn, byte-identical pairs)   entity_mentions=23   entity_canon=13
events_v7=7   entity_aliases_v7=5   preferences=1   unit_fts_rows populated   lex_stats populated
claims=14 — ALL state='pending' (review_required);   open_reviews=34
observations_v7=0  t2_facts=0  graph_edges=0  state_facts=0  facts_fts=0  fts_rows=0  entities=0  claim_entities=0
```
`[measured]` lane report: lex ok, fuzzy partial, dense ok, ent ok, time skipped (intent), graph ok-but-empty-inputs, typed skipped (`intent_not_typed`, `typed.py:870-875`), obs ok-but-empty-inputs.

**Then I invoked the orphan producers directly in a transaction** (then rolled back): `build_edges(conn, scope, gen, unit_ids)` wrote **52 edges** (16 `co_mention`, 36 `same_session`) with zero new inputs; `consolidate_scope_v7(...)` ran clean — 16 units processed, 2 slots touched, 2 profiles written, **0 observations** (each slot needs ≥2 supporting units; 8-turn sample too thin — on 990 real turns this becomes nonzero). `[measured]`

## 5. (c) Minimal deterministic write-path spec

The machinery is already deterministic T0 ("no models, no inference beyond the explicit connective lexicon" — `graph_jobs.py:6-7`; consolidate reads only `state_facts`/`preferences`/`events_v7`/`entity_mentions` — `consolidate_v7.py:442-492,599`). **No new extraction is needed to light graph+obs lanes — only wiring.** Ordered by cost:

1. **Wire `build_edges` at the tail of `project_units_v7`** (~10 lines): after the units/mentions inserts, call `build_edges(conn, scope_id, generation, minted_unit_ids)` inside the same tx (it resolves bare ids against `units`, reads `entity_mentions`/`entity_canon`/`source_revisions` itself — `:697-761`). Emits `co_mention`/`same_session`/`adjacent_turn`/`temporal_near`/`causal` (`:85-91`, causal = connective-lexicon only, byte-pinned both clauses). `[measured]` works today.
2. **Trigger `consolidate_scope_v7`**: options — (a) call it in `project_source_v7` after the projection savepoint (bounded `budget_units`, resumable cursor — `:1349-1395`); (b) map a `JobKind` to it and enqueue from `enqueue_source_jobs`; (c) call from a drain-completion/`Memory.consolidate()` hook. Feeds `observations_v7` (+ `obs_profiles_v7`) from existing fact tables.
3. **`t2` without an LLM** ("t2-lite"): `verify_quotes`+`commit_facts` are already deterministic; only `propose_facts` is model-shaped. A deterministic proposer can synthesize `statement="subj pred obj"` rows from `extract_events`/`extract_state_facts`/`extract_preferences` outputs, carrying each extractor's own byte pins as `quotes` → `verify_quotes` byte-checks → `commit_facts` writes `verified=1` rows (`t2.py:701,820-872`). This lights the `t2_facts` slice of lane_typed (highest typed prior, `P_T2_VERIFIED=2.0`, `typed.py:117`) with honest pins.
4. **Broaden deterministic extractors** (density for typed/obs lanes): `extract_events` is lemma-canonical SPO with coordination, polarity/negation, pinned verification (`events.py:4-37`) — widen the §32.10 predicate lexicon and add quantity/time-expression capture (`occurred_*` normalization already exists); `extract_mentions` (`entities_v2.py:263`) already covers capitalized runs + speaker. Each extra assertion feeds `consolidate_scope_v7`'s slot tally (`_batch_assertions`, `:442`) and co-mention edges.
5. **Claim-side (facts_fts/entities/claim_entities)**: two honest options — (a) eval-time auto-approval: drain open reviews in the harness (the review list is durable; `_admit_pending`-style triage exists in eval code) or set `admission.require_review=False` for the eval arm — flips `ingest.py:1821` on, indexing claim FTS + enqueuing EMBED; (b) wire `EntitiesRepo` into the admit path (`repos.py:958/1050`). Pending-claim recall stays correct; the change is scope-gated to the eval config.
6. **Dedup session+turn units**: single-turn sources produce a session unit covering the identical byte span (`emit_sessions` ~`units_v7.py:1211-1243`) — suppress the session unit when `len(members)==1` or exclude kind=='session' from `unit_fts` nomination. Halves the lexical pool for the exact-echo distribution the parent measured.

## 6. (d) How Hindsight (TEMPR) and Mem0 populate equivalent structures at ingest

**Hindsight / TEMPR** (Vectorize; paper `[paper]` https://arxiv.org/pdf/2512.12818, docs `[vendor]` https://hindsight.vectorize.io/developer/retain , …/retrieval):
- `retain()` runs **LLM-driven rich fact extraction** — facts + emotions + reasoning, narrative context preserved — classified `experience` vs `world` by *speaker context* ("who is speaking", not grammar).
- **Entity recognition + resolution at ingest**: people/orgs/places/products; fuzzy name matching reinforced by co-occurrence + temporal proximity; label entities via controlled vocab.
- **Knowledge graph built at ingest** with four link families — entity, temporal, semantic, causal — the same family set as `build_edges`'s `co_mention`/`temporal_near`/`semantic_knn`/`causal`.
- **Bi-temporal**: occurred-time vs learned-time tracked separately — matches `occurred_*` vs `recorded_at` in `units`.
- **Observations consolidated automatically in the background after retain** — i.e. consolidation is a first-class post-ingest step, exactly the trigger verbatim is missing.
- **TEMPR recall** = four parallel strategies (Semantic, Keyword/BM25, Graph Traversal, Temporal) fused by consensus + rank + neural rerank — the multi-lane parallel-fusion pattern verbatim's v7 pipeline mirrors. Reported LoCoMo 89.61% / LongMemEval 91.4% with 20B backbone `[paper]`.

**Mem0** (paper `[paper]` https://arxiv.org/html/2504.19413v1 , docs `[vendor]` https://github.com/mem0ai/mem0/blob/main/docs/core-concepts/how-it-works.mdx):
- `add(infer=True)` pipeline: **context lookup → LLM fact extraction → dedup + embed → entity extraction**; entities become Graph Memory nodes (Mem0^g: directed labeled graph, subject–relation–object triplets with ADD/UPDATE/DELETE/NOOP conflict ops). Raw storage without extraction only at `infer=False`.
- Reported +26% LLM-judge vs OpenAI, 91% lower p95 vs full-context `[paper]`.

**Delta vs verbatim**: both systems populate *all* downstream structures inside `retain()`/`add()` (or an auto-triggered post-step) — nothing ships unwired. Verbatim's equivalent structures exist and are deterministic but have **no invoker** (graph/obs/t2) or a **default-gated** invoker (claim FTS). Also note both comparators use an LLM at ingest; verbatim's design intent is T0-deterministic, so the t2-lite deterministic proposer (§5.3) is the honest analog.

## Verdicts

Ranked by expected lift on `any@10` (parent's measured 0.195 vs flat_bm25 0.587 `[parent]`; lane-starvation contribution per parent: 4/8 lanes emit zero candidates `[parent]`):

1. **[P0] Call `build_edges` inside `project_units_v7` post-insert** — lights the graph lane today, zero schema/LLM/config cost. Expected lift: candidate coverage on entity-anchored and session-local questions (the lanes' zero-candidate contribution removed; parent attribution shows obs/typed/facts/graph misses `[parent]`). Cost: ~10 lines + tests exist.
2. **[P0] Trigger `consolidate_scope_v7` after source projection** (option §5.2a/b) — lights obs lane; machinery proven on repro. Lift: cross-turn aggregated observations answer "what do we know about X" patterns BM25 can't reach. Cost: ~20 lines (call + cursor test).
3. **[P1] Flip `admission.require_review=False` for the eval arm (or harness-side review drain)** — lights `facts_fts`/claim-side FTS + EMBED immediately; lowest-risk change since it's a config on `VerbatimArm`, not engine code. Lift: restores the claim-FTS lane entirely (currently 0 candidates; parent lists `facts_fts` empty `[parent]`). Cost: 1 config flag.
4. **[P1] t2-lite deterministic proposer** (`events`/`state_facts`/`preferences` → `verify_quotes` → `commit_facts`) — lights `t2_facts` slice of typed lane with verified pins, no LLM. Lift: typed lane gains its top-prior row class (`P_T2_VERIFIED=2.0`, `typed.py:117`). Cost: ~100 lines.
5. **[P2] Suppress byte-identical session units for single-message sources** — halves `unit_fts` pool; directly attacks the parent's 62% pool-cap+deadline truncation share `[parent]`. Lift: fewer packed-out hits at pool=200 (bounded; pool-cap fix likely dominates). Cost: ~15 lines + test.
6. **[P2] Widen deterministic SPO/time/quantity extraction** (extend `extract_events` lexicon, `extract_state_facts` key rules, capture time-expr/quantity into `events_v7`/`state_facts`) — feeds obs slot density (≥2-support rule) and typed term coverage. Lift: recall on paraphrase/state questions — the parent's residual 38% gap `[parent]`. Cost: moderate, test-bound.
7. **[P2] Wire `EntitiesRepo` into admit** (or accept claim-side entity links from the eval's metadata speaker) — revives `entities`/`claim_entities`. Lift: only if parent's "entities" meant v2 tables — verify with one COUNT on their store first. Cost: ~30 lines.
8. **[P3] Check `intent_not_typed`/`intent_*` skip rate in the eval** — typed/time lanes self-skip on non-fact-shaped intents (`typed.py:870-875`, `obs.py:473-478`); if the query classifier under-emits typed classes on LoCoMo questions, that's a routing gap on top of the data gap. Cost: instrumentation only.

Not the mechanism: `worker="external"` (drain runs all enqueued jobs — `ingest.py:1102`), `infer=True` vs `infer=False` (harvest proposes candidates either way; they all land PENDING under `require_review`), unit `kind` filtering (turn units exist and carry mentions). The gap is **producer wiring + one config default**, not worker mode or infer path.
