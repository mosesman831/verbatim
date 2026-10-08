notes_markdown (also at /tmp/cat2_temporal_notes.md):

# cat-2 temporal composite verification — measured answers to (a)–(d)

**Provenance.** Store rebuilt locally via `python3 -m eval.v7.track_r --dataset locomo --split dev --arms verbatim --workdir /tmp/v7w` on `mosesman831/verbatim-new` @ `892f2b8` (the parent's `/tmp/v7-trackr-ezeh8hhn/mem.db` doesn't exist on this child VM; child sessions run separate VMs). Rebuild converges to the same fixture: `valid_intervals`=7122 (exactly the brief's number), `units`=7020, `events_v7`=1248, `claims`=7122, `entity_canon`=1513, corpus_digest `41df5b3a…` matches the parent's report. Outcome truth = parent report JSON; mechanism anatomy = store snapshot + live `Memory(path, user_id='track-r').search()` probes on all 175 cat-2 queries at limit=64. **[measured]** = my runs; **[code]** = file:line; **[report]** = parent run JSON; **[note]** = bundled research notes.

## 0. Frame

- cat-2 temporal = 175 tasks; verbatim any@10 = 0.371 (65) vs flat_bm25 0.640 (112); session 0.560 vs 0.880 **[report]**.
- Outcome classes @k=20 (attribution.py:119–190): delivered 83, lane_miss 63, rank_shift 24, packed_out 5 **[report, counted]**.
- On my store: ~61 misses have gold absent from top-64; ~31 have gold at ranks 11–60 **[measured]**.

## 1. (a) Parseable time anchors — **26/175 (14.9%)**

Classifier = real `build_query_view(query, now_us, known_canons=<1513 entity_canon rows>)` (facade.py:2814–2819; window via `resolve_query_window`, query_view.py:187) **[measured]**.

- intent_v2 primary: `temporal_point` 140 (80%), `temporal_range` 22 (12.6%), `duration` 4, `temporal_order` 3, `count_aggregate` 2, `comparison` 1, `multi_hop` 3 **[measured]**.
- Only **26/175** produce a resolved window (rules T03×14, T02×5, T04×2, T05/T18/T21/T23/T24 ×1). The other 149 (85%) are open \"when did X\"/\"which year\" questions — a window prefilter can't fire by construction.
- **22/26 windows land in corpus era (2022–2024); 4 anchor wrongly to 2026 wall-clock**: \"visit in May\"→May 2026 (yearless month), \"Cyberpunk 2077\"→year 2077 (T04), \"more recently\"→Aug–Sep 2026 (T21) **[measured]**. Root cause: all resolution anchors at caller `now_us` (query_view.py:251–255) — no `query_timestamp`/corpus-era anchor; the c6 note's A_q>A_sess>A_sys precedence gap **[note c6 §1]**.
- Query entity extraction works only via `known_canons` n-gram match (entities_v2.py:322–365): `norm.text` is folded so the capitalized-run fallback is dead code (deviation note at query_view.py:40–48 admits it); entity_canon vocab is populated but noisy ('has','lost','nate oh','deborah good' are canons) — e.g. \"When Jon has lost his job…?\" → entity_canons ('jon','has','lost') **[measured]**.

## 2. (b) `valid_intervals` — **No usable per-question times**

7122 rows, one per claim (schema.py:136–151):
- precision: `year` 2881 (40.5%), `unknown` 4031 (56.6%), `day` 99, `instant` 111; basis: `explicit_year` 2841, `unknown` 4031, `asserted_current_at` 152, `relative_day` 57, `relative_week` 41 **[measured]**.
- `from_us` non-NULL on 3091 (43.4%); of those 2877 are corpus-era 2022–2024 (93%) and **212 are ingest-anchored (2026-09-14..23)** — `_extract_valid(candidate_text, event_us)` (claims.py:585–602; core/time.py:108–120) resolves \"yesterday\"/\"last week\" against `event_us` = add-receipt time = **ingest wall clock**. Same anchor bug as the query side, on the write path.
- `units.occurred_*` NULL on **all 7020 units (0%)** — the arm passes `when` inside `metadata` (arms.py:220–233) but `units_v7._norm_occurred` reads only `occurred`/`occurred_*` keys (units_v7.py:295–330); `recorded_at_us` falls back to ingest time (units_v7.py:953–963) **[code + measured]**.
- Verified consequence chain: occurred NULL → `_when_tokens` stamps ingest-date tokens on every unit (units_jobs.py:701–715) → `when` FTS field = literally `'2026-09-23 2026-09 2026 september wednesday'` × 7020 rows (df=7020 each, idf≈0) **[measured lex_df/unit_fts]** → events_v7 occurred NULL on all 1248 (events.py docstring; units_jobs.py:906–945) → window scan `occurred_* IS NOT NULL` vacuous (temporal.py:621–622) → `recorded_fallback` scans up to SCAN_ROW_LIMIT=4096 ingest-ordered units for every no-window temporal query (temporal.py:691–716).
- Gold-side coverage for the 92 misses: gold turns DO produce claims (92/92 via spans→claim_evidence) but intervals are `unknown` 188 / `explicit_year` 101 (year-only) / `relative_*` 31 (2026 junk) / `asserted_current_at` 13 — **day-granularity corpus-era coverage ≈ 0; year-level ≈ 30%** **[measured]**.

## 3. (c) Composite-piece coverage vs the ~120 misses

| piece | measured coverage | verdict |
|---|---|---|
| window prefilter (needs occurred fix) | 26 anchored queries; 16 gold-session-inside; **9 are current misses**; +4–5 rescued by event-resolution (`resolve(gold_text, session_date)` inside window — e.g. \"Two weeks ago\"→Jan-18 \u2208 Jan-2023) | ~13–14/26 ≈ **11–12% of misses** — only piece with direct coverage |
| in-text resolver on gold turns | fires on **103/120 misses (86%)**; strict day-match 42/67 (63%) — most answers are window-form (\"week before 2 May\") | raw material exists; needs index + anchor fix |
| events-first (current code) | emits gold in **17/175 (9.7%); 1 of misses** | nearly dead: subject_canon NULL 745/1248 (59.7%); SQL ANDs subject+predicate (temporal.py:514–521); predicate family matched in only 33/175 |
| slot supersession | ~7–9 order-type questions; `state_facts`=156 rows only | small tail; targets currency not order |
| date-range surfacing in pack | 0 retrieval effect | answer-quality feature only |
| ±10% bounded boosts | miss gold ranks measured 11–60 | can't bridge 20–50-rank gaps |

**Dominant blocker (measured two ways):** (i) `_collect_postings` fetches full MATCH rowid sets for *every* content term pre-nomination (lexical.py:1076–1099) with `df_theta` disarmed (lexical.py:1005); the `[when]` doc prefix (arms.py:187–208) pollutes `text` df — '2023' **3303**, '2022' 2908, 'pm' 4670, 'am' 2581, months 481–643 — so one '2023' term nominates ~47% of the store and every cat-2 date query burns the 500ms deadline mid-scan; (ii) **60/61 never-surfaced misses match a trivial `unit_fts MATCH <or-of-content-terms>` probe** — gold is indexed and lexically reachable; it drowns under pool-cap/deadline/junk-flood, not paraphrase (the one true FTS miss, conv-48/q056, shares no terms) **[measured]**.

## 4. (d) Residual needs

- **Relative dates** (~7 queries + the dominant answer form \"week before X\" \u2190 gold \"last week\" + session date): needs `resolve(text, anchor=occurred=session_date)` at write + `q_ts>session>wall` at query.
- **Durations** (~9–12 queries; answers \"three years\"): stated-duration extraction or two-anchor interval arithmetic.
- **Ordering** (~7–9 \"first/second/third/more recently\"): occurred-ordered events_v7 (currently all NULL).
- **Year answers** (40/175 = 23%): event-time coarsening to year — cheapest residual.

## Verdicts

Ordered by expected cat-2 lift (any@10 / 175; estimates vs measured baselines):

1. **Populate `occurred` from `when` metadata at write** — `when`→`occurred_at` alias + \"H:MM am/pm on D Month, YYYY\" parser (all 134 session strings parse). Un-breaks `when` FTS tokens, window scan, events occurred, claim valid anchoring, recorded_fallback ordering. **Direct lift ≈ +9–14 tasks; ~20–40 LoC + reindex — highest leverage per LoC.**
2. **De-flood date/stopword postings** — strip `[when]` prefix from `text` field; arm `df_theta` (~0.15·N) or pre-cap per-term fetch before `_collect_postings` reads whole sets. 60/61 never-surfaced misses are FTS-reachable. **Expected +15–25 tasks** toward bm25 parity; ~50–100 LoC.
3. **Anchor precedence `q_ts>session_ts>wall`** — fixes the 3–4 wrong-2026 windows; precondition for relative queries. **+2–4; small cost.**
4. **Loosen events-first emission** — OR-match subject/predicate, coref fallback to `speaker_canon` (fixes 59.7% NULL subjects), demote predicate-lemma gate to a boost. Emit coverage 17/175 → plausibly ~80–100 (gold-has-event = 113/175). **Conditional +5–15** behind #2.
5. **Event-time extraction from turn text** — `resolve(text, anchor=occurred)` at write → event occurred; 103/120 misses' gold resolves. Rescues outside-window cases + enables year matching (40 answers). **+5–10; medium cost; depends on #1.**
6. **Slot supersession** — ~7–9 order/currency questions only; defer behind #1/#5.
7. **Date-range surfacing in pack** — recall-neutral; ship opportunistically for answer correctness.

**Bottom line:** the formula-search temporal composite as specced applies to only ~26/175 queries and fixes ≈11–12% of misses. The cat-2 gap is dominated by two upstream defects: (i) `occurred` never populated — `when` metadata ignored — which vacuates the window scan, nulls events_v7 occurred, anchors claim valid-times to ingest-2026, and stamps idf≈0 junk into every `when` field; (ii) date/stopword posting floods starve the 500ms deadline — 60/61 never-surfaced misses are OR-FTS-reachable yet invisible to the fused pool. Fix (i)+(ii) → plausibly ~0.55–0.65 item any@10 (bm25 parity); the composite then adds the last few points on anchored/relative/ordering residuals.
