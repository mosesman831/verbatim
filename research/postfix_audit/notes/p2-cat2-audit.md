# Cat-2 temporal fix verification — LoCoMo dev split (verbatim@1c86f4f)

Repo: `github.com/mosesman831/verbatim` HEAD `1c86f4f` \"Beat-it retrieval plumbing fixes\". Store rebuilt exactly as spec'd: `VERBATIM_EVAL_LOCOMO=1 python3 -m eval.v7.track_r --dataset locomo --split dev --arms verbatim --workdir /tmp/v7w` (timeout_ms=2000). Corpus `research/v7_formula_search/locomo10.json` (digest 41df5b3a…). Dev split = conv-30/42/47/48/49 → 990 tasks, **175 cat-2 temporal**. Built store `/tmp/v7w/mem.db` (~250MB); live lane probes run on a copy (`/tmp/probe.db`) driving `run_search` directly with the same wiring as `Memory._search_v7`. All values measured on the built db; baselines from `research/beat_it/notes/r8-temporal-cat2.md`. Read-only; nothing committed.

## (a) units.occurred_* — FIXED

| metric | baseline | now |
|---|---|---|
| units total | 7020 | **4143** (2877 turn + 1266 sentence_window; 0 session — twin minting suppressed) |
| occurred NULL | 7020 (100%) | **0 (0%)** |
| precision / source | — | `instant`/`explicit` on all 4143 (`_parse_when_str` on LoCoMo `when`) |
| occurred years | — | **2022: 1868 (45.1%), 2023: 2168 (52.3%), 2024: 107 (2.6%); 2026: 0** |

Corpus-era dates on every unit, including derived sentence_windows.

## (b) unit_fts `when` field — FIXED (text-field flood persists, now gated)

| probe | baseline | now |
|---|---|---|
| distinct `when` strings | 1 (`'2026-09-23 … wednesday'` ×7020) | **126 real session dates** |
| rows containing '2026-09-23' / '2026' | 7020 | **0 / 0** |
| df('wednesday') in `when` field | 7020 | **590** (real weekday spread) |

Residual: the **`text` field** still carries date-token floods — df('2023')=1821, 'pm'=2560, 'am'=1459. HEAD mitigates rather than eliminates: `lex_df` prefetch gate (`_DF_PREFETCH_FLOOR=1400`, `θ=0.2·N_elig`) bars df>1400 terms before MATCH. Observed firing live: 'on' (3647), 'to' (1712) gated; 'pm'/'2023'/'am' also >1400 → gated. 'wednesday' (590) and similar mid-flood terms under floor still fetch.

## (c) events_v7 — FIXED; valid_intervals — NOT fixed

| table | baseline | now |
|---|---|---|
| events_v7 occurred NULL | 1248/1248 | **0/1248**; years 2022:622, 2023:607, 2024:19 |
| events_v7 subject_canon NULL | — | **743/1248 (59.5%)** — still gappy |
| valid_intervals rows | 212 anchored 2026 | **7122 total; relative_day=57 + relative_week=41 → all 98 still anchored 2026; asserted_current_at 112/152 in 2026; total from_us-2026 = 210** (plus 2 rows year=2077 from \"Cyberpunk 2077\" jokes) |

The claims path (`core/claims.py::_extract_valid` → `parse_time_expression(text, event_us)`) still anchors relative_day/week to the envelope's ingest-clock `event_us` — the turn `when` metadata is never threaded there. So relative intervals remain ingest-2026 junk; only the units/events projections got the fix.

## (d) cat-2 metrics — big improvement, beats bm25 on item granularity

| metric | baseline | now | bm25 |
|---|---|---|---|
| item any@10 | 0.371 | **0.691** | 0.640 |
| session any@10 | 0.560 | **0.869** | 0.880 |
| item any@20 | — | 0.766 | — |
| session any@20 | — | 0.949 | — |
| item all@10 / prop@10 / ndcg@10 / mrr@10 | — | 0.634 / 0.661 / 0.427 / 0.361 | — |
| abstain_rate (status=insufficient) | — | 0.160 | — |
| p50 latency | — | 674ms (p95 ~834ms; vs overall-run p95 1580ms) | — |

Attribution at k=20 (per_task): **delivered 134 (was 83), lane_miss 15 (was 63), rank_shift 17, abstain 9, packed_out 0**. Note: `delivered` counts probe@k20, so 134 > the 121 implied by any@10 — consistent.

Metric quirk worth flagging: overall **session-granularity ndcg@10 reports 1.072** (>1.0 impossible for proper nDCG) — normalization bug in session-aggregated ndcg; item granularity values are sane.

## (e) window scan — fires now; recorded_fallback unchanged

| probe | result |
|---|---|
| queries with parsed window | 26/175 |
| window-scan actually fired | **26/26** (`window+recorded`=25, `events+window+recorded`=1) — the `ORDER BY (occurred_start_us IS NULL), occurred_start_us` path runs now that occurred exists |
| recorded_fallback ordering | **unchanged code**: `ORDER BY u.recorded_at_us`, `SCAN_ROW_LIMIT=4096` — still ingest-ordered, not occurred-ordered |
| fallback fired / truncated | 105/175 fired; **96/105 truncated** (4096-cap or deadline); lane status `partial` on 106 queries total |
| events-first lane | `events_index=ok` 149/175, but **`events_matched>0` on only 43/175** — strict subject_canon∧predicate_lemma AND + 59.5% NULL subjects keeps the lane mostly silent |
| query anchor | `query_time_us` = wall-clock `now_us()`; corpus `question_date` not threaded. Consequences measured: 2 windows anchored to 2026 (yearless-month query, \"more recently\"), 1 window at year 2077 (\"Cyberpunk 2077\" → date rule T04) |
| windowed outcomes | 15/26 delivered, 11 miss |

## (f) residual 41 cat-2 misses — classification

| class | n | anatomy |
|---|---|---|
| no-window, lane_miss (gold absent from top-64 pool) | 15 | dominated by **relative-expression golds**: \"yesterday\", \"last week\", \"next month\", \"3 years now\", \"tomorrow\" — gold turn carries no absolute date so nothing to match; naive OR-probe finds gold in top-30 for only 1/10 |
| no-window, rank_shift (gold in pool, out of top-10) | 13 | ranking |
| no-window, abstain/other | 2 | incl. cases with events matched but withheld |
| windowed miss, gold *inside* window | 4 | in-window ranking |
| windowed miss, gold *outside* window | 7 | **session-date \u2260 event-date anchor mismatch** — window anchored on the turn/question session date but the event happened a different day (same relative-expression root cause) |
| lane skipped | 1 | conv-48/q054 \"adopt first\" — non-temporal intent classification |

## Verdicts

**Worked**
- `occurred_from_when` write-path fix (r8 lever 1): units 0%→100% populated with corpus-era dates; `when` field fully de-flooded (7020→0 junk); events_v7 occurred 100%.
- Window-scan phase now fires on all 26 anchored queries — previously dead code.
- Cat-2 delivered 83→134, lane_miss 63→15, any@10 0.371→0.691 item / 0.560→0.869 session — beats bm25 on item, parity on session.
- `lex_df` prefetch gate live and firing (df>1400 terms barred pre-MATCH).
- Session-twin suppression shrank units 7020→4143 and de-flooded the corpus.

**Didn't / partially worked**
- `valid_intervals` relative_day/week still 100% ingest-2026 anchored — `_extract_valid` never sees turn `when` (claims path unfixed).
- `recorded_fallback` untouched: ingest-ordered, 4096-capped, fires on 105 queries and truncates on 96 → `partial` lane status on 60% of cat-2.
- Events-first still ANDs subject_canon∧predicate_lemma against 59.5%-NULL subjects → contributes on only 43/175.
- Anchor precedence q_ts>session>wall NOT implemented — query_time is wall clock; 2 wrong-2026 windows + 1 year-2077 window measured.
- Text-field date flood persists under the df-gate floor (590–1400 df terms still fetched).

**Next cat-2 lever (ranked by measured coverage)**
1. **Write-time in-text temporal resolution anchored to unit.occurred** — resolve \"yesterday/last week/next month/3 years now/tomorrow\" in turn text at ingest (occurred is now available as anchor) into absolute event intervals/postings. Directly addresses ~22/41 misses (15 relative-expr lane_miss + 7 windowed anchor-mismatch) — the dominant residual class.
2. **Loosen events-first + backfill subject_canon** (speaker/name fallback when canon NULL; weighted OR instead of strict AND) so the event lane can carry no-window \"when did X\" queries — currently silent on ~57%.
3. **Anchor precedence**: thread corpus `question_date`/session anchor over wall clock (fixes wrong-2026/2077 windows); make recorded_fallback occurred-ordered (or index-scan) so 105 fallback queries scan by event time, not ingest order.
4. **Claims path**: pass turn `when` as `event_us` into `_extract_valid` so relative_day/week anchors move off ingest-2026 (210 rows today) — matters wherever typed-lane state facts consume intervals.
5. In-window + deep-pool ranking (4 in-window + 13 rank_shift residuals): occurred-recency boost within window and for no-window temporal queries.

Artifacts: `/tmp/v7w_report.json|.md`, `/tmp/lane_rows.json` (per-query lane stats), `/tmp/probe_lanes.py`, `/tmp/audit1.py`, `/tmp/probe.db` (store copy). Env note: system python3=3.10 < required ≥3.11 — used pyenv 3.12.13 with `cryptography` installed; repro needs that.