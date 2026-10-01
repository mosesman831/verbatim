# V8-09 temporal write-path fixes — measured on the real dev store

**Verbatim @ HEAD 1c86f4f (occurred-population fix landed), LoCoMo dev, cat-2 temporal n=175.**
Prototyped entirely in `/tmp` copies (nothing committed/pushed). Method: post-hoc sqlite mutations + tree source-patches over the rebuilt baseline store — measures write-path arms in seconds rather than ~17min/arm rebuilds.

## 0. Method and pairing discipline

- Baseline rebuilt once as instructed → `/tmp/v7w/mem.db` (238M) + `report_baseline.json`.
- Arms run on copies of that store (`/tmp/v7arm/<arm>/wd/mem.db`) with DDL/DML mutations, plus per-arm patched trees (`/tmp/v7arm/<arm>/tree`) for read-path changes. Ref↔source_id mapping via deterministic `external_id = hmac("memory-add|"+ns+"|"+payload)` — 2877/2877 mapped, no re-ingest.
- **Paired determinism:** `_search_v7`'s `now` frozen to the baseline run instant via `V7_PROBE_NOW_US` env (patch on every arm tree). Without this, `now`-keyed scoring (currency/proximity/validity) drifts between runs and shuffles delivered ranks.
- **Noise band:** original vs frozen baseline differ by ±4 tasks (43 vs 44 misses; delivered *sets* identical, tail-order churn). **All closure counts below are computed on the stable intersection of 40 tasks** that miss under both references; the prompt's "41 residual" sits inside this noise band.
- Baseline measured (frozen): cat-2 item-level **any@10 .6686 / all@10 .6057 / ndcg .4234 / mrr .3679 / any@20 .7486**; session-level any@10 .8743 / any@20 .9429. (Original: .6743/.6171/.422/.364/.7543 — drift-noise away.)
- Frozen-baseline miss taxonomy (44; stable 40):
  - `no-window:not-pooled` **12** — "when did X" queries, no resolvable window, gold never pooled by temporal lane (lane_miss mostly)
  - `abstain` **10** — verdict withheld
  - `no-window:fallback-buried` **9** — gold pooled at lane rank 2–53 but fusion/pack drops it below k
  - `windowed:mention-rescue` **3** — gold `occurred` outside window, a mention inside could rescue (q032, q026, q068)
  - `windowed:in-window-rank` **3** — gold inside window but ranked out (q065, q024, q041)
  - `windowed:occ-out-no-mention` **2** — gold outside window, no mention rescue (q011, q078)
  - `windowed:misanchored` **1** — q018 "May" → May-2026 instead of 2022

## 1. Metrics (cat-2 temporal, paired vs frozen b0)

| arm | any@10 | all@10 | ndcg@10 | mrr@10 | any@20 | session any@10 | stable closed | new misses | net misses |
|---|---|---|---|---|---|---|---|---|---|
| b0 (frozen baseline) | .6686 | .6057 | .4234 | .3679 | .7486 | .8743 | — | — | 40 |
| a1 mentions-union (V8-09.04) | .6743 | .6114 | .4285 | .3728 | .7657 | .8686 | **1** | 0 | 39 |
| a2 fallback order+restrict (V8-09.06) | .6743 | .6114 | .4243 | .3672 | .7657 | .8857 | **3** | 1 | 38 |
| a3 claims anchor (V8-09.05) | .6743 | .6171 | .4266 | .3690 | .7543 | .8800 | 0 | 0 | 40 |
| a4 events 0.5/0.5 + backfill (V8-09.07) | **.6800** | .6114 | **.4690** | .4212 | .7486 | .8743 | **4** | 5 | 41 |
| a5 mentions+beats (V8-09.08) | .6743 | .6114 | .4280 | .3722 | .7657 | .8629 | 1 | 0 | 39 |
| a6 as_of=last-session (V8-09.01/02) | .6400 | .5829 | .3379 | .2632 | .7429 | .8343 | 1 | 3 | 42 |
| a24 (a2+a4) | .6686 | .6057 | .4651 | .4177 | .7429 | .8743 | 3 | 5 | 42 |
| **all_n6 (a1+a2+a3+a4+a5)** | **.6800** | **.6171** | **.4673** | .4177 | .7543 | .8686 | **4** | 5 | 41 |
| all (incl. a6) | .6114 | .5429 | .3193 | .2522 | .7257 | .8171 | 4 | 9 | 45 |

bm25 reference (from prompt): item any@10 .640 / session .880. All non-a6 arms stay above on item-level; session-level none reach .880.

## 2. Per-miss-class closure table (stable set n=40)

| miss class | n | a1 | a2 | a3 | a4 | a5 | a6 | all_n6 |
|---|---|---|---|---|---|---|---|---|
| no-window:not-pooled | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| abstain | 10 | 0 | 1 | 0 | 2 | 0 | 1 | 1 |
| no-window:fallback-buried | 9 | 0 | 2 | 0 | 2 | 0 | 0 | 2 |
| windowed:mention-rescue | 3 | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| windowed:in-window-rank | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| windowed:occ-out-no-mention | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| windowed:misanchored | 1 | 0 | 0 | 0 | 0 | 0 | 0* | 0 |
| **TOTAL closed** | **40** | **1** | **3** | **0** | **4** | **1** | **1** | **4** |

\* a6 surfaces q018's gold into the lane pool (rank 22, was unpooled) but not into delivery.

**Stable closures:** a1: q032 | a2: q036, q002, q055 | a4: q036, q053, q010, q055 | a5: q032 | a6: q055 | all_n6: q036, q032, q053, q010.
**New misses introduced** (delivered→miss; gold still pooled at identical lane rank — downstream fusion/abstain churn, not lane regressions): a2: q066 | a4/a24/all_n6: q011, q025, q042, q066, q029 | a6: q043, q066, q075 | all: +q030, q012, q017.

## 3. Per-arm findings

### a1 — write-time mention resolution (V8-09.04): PARTIAL PASS
`unit_time_mentions` built over all 4143 latest units, anchored on each unit's `occurred` mid: **4258 rows**, 271 unbounded dropped, 0 ambiguous, **3 rows anchored ≥2026 — all genuine "2077" explicit-year content, zero ingest-misanchors**. Window scan unions occurred+mention overlap on all 26 windowed tasks.
- All 3 mention-rescue golds now surface in-pool: q032 rk1, q026 rk2, q068 rk5 (all unpooled at baseline).
- But only q032 converts to delivery — q026/q068 still fusion-dropped below k. Lane rescue works; delivered-rank conversion needs downstream weight (arm 5 didn't add it).
- 12 not-pooled misses unaffected (they're no-window — mentions never consulted).

### a2 — recorded_fallback ordering+restriction (V8-09.06): PASS (small)
`ORDER BY (occurred IS NULL), occurred {ASC|DESC}` (ASC only for first/earliest intents), restricted to units carrying ≥1 nominated term (unit_fts) or entity canon.
- **scan_truncated 101 → 2** (was saturating the 4096 cap on ~all fallback tasks); median rows swept 2614 vs ~4096. Big honest operational win.
- Closes 3 stable misses (q036, q002 — the fallback-buried rank 44→delivered, rank 30→delivered — plus q055 abstain→delivered). 1 new miss (q066, gold still pooled rk2 — downstream churn).
- Note: fallback fires on 100/175 tasks (events-produced pool does NOT suppress it as the spec's "not pool" gate implies — pool is empty after event phase on these).

### a3 — claims event_us anchoring (V8-09.05): PASS (correctness, not recall)
`_extract_valid` re-run with each claim's unit-occurred mid as anchor: **2026-misanchored valid_intervals 212 → 2** (the 2 are genuine 2077 explicit_year rows). All **98 relative_day/relative_week rows now correctly anchored (98/98, spec target met)** — 210 rows changed. Eval effect ~0 (closures attributable to boundary noise only). Ship for correctness; the misanchors weren't what was suppressing cat-2.

### a4 — events-first loosening (V8-09.07): PARTIAL, VOLATILE
DB: `subject_source` column added; 473 NULL-subject rows backfilled to `speaker_canon` (subject_form none=454 + pronoun=19); description/second_person/reused correctly left NULL (270). Query side: subject|predicate OR at 0.5/0.5 weight, EV_MAX 3.5→2.5.
- events_matched contribution **unchanged at 48/149** — the disjunction changes *scores*, not match counts.
- Best single-arm metric gain: ndcg +.046, mrr +.053, any@10 .680. Closes 4 stable (q036, q053, q010, q055 — incl. 2 abstains).
- **But introduces 5 new misses** — all downstream: gold stays pooled at identical lane rank; the rescored pool reshuffles fusion/abstain margins. Net recall-neutral, strong precision-side gain.

### a5 — mention-beats-session scoring (V8-09.08): NO MEASURABLE INCREMENT
Re-anchors pooled candidates to mention mid when the mention interval beats the unit anchor. Identical results to a1 (closures/metrics within noise) — the targeted session≠event-date cases either already delivered or never reached delivered rank anyway. The mechanism fires (t_axis='mention' on pooled candidates) but converts nothing extra on this corpus.

### a6 — as_of param (V8-09.01/02): MECHANISM WORKS, NET NEGATIVE
`as_of` threaded `search() → _search_v7 → build_query_view` (result-cache probe bypassed when set; adapter passes last-session date mid).
- Anchoring verified correct: q018 "May" resolves May-**2022** (was May-2026) → gold surfaces at pool rk22; q055's "recently" window moves 2026-08→2023-08 → task delivers.
- **But net-negative**: any@10 −.029, ndcg −.086, session-level −.040. Re-anchoring `now` to each conversation's last session re-sorts every now-dependent score (temporal proximity, recorded/occurred distance priors) for ALL tasks — the 2 window-misanchor fixes are swamped by ~4 tasks' worth of delivered-rank churn elsewhere (3 confirmed new misses: q043, q066, q075 — all no-window tasks broken by anchor-shift scoring drift, plus abstain flips).
- Verdict: needs scoping — apply as_of to the query window anchor only, or only on queries containing relative expressions; don't re-anchor global `now` for scoring.

### Combinations
- **all_n6 (everything except as_of): best result — any@10 .680 (+.011), all@10 .6171 (+.011), ndcg .4673 (+.044), mrr .4177 (+.050); 4 stable closures, 5 new misses (all the a4/a2 downstream-churn set).** The gains are real but mostly precision/ranking-side (ndcg/mrr), not recall.
- `all` (with as_of): poisoned by a6 — .6114 any@10.

## 4. What the write-path cannot fix

The largest miss classes are untouched by every arm:
- **12 no-window not-pooled** — gold never enters the temporal lane pool (it's a lexical/coverage problem upstream of temporal, or units carrying no nominated terms).
- **9→7 fallback-buried / fusion-buried** — gold pooled but packed out below k; a2's ordering recovered only 2 of 9. The fix lives in fusion scoring/lane weighting, not the write path.
- **10 abstains** — a4's event scores released 2; the rest withhold under every configuration.
- **3 in-window-rank, 2 occ-out-no-mention, 1 misanchored (delivery)** — unmoved.

Net: on the stable 40, the entire V8-09 bundle converts ~4 misses (−10%) at ~5 introduced (−2 net by count), while improving ranking metrics meaningfully (ndcg +.044). The honest headline number is metric gain, not miss closure.

## 5. Caveats

- Post-hoc store mutations approximate write-time changes; a real rebuild could shift internals slightly (mentions built at ingest would anchor on then-current occurred — here identical since occurred is already correct at HEAD).
- Freeze env `V7_PROBE_NOW_US` was injected for pairing only — not part of any arm's semantics.
- a4's `subject|predicate` OR at 0.5/0.5 with EV_MAX 2.5 is one reasonable reading of the spec's weighted disjunction; alternatives (predicate-only boost, hard subject gate) unexplored.
- Delivered-set churn ±4 tasks/run is inherent; stable-set reporting already controls for it.

## Verdicts

| spec | arm | verdict |
|---|---|---|
| V8-09.04 mention resolution | a1 | **Ship** — union works (3/3 rescue golds surface in-pool, 0 2026-misanchors), converts 1 miss; combine with fusion weight to convert the other 2 |
| V8-09.06 fallback ordering | a2 | **Ship** — closes 3, kills scan truncation (101→2, ~40% fewer rows); tiny regression tail |
| V8-09.05 claims anchor | a3 | **Ship for correctness** — 212→2 misanchors, 98/98 relatives correct; no recall effect |
| V8-09.07 events loosening | a4 | **Ship with tuning** — biggest metric gain (+.046 ndcg) but 5 new misses; gate the 0.5/0.5 disjunction tighter or compensate in fusion |
| V8-09.08 mention-beats-session | a5 | **Skip for now** — fires but converts nothing measurable beyond a1 |
| V8-09.01/02 as_of | a6 | **Do not ship as prototyped** — mechanism correct (misanchored windows fixed) but global now-re-anchoring costs more than it wins; rescope to window-anchor-only or relative-expression queries |
| bundle | all_n6 | **Best combined: +.011 any@10/.6171 all@10, +.044 ndcg, −4 stable misses, +5 churn misses** |

Where the remaining ~36 stable misses live: **12 not-pooled (upstream pooling/coverage), ~9 abstain, ~7 fusion-buried, ~6 windowed hard cases** — the next wins are in fusion/lane scoring and abstention policy, not the write path.
