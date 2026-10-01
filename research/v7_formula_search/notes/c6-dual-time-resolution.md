# c6-dual-time-resolution — dual-time indexing + relative-phrase resolution for Verbatim

## 0. Executive answer to the four sub-questions

1. **Anchor precedence.** Resolve every non-self-contained temporal phrase in a query against **the question's own message timestamp `t_q`**, falling back to session-start only for explicitly session-scoped phrases, and to wall-clock `now` *never* for stored-event references and only for live state queries lacking any timestamp. This is exactly what both primary reference systems do: LongMemEval stores a `question_date` field and per-session `haystack_dates` (all temporal expressions resolve session-relative to `question_date`), and Zep/Graphiti carries a per-message `reference_time` ("reference timestamp from the episode that produced this edge") used to resolve "next Thursday", "in two weeks", "last summer".
2. **SQLite layout.** Store four i32 *minute* columns — `said_at`, `valid_from`, `valid_until`, `rec_from`, `rec_until` — plus a `t_qual` pin flag. The only index the default profile needs is a **B-tree on `said_at`** (and `valid_until` for the closed-set lane): a resolved window `said_at BETWEEN a AND b` costs **0.105 ms at 100k rows for a 7-day window** (measured). Do NOT add an R*Tree to the default profile: a 4-dim `rtree_i32` over both axes costs **25.3 ms at 100k** to enumerate the open set — pointless, because open-interval facts match every late point (~80% of a real store) and are correctly handled as a per-candidate eligibility predicate at **0.055 ms per 40 candidates** instead.
3. **What query-time resolution adds over known-at.** Known-at (`r_from ≤ S < r_until`) is the *recorded* axis — "store contents as of seq S". Resolution produces a *valid/said-time* interval `[a,b)` from natural language. They are orthogonal predicates AND-ed together; `S` answers provenance questions, `[a,b)` answers world-time questions. A "double-temporal" query ("as of last Tuesday's checkpoint, what did we discuss in March") needs both.
4. **Edge cases.** Timezone-free dialogue: all anchors live in one naive writer clock; store an optional `tz_offset` but never retro-shift. "This morning" resolves against the *message's* day, so it cannot leak across sessions by construction. Conflicting cues → keep existing abstain+warn; add a third status `ambiguous_union` for non-contradictory multi-cue queries.

## 1. Anchor model and precedence

**Anchors, in precedence order:**

| Rank | Anchor | When used |
|---|---|---|
| A_exp | explicit datetime in phrase | "2024-03-15", "March 2024", "5pm March 3rd" — self-contained, no anchor needed |
| **A_q** | question/message timestamp `t_q` | **default for every relative phrase** ("last week", "3 days ago", "in March", weekday names, "this morning", "yesterday") |
| A_sess | session start `t_s` | only phrases that explicitly scope to the session: "earlier in this conversation", "earlier today" → interval `[t_s, t_q)` — narrows lower bound; the upper bound still comes from A_q |
| A_sys | wall-clock now | ONLY present-tense state queries ("what is my current X") when the query carries no timestamp; in eval/batch replay A_sys ≡ t_q anyway |

Rationale for A_q over A_sys (the bug this prevents): resolving "last week" against search-execution time silently re-anchors every replayed/eval question to today; LongMemEval ships `question_date` precisely because expressions are written relative to when the question was asked. Zep's `reference_time` is the same idea one level down (per-utterance).

**Conventions adopted** (deterministic, Duckling-EN-compatible except where noted; Duckling/Time/EN/Rules.hs verified):
- Calendar-unit semantics for `N <unit> ago` when unit ≥ day: snap to the containing unit shifted back N (Duckling `predNth(-1)` on the containing cycle). For sub-day units, exact offset `[t−N·u, t−N·u+u)`.
- `this <unit>` = containing unit ∩ (−∞, t_q] (week-to-date, etc.). `last <unit>` = immediately previous complete unit (ISO-8601 weeks, calendar months/years). `next <unit>` = following complete unit.
- Weekday names: bare `"<DOW>"` → most recent such day **≤ day(t_q)** (past-biased for recall; Duckling instead resolves bare DOW to the nearest-future occurrence — noted deviation, chosen because Verbatim queries are overwhelmingly recall). `"last <DOW>"` → most recent `<DOW>` strictly before day(t_q). `"this <DOW>"` → the `<DOW>` of the ISO week containing t_q. `"next <DOW>"` → the `<DOW>` of the next ISO week (Duckling `ruleNextDOW`: `intersect dow (cycleNth Week 1)`).
- Day-parts (local, constants tunable; "last night" verified = 6h before day-start in `ruleLastNight`): morning [04:00,12:00), afternoon [12:00,18:00), evening [18:00,24:00), "last night" [day_start−6h, day_start), "late last night" [day_start−3h, day_start).

## 2. The deterministic resolution table

`resolve(phrase, t_q, ctx) → (interval [a,b) | SET{intervals}, status, qual)`. `d(t)` = local day-start of t; `W(t)` = ISO Monday of t's week; `M(t)` = first of t's month; `Y(t)` = Jan 1 of t's year. All intervals half-open.

| Phrase (regex family) | Anchor | Interval [a, b) | qual |
|---|---|---|---|
| ISO date / "March 2024" / "March 5" | — | stated grain interval (year→decade, month→month, day→day) | pin |
| "N {sec\|min\|hour}s ago" | A_q | [t_q − N·u, t_q − N·u + u) | pin |
| "N days ago" | A_q | [d(t_q)−N d, d(t_q)−(N−1)d) | pin |
| "N weeks\|months\|years ago" | A_q | containing unit shifted −N, full unit interval | pin |
| "today" | A_q | [d(t_q), d(t_q)+1d) | pin |
| "yesterday" / "tomorrow" | A_q | [d(t_q)∓1d shifted days) | pin |
| "this week" | A_q | [W(t_q), t_q) | pin |
| "last week" | A_q | [W(t_q)−7d, W(t_q)) | pin |
| "next week" | A_q | [W(t_q)+7d, W(t_q)+14d) | pin |
| "this month" / "last month" / "next month" | A_q | containing/prev/next calendar month | pin |
| "this year" / "last year" | A_q | [Y(t_q), Y(t_q)+1y) etc. | pin |
| "the past\|last N days" (duration) | A_q | rolling [t_q − N·d, t_q) — NB: Duckling treats "past week" as rolling duration, distinct from "last week" | pin |
| "a couple\|few days ago" | A_q | [d−4d, d−2d) | approx |
| "<DOW>" bare | A_q | most recent DOW-day ≤ d(t_q) | approx (past-vs-future convention) |
| "last <DOW>" | A_q | most recent DOW-day < d(t_q) | pin |
| "this <DOW>" | A_q | DOW of ISO week of t_q | pin |
| "next <DOW>" | A_q | DOW of next ISO week | pin |
| "in <Month>" / "last <Month>" | A_q | most recent `<Month>` with month ≤ containing month of t_q | pin; approx if t_q inside that same month (co-speaker ambiguity w/ previous year) |
| "this morning" | A_q | [d+4h, d+12h) ∩ (−∞, t_q] | pin (bounds tunable) |
| "this afternoon" / "this evening" | A_q | [d+12h, d+18h) / [d+18h, d+24h) ∩ (−∞, t_q] | pin |
| "last night" / "tonight" | A_q | [d−6h, d) / [d+18h, d+24h) | pin |
| "earlier (today)" | A_q | [d(t_q), t_q) | pin |
| "earlier in this conversation / this session" | A_sess | [t_s, t_q) | pin |
| "in our last\|previous conversation" | session tbl | previous session's [start, end); abstain if no session table | pin |
| "recently" / "lately" / "the other day" | A_q | [t_q − RECENT_DAYS·d, t_q), RECENT_DAYS=7 default | approx — ranking cue, not a hard eligibility filter |
| "always/ever/so far" | — | no constraint | none |
| "currently/now" (state) | A_q or A_sys | point {t_q}: `valid_from ≤ t_q < valid_until` (open-set predicate) | pin |
| "when/what time/how long ago" (open) | — | no constraint; emit fact's own timestamps into pack | none |
| unresolvable / missing t_q | — | no constraint + warn | none |
| conflicting cues ("yesterday in March" while t_q=June) | — | abstain + warn (existing behavior) | none |
| multi-cue non-conflicting ("Monday morning") | A_q | intersect grain-wise: day(Mon) × [4h,12h) | pin |

**Worked example** (verified executable, `/tmp` prototype): t_q = 2026-09-23 15:30 Wed → "last week" ↦ [2026-09-14, 2026-09-21); "3 days ago" ↦ [2026-09-20, 2026-09-21); "in March" ↦ [2026-03-01, 2026-04-01); "last Friday" ↦ [2026-09-18, 2026-09-19); "this morning" ↦ [2026-09-23 04:00, 12:00); "last night" ↦ [2026-09-22 18:00, 24:00). At t_q = Monday 23:30, "this week" ↦ [that-Mon 00:00, 23:30) — a 23.5h interval, correctly not the whole week. At t_q = Friday 09:00, "last Friday" ↦ the Friday 7 days back (strict-previous convention).

## 3. Write-path interaction (how rows get their times)

On `Memory.add`:
- `said_at` = message timestamp (always present in real data; NULL only if caller omits it → row excluded from windowed lanes, still eligible all-time).
- `valid_from/valid_until` = world-time truth interval. Default `[said_at, sentinel)` for assertions; for utterance/event rows `valid_from = said_at`. Content-time phrases inside the message ("I moved to Lisbon *last March*") are resolved against `said_at` (the same table, anchor = message ts = Zep `reference_time`) and stored with `t_qual='pin'`; unresolved → `t_qual='unresolved'`, `valid_from=said_at`.
- `rec_from` = monotonic append seq; `rec_until` = sentinel. On `forget`/supersede: never overwrite (hard rule) — close `rec_until` (store-side retraction) and/or `valid_until` (world ended: "I moved — no longer in Lisbon") on a new row, per Graphiti's `expired_at` vs `invalid_at` split. Graphiti fields confirmed from source (`graphiti_core/edges.py`): `created_at`, `expired_at` "when the node was invalidated", `valid_at` "when the fact became true", `invalid_at` "when the fact stopped being true", `reference_time` "reference timestamp from the episode".

## 4. SQLite schema and indexes

```sql
CREATE TABLE memories (
  id INTEGER PRIMARY KEY,
  payload BLOB NOT NULL,            -- retained bytes (byte-pin target)
  said_at    INT,                   -- minutes-since-epoch, naive writer clock
  tz_off     INT,                   -- minutes, optional, metadata only
  valid_from INT NOT NULL,          -- world-time start (default = said_at)
  valid_until INT NOT NULL DEFAULT 2147483647,   -- sentinel, NOT NULL (index-friendly)
  rec_from   INT NOT NULL,          -- append seq
  rec_until  INT NOT NULL DEFAULT 2147483647,
  t_qual     TEXT NOT NULL DEFAULT 'pin'         -- pin|approx|unresolved
);
CREATE INDEX idx_mem_said   ON memories(said_at);
CREATE INDEX idx_mem_vuntil ON memories(valid_until);   -- closed-set lane
-- optional (quality profile): closed-intervals-only rtree
CREATE VIRTUAL TABLE rt2c USING rtree_i32(id, valid_from, valid_until);
-- maintained for rows WHERE valid_until < 2147483647 (~20% of store)
```

Quantization: all valid axes in **minutes** (rtree_i32-safe until year ~6053; sub-minute precision is meaningless at dialogue granularity). Sentinels instead of NULL so `>` predicates stay index-sargable.

**Queries** (all measured below):
- Lane (recall): `SELECT id FROM memories WHERE said_at BETWEEN :a AND :b` (+ `OR (valid_from < :a AND valid_until > :b)` spanning clause, closed rows only, optional).
- Lane (changed/superseded): `SELECT id FROM memories WHERE valid_until BETWEEN :a AND :b`.
- State: no lane query — flag candidate rows `valid_until = sentinel` at eligibility.
- Eligibility per candidate `f` for resolved `[a,b)`: `(f.said_at ∈ [a,b)) OR (f.valid_from < b AND f.valid_until > a) OR t_qual='unresolved'` — permissive overlap, checked on the fetched row; costs ~1.4 µs/row.
- As-of: add `rec_from ≤ :S AND rec_until > :S` on the same row check.
- OPTIONAL true interval-overlap on closed rows: `SELECT id FROM rt2c WHERE valid_from <= :b AND valid_until >= :a`.

## 5. Measured latency and size (4-core Ubuntu, CPython+sqlite3, /tmp prototype — measured, not quoted)

Corpus: facts uniform over 5y of `valid_from`/`said_at`; 80% open (`valid_until`=sentinel), 20% closed with log-uniform durations 1h–90d; 5% `rec_until`-closed. 600 queries per cell, warmup 50.

**Point-in-bitemporal containment** `v_from≤T<v_until ∧ r_from≤S<r_until` (T uniform, S latest 70%):

| strategy | 10k mean | 10k p95 | 100k mean | 100k p95 | avg hits |
|---|---|---|---|---|---|
| A B-tree(v_from) + row filter | 2.02 ms | 2.77 | **28.8 ms** | 32.0 | ~37k |
| B B-tree(v_until) + row filter | 2.01 ms | 2.77 | 28.9 ms | 32.1 | ~37k |
| C rt2 + join recheck | 2.38 ms | 4.52 | 52.0 ms | 99.1 | ~36k |
| D rt4 + join recheck | 2.40 ms | 4.49 | 46.1 ms | 88.8 | ~37k |
| E open-partition + rt2c | 3.28 ms | 6.60 | 49.1 ms | 96.9 | ~35k |

Lesson: at ~37% selectivity EVERYTHING is slow because the rowid→row join of ~36k hits dominates (rtree probe alone = 5.4 ms at 100k). **Interval containment is a filter, not a lane.**

**Lane queries (id-only, no join) — the ones Verbatim actually needs:**

| query | 10k | 100k | hits@100k |
|---|---|---|---|
| said_at∈1d (btree) | 0.007 ms | **0.020 ms** | 54 |
| said_at∈7d (btree) | 0.015 ms | **0.105 ms** | 383 |
| said_at∈30d (btree) | 0.047 ms | **0.424 ms** | 1,634 |
| said_at∈1yr (btree) | 0.465 ms | 4.96 ms | 17,935 |
| same windows via rt2 (id-only) | 0.07–0.53 ms | **1.41–6.26 ms** | same |
| open-set @now (rt4 id-only) | 2.41 ms | **25.3 ms** | 80,295 |
| overlap 7d window (rt4 id-only) | 1.24 ms | **13.1 ms** | 41,343 |
| v_until∈7d unindexed seq scan | 0.38 ms | 5.09 ms | 75 |
| eligibility: fetch+check 40 candidates | 0.051 ms | 0.055 ms | ~30 pass |

Arithmetic check: 100k facts / 2.63M min × 10,080 min(7d) = 384 expected hits — measured 383 ✓. Sequential-scan floor at 100k = 5.09 ms → 51 ns/row; that's the budget ceiling for any filter approach.

**Sizes at 100k rows** (dbstat, 4KB pages): `idx_vfrom` 1.25 MB (≈12.5 B/row), `idx_vuntil` 1.25 MB, rt2 5.29 MB (53 B/row), rt4 4.73 MB (47 B/row), open-partition+idx 3.3 MB. Recommended default profile (said_at + v_until B-trees) ≈ **25 B/memory ≈ 2.5 MB/100k**; +9 B/row amortized if rt2c is shipped.

## 6. Query-time policy — what it adds over known-at

Existing code: known-at = `rec_from ≤ S < rec_until` (recorded-axis as-of). Resolution adds two query-side artifacts:
1. a hard interval `[a,b)` for lane generation + eligibility (`t_qual='pin'` → filter; `'approx'` → soft window for ranking, e.g. proximity boost `b_t = min(B_MAX, b0·exp(−gap/τ))` with `gap = distance(candidate's said_at/valid interval from [a,b))`, τ=1 window-width, b0 and B_MAX bounded by the existing ±5–10% boost envelope — this is the natural place for it, fused post-RRF);
2. display metadata for the pack ("about 3 days ago · 2026-09-20") and the abstain/warn flag.

Net per-query cost added to the pipeline at 100k: resolver ~10 µs (regex table) + lane ≤0.5 ms typical + eligibility 0.055 ms ≈ **<1 ms** — inside the tens-of-ms p95 budget with room to spare even in the quality profile.

## 7. Comparison tables

**Resolvers (for reference; ship a built-in rule table, no deps):**

| resolver | deps | semantics fit | verdict |
|---|---|---|---|
| built-in rule table (this note) | none | exact match, deterministic | **ship** |
| dateparser | dateutil/regex wheels | decent conventions, library weight + nondeterminism risk | reject |
| parsedatetime | stdlib-ish | loose conventions, unmaintained cadence | reject |
| Duckling | Haskell runtime, MBs | reference semantics — borrow conventions, not code | conventions: ship |
| HeidelTime/SUTime | JVM | overkill | reject |

**Index candidates:**

| candidate | 100k cost | verdict |
|---|---|---|
| B-tree said_at (origin-in-window lane) | 0.02–0.42 ms | **ship** |
| B-tree valid_until (closed lane) | ~0.4 ms | **ship** |
| per-candidate eligibility check | 0.055 ms/40 | **ship** |
| rt2c closed-rows-only rtree | ~1 ms when used | **optional** (quality profile: true interval-overlap on closed facts) |
| rt4 4-dim rtree both axes | 25.3 ms open-set | **reject** for default; join-dominated, non-selective |
| point-in-time lane (any impl) | 29–52 ms | **reject** — ~37% selectivity makes it useless as a lane |

## 8. Confirmed constants under review / new constants

- `RECENT_DAYS = 7` — new; ranking-cue horizon for "recently". Confidence: medium — sweep 3–14 on LoCoMo temporal questions.
- Day-part bounds (4/12, 12/18, 18/24, night = 6h pre-midnight) — from Duckling EN rules; confidence medium — Duckling's exact morning bound is 4am-ish convention, treat as tunable.
- `said_at`, `valid_*`, `rec_*` schema with i32-minute quantization and `2147483647` sentinel — measured-safe.
- Anchor precedence A_exp > A_q > A_sess > A_sys — evidence: LongMemEval `question_date` + Zep `reference_time`, both primary.
- No change recommended to RRF k=60 or boost alphas from this question's scope (temporal boost slots into existing bounded-boost envelope).

## 9. Coverage check vs assignment

1. ✓ precedence rules + ambiguity (§1–2, §4 policy).
2. ✓ bi-temporal layout + indexes + 100k latency (§4–5, measured).
3. ✓ known-at vs query-time resolution (§6).
4. ✓ edge cases incl. timezone-free, cross-session "this morning", abstain+warn (§2 table, §0.4, §4 NULL policy).
5. ✓ deterministic resolution table (§2).
UNRESOLVED: exact day-part hour bounds beyond Duckling's "last night" (6h) — morning/afternoon/evening edges are convention, not standard; what would close it = a LoCoMo-side sweep over 2-3 boundary schemes.
