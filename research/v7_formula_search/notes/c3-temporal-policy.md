# c3-temporal-policy — temporal ranking policy for Verbatim

Scope: how time should influence `Memory.search` ranking after the filed correctness bugs are fixed. Three complete designs compared — (a) hard window + time-bucket spread (the existing temporal lane), (b) additive vs multiplicative recency decay (exponential vs linear; Mem0's wide band vs Hindsight's 365-day linear), (c) state-key current/historical supersession — plus interval algebra for valid-time containment and dual-time (event vs recorded) semantics. Recommendation at the end is a composite, not a single winner.

---

## 0. Definitions used throughout

Every memory carries two time axes (bitemporal model, Snodgrass-style; Zep paper §III makes the same split explicit — timeline T for event chronology, T′ for ingestion order):

- **Valid time** `valid_from, valid_to`: when the stated fact is true in the world. A point fact has `valid_from = valid_to`; an ongoing state has `valid_to = ∞` (open interval); a coarse date ("in 2015") is an interval spanning the period.
- **Record time** `recorded_at, expired_at`: when Verbatim learned/stored it and when it was superseded (`expired_at = ∞` while live). In Verbatim terms, `recorded_at` ≈ the turn's timestamp; `expired_at` is set only by supersession.
- `days_ago(m) = (t_query − effective_time(m))/86400`, where `effective_time = COALESCE(valid_from, recorded_at)`. NULL time ⇒ neutral signal 0.5, never a guess (Hindsight's convention, `retrieval.py` L646–652).

Constraint restated: a temporal signal may reorder near-tied candidates but must not let a weaker match outrank a much stronger one. Operationally this means the temporal contribution must be a **bounded multiplicative factor** on the base score — additive terms become unbounded *relative* multipliers on weak bases (a +0.2 add on base 0.10 is a 3× boost; on base 0.90 it is 1.2×).

---

## 1. Design (a): hard window + time-bucket spread — the temporal LANE

This is a *candidate-generation* mechanism for queries with explicit time reference ("last spring", "in 2023"), not a general ranking policy. Verbatim already ships it ("tightest covering interval first, interleave buckets"); Hindsight's TEMPR temporal arm is the same design and gives us verified constants.

**Formula (verified in `hindsight-api-slim/hindsight_api/engine/search/retrieval.py`):**

1. Parse the query's temporal expression into window `[s, e]` (`query_timestamp` anchors relative phrases; same as Mem0's `reference_date`).
2. Candidate pool: interval-overlap predicate (see §4), ANN/lexical-ranked, `_TEMPORAL_POOL_SIZE = 60` per lane (L404).
3. Coverage spread: window → `_TEMPORAL_COVERAGE_BUCKETS = 8` equal buckets; round-robin take the tier-`i` best item from every populated bucket, similarity-desc within a tier; keep `_TEMPORAL_ENTRY_POINTS = 10` (L404–406, 414–460). Degenerate case (all dates in one bucket) collapses to plain rank order.
4. In-window proximity signal: `temporal_proximity = 1 − min(|t_mid(m) − mid_window| / (span/2), 1.0)` (L650) — triangular, 1.0 at window center; folded later as `temporal_boost = 1 + 0.2·(prox − 0.5)` ∈ [0.90, 1.10].
5. Results enter RRF (k=60, `fusion.py` L29) as one peer lane among 4.

**Worked example (measured in my prototype, window = last 92 days, pool of 8, limit 5, 8 buckets):** a 4-fact "Lisbon trip" cluster at days −85…−80 (bases 0.88–0.82) plus dentist −60 (0.70), piano −40 (0.68), review −10 (0.66), job change −50 (0.64). Output order: Lisbon-early, dentist, piano, review, job-change — the dense bucket contributes one item per tier while every populated slice is represented. Exactly the intended behavior: coverage over concentration.

**FALSE-CURRENT failure mode:** the window satisfies *mention* time, not *currency*. "What did Alice do in 2023?" happily returns "Alice works at Vandelay (stated 2023, superseded 2024)" inside the window — true history, fine. The failure is when the window lane *replaces* currency checks: "where did Alice work last year?" surfaces the stale employer with a bucket-spread slot while the current fact sits outside the window and never competes. The design is false-current-*agnostic*: it never asks whether the fact is still true. It also cannot answer "what is true now" at all — degenerate window `[today, today]` matches almost nothing or everything depending on boundary convention.

**Constraint compliance:** excellent by construction — window membership is a filter/lane, and the only score influence is the bounded ±10% triangular proximity. Temporal *exclusion* (hard filter) can however hide the strongest match that happens to sit just outside the window; mitigate by keeping window lanes *admissive* (candidates also survive via other lanes' RRF entries — the lane is additive recall, not a gate on other lanes).

**Verdict: ship as the temporal lane** (it is not, and never was, a substitute for currency handling). Confidence: high — identical mechanics verified in Hindsight code + docs. What would change it: LoCoMo replay needs `query_timestamp` plumbing through `Memory.search` (Hindsight and Mem0 both expose this parameter; Verbatim's "unresolved relative time phrases" bug must be fixed for the lane to fire at all on open-domain queries).

---

## 2. Design (b): recency decay — additive vs multiplicative, exponential vs linear

### 2.1 Candidate formulas (all verified against source code/docs)

| Variant | Age→signal r(d) | Fold into base score | Bounds | Source |
|---|---|---|---|---|
| **Hindsight linear** (default) | `r = clamp(1 − d/365, 0.1, 1.0)` | `final = base · (1 + 0.2(r − 0.5))` | boost ∈ [0.92, 1.10] | `reranking.py` L275–276, L408 |
| **Hindsight exponential** (opt-in) | `r = 0.5^(d/H)`, `H=90d` | same | boost → [0.90, 1.10] as r→(0,1] | `reranking.py` L271–273; commit a2166ee exposes `HINDSIGHT_API_RECENCY_DECAY_HALFLIFE_DAYS=90`, `..._LINEAR_WINDOW_DAYS=365` |
| **Mem0 platform decay** | scaler `s ∈ [0.3, 1.5]` from last-20 *access* timestamps (not write age): ~1.5 just-accessed, 1.2–1.4 today, 0.4–0.6 weeks-idle, 0.3 floor | `final = base · s` | 5× spread | mem0 docs `memory-decay.mdx` + launch blog; pool widened to `top_k×3`, floor 50; OSS SDK rejects `decay=True` (`notices.py`) |
| **Li & Croft 2003 prior** | `P(d) = γ·e^(−γ·age)` on document age | score multiplier (prior) | unbounded shape, tune γ | CIIR ir-277.pdf §3.3.1; Efron & Golovchinsky make γ query-dependent |
| **Additive** | `r` as above | `final = base + α·r` | α fixed, relative effect unbounded | Verbatim's provisional "+5–10% boosts" are effectively this family |

Hindsight's full combined scoring (`apply_combined_scoring`, `reranking.py` L367–412): `combined = CE_norm · recency_boost · temporal_boost · proof_boost` with α = 0.2, 0.2, 0.1 → worst-case multiplier ≈ [0.79, 1.27]. Note the deliberate design comment at L331–353: when the reranker is a passthrough (all CE scores identical), they *seed* the base from RRF rank so the boosts modulate a real signal — the same failure Verbatim has filed as "min-max fusion tie-collapse". **Bounded multiplicative on a degenerate base = pure recency sort.** The tie-collapse fix is a prerequisite, not a nice-to-have.

### 2.2 Measured comparison — FALSE-CURRENT scenario

Prototype scenario: stale `m1 "Alice works at Vandelay"` (base 0.90, 300d old, superseded) vs current `m2 "Alice works at Meridian"` (base 0.78, 10d), plus three distractors. Computed results:

| Policy | m1 final | m2 final | Correct order? | Protection ratio (max base gap time can flip) |
|---|---|---|---|---|
| Hindsight linear α=0.2 | 0.9·0.92=**0.842** | 0.78·1.097=**0.854** | yes, margin 1.4% | 1.196× |
| Exponential h=90 α=0.2 | 0.828 | 0.846 | yes, margin 2.2% | 1.180× |
| Exponential h=120 α=0.2 | 0.841 | 0.849 | yes, margin 1.0% | 1.164× |
| Additive α=0.2 | 0.936 | 0.975 | yes — but weak-base amplification: base 0.50 got +0.19 (1.38×) vs base 0.90's +0.036 (1.04×) | unbounded in ratio terms |
| Li&Croft wrap `0.85+0.3·e^(−d/120)` | 0.787 | 0.878 | yes, comfortable | 1.353× ceiling |
| **Mem0 access-recency** | 0.9·1.272=**1.145** | 0.78·1.078=**0.841** | **NO — FALSE-CURRENT** | 5× spread flips anything ≤5× |

Two findings matter more than the table:

- **Mem0's access-recency is self-reinforcing staleness.** A stale-but-popular fact (last retrieved 5d ago) gets scaler 1.27, not 0.3 — retrieval resets its clock, so the wrong answer that keeps being served never decays. Measured: m1 wins 1.145 vs 0.841. Wide bands (0.3–1.5) also *violate the constraint*: a 5× spread lets recency dominate any plausible lexical gap. Reject for default; the mechanism (retrieval-count reinforcement) is optional-quality at best and needs an age term, not just last-access.
- **Bounded α=0.2 protects a ~20% base gap only.** My scenario was deliberately knife-edge (0.78 vs 0.90 = 15% gap); the current fact barely won. Any stale fact ≥1.2× stronger on text match still wins under any ±10% bounded boost — decay alone *cannot fix* FALSE-CURRENT; it can only bias it. That is the argument for (c).

### 2.3 Linear vs exponential — the real difference

- Linear-365→0.1 has a **hard cutoff**: d=365 and d=3650 are identical (r=0.1). Exponential keeps resolving old ages (300d→0.10, 1000d→0.0004 → boost 0.90). Under Verbatim's byte-pin rule, stale quotes must remain reachable, so a floor is required either way — the question is only whether the *signal* floors or the *boost* does.
- Exponential is the better default: smooth, no cliff, "half-life" is interpretable and matches the filed intuition that agent memories age like recency-decay processes (Li&Croft; Efron). Hindsight ships exponential h=90 as an option; I recommend **h=120d** for Verbatim — LoCoMo turns cluster in 1–12 month ages, and h=90 already hits r≈0.06 at 365d, i.e., near-maximum penalty inside the dataset's own horizon; h=120 puts neutral at 4 months and leaves residual slope beyond a year.
- Keep Hindsight's exact bounded fold: `boost = 1 + α(r − 0.5)`, α=0.2 → [0.90, 1.10]. Neutral (no timestamp) = r=0.5 → boost exactly 1.0.

### 2.4 Constraint compliance

Multiplicative bounded = the only compliant fold. Rule: `final = base · Π_i(1 + α_i(s_i − 0.5))` with Σ-adjusted cap — stacking recency (±10%), in-window proximity (±10%), proof (±5%), and current-flag (±7.5%, §3) yields a total temporal+evidence envelope of **[0.78, 1.34]** worst case, and ≈[0.84, 1.27] typical (all-high or all-low is rare). Recency *alone* must never exceed ±10%; publish that bound in the docs the way Hindsight does.

**Verdicts:** ship `r = 0.5^(d/120)` + `×(1+0.2(r−0.5))` on default; ship linear-365 as config (Hindsight shows users ask); reject additive (unbounded relative amplification on compressed bases — measured); reject Mem0 access-recency on default (self-reinforcing staleness + 5× band violates the cap; optional only, and only if re-parameterized to ≤±10% and based on age not access). Confidence: high on fold mechanics (code-verified); medium on H=120 (derived, not benchmarked — the Verbatim LoCoMo rerun is what closes it; grid {60,90,120,180} against the fixed retrieval).

---

## 3. Design (c): state-key supersession — "what is true now"

The only design that treats currency as *structure* rather than score. Prior art is Graphiti's bitemporal edges, code-verified:

- Edge fields (`graphiti_core/edges.py` L271–282): `valid_at`, `invalid_at`, `expired_at`, `reference_time`.
- Invalidation rule (`utils/maintenance/edge_operations.py` L545–571): for same-predicate candidates with overlapping validity, if `new.valid_at > old.valid_at` then `old.invalid_at := new.valid_at` and `old.expired_at := now` — the old edge is *truncated*, not deleted (non-lossy), and non-overlapping intervals are explicitly skipped (L553–561: already-ended-before-new-began ⇒ no contradiction).
- Query side (`search_filters.py` L180–260): `invalid_at`/`expired_at` filters express as-of semantics; current facts = `invalid_at IS NULL AND expired_at IS NULL`.

**Verbatim mapping.** Typed facts and observations already carry `(subject/entity, predicate/state_key)`. Add to the fact row: `state_key` (32-bit hash of normalized entity+predicate), `valid_from`, `valid_to` (NULL=open), `superseded_by` (rowid), `expired_at`. Write path: on add, one indexed lookup `WHERE state_key=? AND valid_to IS NULL`; on overlap, set `valid_to = new.valid_from`, `superseded_by`, `expired_at`. Cost: O(log N) — sub-ms per write.

**Retrieval formula.** Two layers:
1. Rank: `final = base · decay_boost · (1 + α_cur)` for `is_current(m)` facts on *state-valued* keys only (α_cur = 0.075, i.e. half of recency's range — currency is decisive *per-key*, small globally). Never applied to episodic memories (events don't supersede).
2. Pack: within each state_key present in the pack, order current first and include ≤1 predecessor labeled `previously valid [from, to]` — the LoCoMo "previously/X used to" questions need the predecessor surfaced, and the byte-pin rule is safe: the label comes from stored interval metadata, not generated text.

**FALSE-CURRENT failure mode (of this design — it's not free):** **FALSE-HISTORICAL** — wrongly superseding non-exclusive predicates. "Alice likes coffee" and "Alice likes tea" under one `likes` key ⇒ the second erases the first. Graphiti handles this via LLM edge resolution; Verbatim is model-free on the add path, so exclusivity must be declared per-predicate (singleton-valued keys like `employer`, `lives_in`, `phone` supersede; multi-valued `likes`, `visited`, `owns` append — never supersede). Missed supersession (synonym keys: `works_at` vs `employer`) is the softer failure — stale fact merely loses its penalty, which decay partially covers.

**Constraint compliance:** the α_cur bump is bounded multiplicative (same cap family); the *strong* form — "current always wins within key" — is enforced at pack order, not by score, so a much stronger textual match on a *different* fact is unaffected. A stale fact with a 5× better lexical match to a *different* query still wins; within its own key, it correctly loses to the current value regardless.

**Verdict: ship** — this is the actual fix for FALSE-CURRENT; decay is the tie-breaker, supersession is the mechanism. Confidence: high on mechanics (Graphiti is production-proven at Zep scale); medium on predicate-exclusivity rules (needs a curated singleton-predicate list — small, reviewable). What would change it: showing LoCoMo cat-3/4 gains collapse if superseded facts are hidden entirely → keep predecessor-surfacing in pack.

---

## 4. Interval algebra for valid-time containment

Allen's 13 relations {before, meets, overlaps, starts, during, finishes, equals + 6 inverses}. The temporal lane's candidate predicate is interval overlap:

`overlap(A, Q) = A.from ≤ Q.end AND A.to ≥ Q.start`

I verified the enumeration computationally: under closed-interval semantics this admits **11 of 13** relations (excludes only `before`, `after`; *includes* `meets`/`met-by`, i.e. touching endpoints count). Under half-open `[from, to)` semantics — the better convention for valid-time (a fact true "until March" is false in March) — `meets`/`met-by` drop out ⇒ **9 of 13**. Hindsight's SQL (L575–584) is closed-interval overlap plus `BETWEEN` point checks for the `mentioned_at`-only and single-ended cases; Verbatim should adopt **half-open** semantics for `valid_to` and closed for bucket windows, and say so in the schema doc — boundary equality ("what was true exactly on March 1") is where off-by-one bugs live.

Containment variants needed:
- **Overlap** (default temporal lane): any intersection.
- **Point-in-valid** ("was X true on Mar 1?"): `valid_from ≤ t < valid_to` — O(log N) on the `valid_from` index.
- **Within** ("during 2023"): `valid_from ≥ s AND valid_to ≤ e` — stricter than overlap; use only when the query says "the whole of".

## 5. Dual-time query semantics

Two independent query axes — keep them explicit in the API, never conflate in one `timestamp` field:

| Question | Axis | Predicate |
|---|---|---|
| "What is true now?" / "Where does Alice work currently?" | valid, t=now | `valid_to IS NULL` per state_key, latest `valid_from` |
| "What did Alice do in 2023?" | valid interval | overlap(valid, 2023) |
| "What did you learn this week?" | record | `recorded_at ∈ week` |
| "What did you know on Tuesday?" (audit/replay) | record-time as-of | `recorded_at ≤ t AND expired_at > t` |
| "What was believed true last June?" | bitemporal as-of | both: `valid` overlap June AND `recorded_at ≤ t' AND expired_at > t'` |

`reference_date`/`query_timestamp` parameter (both Mem0 and Hindsight expose it) anchors relative phrases and replays — essential for deterministic regression tests of temporal ranking.

---

## 6. Complexity, index size, latency (4-core, SQLite)

Per-candidate temporal scoring measured at **0.34 µs** (decay + two bounded boosts, pure Python). Fused pool ≤300 ⇒ **≤0.10 ms**; at the 32-candidate rerank pool ⇒ **0.01 ms**. Bucket-spread over a 400-row window pool ⇒ **0.18 ms**. B-tree range probe on a 100k-row `valid_from` index (in-memory SQLite, ×1000) ⇒ **0.01 ms/query**; log₂(100k)≈17 page descents. Totals added to existing pipeline: **<0.5 ms at 10k, <2 ms at 100k** — all three designs together are under the noise floor of the measured 46/85 ms search p50/p95. Selection is a quality decision, not a latency one.

Resident cost per memory: `valid_from/valid_to` int64×2 (16B) + `recorded_at/expired_at` (16B) + `state_key` uint32 + `superseded_by` rowid (12B) + one composite index `(state_key, valid_to)` ≈ 24–40B/row ⇒ **~70–85 B/memory ⇒ ≈7–9 MB per 100k**. A ring buffer of last-N access timestamps (Mem0-style, if ever enabled) costs N×8B — skip it on default.

---

## 7. Recommended composite (constants)

1. **Temporal lane (keep, fix):** overlap predicate (half-open `valid_to`), 8 buckets, pool 60→10 per lane, triangular proximity `1−|t−mid|/(span/2)` → `×(1+0.2(p−0.5))`. Requires `query_timestamp` param.
2. **Recency (default profile):** `r = 0.5^(d/120)`, `boost = 1+0.2(r−0.5)` ∈ [0.90,1.10]; neutral 0.5 when undated; score coarse calendar spans from period END capped at 0.5 (Hindsight's #3893 rule — "in 2015" is aged from Dec 31 2015, never boosted).
3. **Supersession:** singleton-valued state_keys only; write-path truncation `valid_to := new.valid_from, expired_at := now`; rank `×1.075` current bump; pack shows current first + ≤1 labeled predecessor.
4. **Caps:** per-signal ≤±10%, combined temporal envelope ≤1.35×; never additive; document the bound.
5. **Mem0-style access-recency:** optional "quality" profile only, re-parameterized to ±10% and age-based — as shipped upstream it demonstrably keeps stale-but-popular facts on top (measured 1.145 vs 0.841 FALSE-CURRENT).
