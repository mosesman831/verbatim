# c1-entity-resolution — alias/merge formula without an LLM

**Scope.** The question is how a mention or an existing entity row merges with another entity
(alias resolution / dedupe of the entity table), deterministically and conservatively. This is
a **write-path / maintenance concern** (runs at retain time and as a background consolidation
job), not part of query-time ranking. It sits in the **entity lane** of the pipeline: it decides
what `entities` rows exist and which surface forms map to them.

---

## 1. The four candidate formulas, verbatim

### (a) Hindsight (vectorize-io/hindsight) — verified in repo

Source: `hindsight-api-slim/hindsight_api/engine/entity_resolver.py` @ `a544b196`.

Candidate generation (lines ~619–700): per mention text, a `CROSS JOIN LATERAL` on the
pg_trgm GIN index returns the best `entity_resolution_max_candidates` (default **200**,
`DEFAULT_RETAIN_ENTITY_RESOLUTION_MAX_CANDIDATES`, config.py:1244) entities where
`LOWER(canonical_name) % LOWER(query_text)` under `pg_trgm.similarity_threshold = 0.15`
(`DEFAULT_ENTITY_TRGM_SIMILARITY_THRESHOLD`, config.py:1367). Oracle backend substitutes
UTL_MATCH Jaro-Winkler ≥ 70/100 (comment: "≈ pg_trgm 0.15").

Scoring (lines 1053–1085), per candidate:

```
score = 0.5 * SequenceMatcher(mention.lower, canonical.lower).ratio()
      + 0.3 * ( |nearby_entities ∩ cooc_partners(candidate)| / |nearby_entities| )
      + 0.2 * max(0, 1 - |Δdays|/7)          # only if Δdays < 7
merge iff best_score > 0.6                    # strict >
```

Auxiliary constants: `_SCORING_YIELD_EVERY = 256` ("256 candidates ≈ 13 ms" → ~50 µs per
SequenceMatcher call, their comment); intra-batch union-find of new names at trigram
Jaccard ≥ **0.5** (`intrabatch_merge_similarity`); `resolve: false` mentions are exact-
case-insensitive only; label entities never fuzzy-merge. Co-occurrence is *recall of the
mention's nearby set against the candidate's partner set* — asymmetric, no rarity weighting,
never negative.

### (b) Dedupe (dedupe.io) — Fellegi-Sunter-flavored supervised matching

From docs.dedupe.io/dedupe.io how-it-works: per-field distance (AffineGap distance, a Hamming
variant), then **regularized logistic regression** over labeled duplicate/non-duplicate pairs
turns weighted field distance into P(duplicate). **Active learning** picks the pair nearest
P≈0.5 for human labeling. Blocking is mandatory — their own arithmetic: 1k records →
499,500 pairs; 100k records → ~5×10⁹ pairs ("six days at 10k comparisons/sec"). Under the hood
this is Fellegi-Sunter with *learned* weights instead of hand/EM-estimated m/u.

Fellegi–Sunter (JASA 1969, doi 10.1080/01621459.1969.10501049): each field comparison γᵢ takes
weight log(mᵢ/uᵢ) — mᵢ = P(agreement|match), uᵢ = P(agreement|nonmatch); the pair score is the
sum; a **three-way rule** orders the comparison space by likelihood ratio and applies TWO
thresholds T_μ (link, where cumulative false-link prob ≤ μ) and T_λ (non-link, where cumulative
false-nonlink ≤ λ): link / possible-link / non-link. The abstain band is native to the theory.

### (c) Senzing — principle-based deterministic matching

From Senzing's deep-dive (zendesk 360045732894): no training; ~30 prebuilt **principles**
combining names with feature evidence under three **expected-behaviors** per attribute:
**Frequency** (how many entities share a value: SSN≈1, address≈few, DOB≈many), **Exclusivity**
(can one entity hold >1 value), **Stability** (does the value persist). Comparators emit a
6-level verdict (same/close/likely/plausible/unlikely/not-same). Decisions are three-way:
**same / possibly-same / related** — e.g. "close name + shared frequency-one feature → same";
"close name + shared freq-few feature but contradictory exclusive feature → related, *not
same*". Generic values (a phone shared by hundreds) are excluded from candidacy. Decisions are
re-evaluated as statistics evolve.

### (d) Alias sieve (recommended composite)

Canonical form → exact map, then a Fellegi-Sunter additive match-weight over hand-set,
deliberately conservative m/u levels, with Senzing's generic-value suppression inside the
co-occurrence feature. No labels, no model, deterministic, every constant auditable.

---

## 2. Recommended formula (ship as default profile)

### 2.1 Canonicalization (free, deterministic)

```
canon(s) = lowercase( NFKD(s) minus combining marks )
           strip possessive "'s";  punctuation → space; collapse whitespace
```

### 2.2 Features → comparison levels → log₂(m/u) weights

`w = log2(m/u)`. Chosen m/u are conditioned on "the pair reached the merge stage" (passed
blocking); hand-set conservative, each with a stated rationale.

| feature | level | definition | m | u | w |
|---|---|---|---|---|---|
| name | exact | canon(a)==canon(b) | 0.40 | 0.02 | **+4.32** |
| name | alias-nick | nickname-table hit | 0.35 | 0.01 | **+5.13** |
| name | alias | token-subset, JW≥0.88, or trgm≥0.5 w/ shared token | 0.35 | 0.03 | **+3.55** |
| name | weak | JW∈[0.75,0.88) or shared first-initial | 0.20 | 0.10 | +1.00 |
| name | disagree | none of the above | 0.05 | 0.90 | −4.17 |
| cooc | strong | rarity-weighted shared partners W ≥ 0.7 | 0.55 | 0.02 | **+4.79** |
| cooc | one | 0.2 ≤ W < 0.7 | 0.15 | 0.08 | +0.91 |
| cooc | nodata | a side has <2 partners | 0.50 | 0.50 | 0.00 |
| cooc | contradict | W<0.2 AND both sides ≥2 partners | 0.03 | 0.35 | **−3.55** |
| time | overlap | mention windows intersect (±14 d) | 0.40 | 0.10 | +2.00 |
| time | nodata | either side single-mention | 0.50 | 0.50 | 0.00 |
| time | contradict | windows disjoint >90 d, ≥3 mentions each | 0.25 | 0.50 | −1.00 |

Rarity weight per shared partner p (Senzing generic-value rule, made arithmetic):

```
w_p = max(0, 1 − df_p / (0.01·N))        # df_p = #entities co-occurring with p
W   = Σ_{p ∈ partners(a)∩partners(b)} w_p
```

so a partner shared by ≤~0.1 % of entities counts ≈1, one shared by ≥1 % (moms, bosses,
"work") counts 0 — two sisters both adjacent to hub-node "mom" get **zero** co-occurrence
credit. This is the single most important guard in the design.

**Prior** `M₀ = log2(λ/(1−λ))`, λ = 0.05 → **−4.25** for entity↔entity merge. For
mention→entity linking the same scorer runs with λ = E/(E+N) estimated per bank
(start at 0.5 → M₀=0): a mention is presumed to refer to an existing entity.

### 2.3 Merge equation and thresholds

```
M = M₀ + w_name + w_cooc + w_time
MERGE   if M ≥ T_merge  = log2(0.95/0.05) = +4.25   (≈ p ≥ 0.95)
REJECT  if M ≤ T_reject = log2(0.20/0.80) = −2.00   (≈ p ≤ 0.20)
ABSTAIN otherwise → row in pending_alias; re-scored when either side gains a
                    co-occurrence edge or a new surface form (event-driven, no polling)
```

Note the structural property: M₀=−4.25 and the largest single-feature weight is +5.13, so
**no feature alone can ever merge** — every merge needs ≥2 agreeing evidences, and any
contradicting evidence (−3.55, −4.17, −1.00) can veto a moderate positive. Hindsight's
bounded 0–1 sum cannot express veto: all its terms are ≥0, so "contradictory context" and
"no context" score identically. That asymmetry is exactly what the two-Caroline rule needs.

Mention resolution (argmax variant): link to argmax_c M_link(c); if top two candidates are
within ~1 weight unit → ambiguity → create a new entity *and* record a pending_alias pair
(a wrong attach is costlier than a resolvable duplicate).

### 2.4 Worked examples (all computed in the local prototype)

Bank N=10k, df: mom=200, david=150, work=500 (hubs); katie=3, piano=5, surfing=2 (rare).

| pair | name lvl | cooc lvl (W) | time | M | p | verdict |
|---|---|---|---|---|---|---|
| Caroline vs Caroline, share only hubs mom+david | exact +4.32 | contradict −3.55 | overlap +2.00 | −4.25+4.32−3.55+2.00 = **−1.48** | .27 | ABSTAIN (not merge) ✓ |
| Caroline vs Caroline, + disjoint years | exact | contradict | contradict −1.00 | **−4.47** | .04 | REJECT ✓ |
| Caroline vs Caroline, share rare katie+piano | exact | strong (0.97) +4.79 | overlap | **+6.86** | .99 | MERGE ✓ |
| Carol vs Caroline, share katie+piano | alias-nick +5.13 | strong (1.92) +4.79 | overlap | **+7.66** | .995 | MERGE ✓ |
| Carol vs Caroline, thin context | alias-nick | nodata 0 | nodata | +0.88 | .65 | ABSTAIN — nick never merges alone ✓ |
| Alice Chen vs Alice C., no context | alias +3.55 | nodata | nodata | −0.70 | .38 | ABSTAIN ✓ |
| Bob vs Robert Chen, disjoint contexts | alias-nick | contradict | contradict | **−3.66** | .07 | REJECT ✓ |
| Bob vs Robert Chen, share katie+piano | alias-nick | strong | overlap | **+7.66** | .995 | MERGE ✓ |
| Melanie Gates vs Melanie, share hub mom only | alias | nodata (hub→0) | overlap | +1.30 | .71 | ABSTAIN ✓ |
| wren vs Wren, strong rare ctx | exact | strong | overlap | +6.86 | .99 | MERGE ✓ |

Same cases through Hindsight's scorer (mention form): Caroline/Caroline-sisters **0.84→merge
(false merge)**, Melanie Gates→Melanie **0.79→merge (false merge)**, Bob→Robert Chen with
strong cooc **0.586→no merge (false keep — confirms PR-2694's caveat that dissimilar names
can't unify on name evidence)**, Carol→Caroline thin context 0.385→keep. The sieve fixes all
three failure directions.

### 2.5 Do-not-merge guard rules

- **G1 type/kind mismatch** (PERSON vs ORG/label) → REJECT before scoring.
- **G2 authored names** (`resolve:false`, caller-supplied canonical names, labels) →
  canonical-exact only, never fuzzy (Hindsight #3479/#1558).
- **G3 speaker exclusivity**: two entities that are both distinct speakers (post-fix speaker
  field) → unconditional REJECT; a speaker is one person.
- **G4 hub suppression**: shared partners with df_p > 1%·N contribute 0.
- **G5 nickname humility**: alias/alias-nick can never reach T_merge without cooc evidence —
  true by construction: max name weight +5.13 + prior −4.25 = +0.88 < T_merge +4.25.
- **G6 ordered consolidation**: apply pending merges in descending M; after each merge,
  recompute features of remaining candidates against the *merged union* partner set
  (blocks single-linkage drift A~B, B~C ⇒ A~C).
- **G7 provenance**: merged surface forms persist as alias rows on the survivor; a merge is a
  metadata union, never a rewrite of pinned bytes; forget(alias) splits it back.
- **G8 intra-batch fold** (keep Hindsight's): union-find new same-retain names at trgm ≥ 0.5
  — O(b²) on tiny b, ~5 ms at b=250 measured in their comment.

---

## 3. Complexity — per-pair and per-pass, measured on this 4-core box

Measured (pure CPython, realistic name lengths): trigram-Jaccard 5.5 µs, Jaro-Winkler 5.8 µs,
SequenceMatcher 10.3 µs, cooc set-op ~5 µs → **per-pair eval ≈ 10–25 µs** (matches Hindsight's
documented ~50 µs/SequenceMatcher-candidate once you add their overhead).

**Mention → entity (per retain/query mention):**
token-inverted-index probe with df cap (`skip tokens with df > max(50, 0.005N)`) + trgm
rescore. Measured on a long-tail bank: avg 0.2–2 candidates, **0.01–0.02 ms**. Generic
single-name mentions (bare "Caroline") miss the text index by design (~80 % miss rate in sim)
and fall through to the **co-occurrence lane**: entities sharing ≥1 rare partner —
O(Σ df_p over rare partners) ≤ partners × 0.01N ≈ 1k lookups ≈ **0.5 ms**. Bound by the
200-candidate cap: ≤ 200 × ~10 µs ≈ **2 ms** worst, ~10 ms if JW is run per candidate.
→ per-mention cost ≤ ~2 ms at both 10k and 100k. Well inside tens-of-ms budget.

**Entity ↔ entity consolidation pass (background job):**
blocked pairs = Σ_entities |rare-token postings ∪ rare-cooc neighborhoods|.
Measured ~2.2k pairs @10k → **0.02 s**; ~116k @100k → **1.2 s** at 10 µs/pair serial.
Pessimistic bound (common-name-heavy bank, ~8 candidates/entity): 80k pairs @10k ≈ 0.8–4.4 s;
800k @100k ≈ 8–44 s serial, ÷4 on 4 cores ≈ 2–11 s. Incremental (only pairs touching new
entities): ≈ 10 candidates × 10 µs ≈ **0.1 ms per new entity**. Nothing runs on the search
hot path.

All-pairs comparison (Dedupe's own arithmetic): 100k entities → 5×10⁹ pairs → 13 h at
10 µs/pair — impossible without blocking; the index above is the blocking.

**Resident size per entity:** canonical string+map ≈ 140 B; token postings ≈ 90 B
(~2.2 tokens × ~40 B); cooc partner set ≈ 400–500 B (~10 partners); df counters 8 B.
Total ≈ **0.6–0.7 KB/entity → ~60–75 MB at 100k entities**. Nickname table
(carltonnorthern/nicknames, 2,828 rows) ≈ **113 KB** static, biased to US names — treat
as one feature level, not ground truth.

---

## 4. Comparison table

| candidate | equation core | labels? | deterministic | same-name safe | nick-conditional | per-pair | verdict |
|---|---|---|---|---|---|---|---|
| Hindsight 0.5/0.3/0.2 > 0.6 | bounded weighted sum, ≥0 terms | no | yes | **no** (0.84 merge on hubs) | **no** (can't express veto; Bob→Robert fails) | ~50 µs | keep for mention-linking only, patched |
| Dedupe logistic + active learning | learned field weights → P(dup) | **yes — fails default** | yes | yes, when trained | yes | field dists ~ms | reject (no human labeler in Verbatim) |
| Senzing ~30 principles (FES) | comparator verdicts → rule lattice | no | yes | yes | yes | cheap | ship as *design elements* (generic-value suppression, 3-way verdict) |
| **F-S sieve (recommended)** | `M₀ + Σ log2(m/u)`, 3 thresholds | no | yes | yes | yes | ~10–25 µs | **ship** |
| F-S + EM self-calibration (Splink-style) | same, m/u fit by EM | no | yes | yes | yes | same | optional (quality profile) — real m/u from bank stats |
| raw Levenshtein feature | edit distance | no | yes | n/a | n/a | ~same as JW | reject — redundant with JW, no prefix bias |

---

## 5. Verdicts

- **Hindsight formula → optional (patched) for mention-linking; reject as merge rule.**
  Verified constants: weights 0.5/0.3/0.2, threshold 0.6 strict, trgm candidate floor 0.15,
  200-candidate cap, intra-batch union-find at 0.5. Its cooc term is recall without rarity
  weighting and can't go negative — false-merges homonyms sharing hub partners and
  false-rejects nicknames. Confidence: high (constants verified at entity_resolver.py:1053–
  1085; the two failure modes reproduced numerically). Would change: nothing — the flaws are
  structural, not parametric.
- **Dedupe → reject.** Supervised logistic weights need labeled pairs; Verbatim has no
  labeling loop. Its blocking math and active-learning-picks-uncertain-pair idea do justify
  the abstain→pending_alias design. Confidence: high on the mechanism; the reject is about
  labels, not quality.
- **Senzing principles → ship as components.** Adopt: expected-behavior classes (Frequency/
  Exclusivity/Stability), generic-value suppression (= the df_cap), three-way same/possible/
  related verdict (= merge/abstain/reject), re-evaluate-on-new-data (= event-driven pending
  pairs). Don't port their proprietary comparators. Confidence: medium-high — principles are
  documented, exact internal thresholds are not public; our F-S translation stands on its own.
- **F-S sieve (candidate d, recommended) → ship default.** Deterministic, auditable,
  no single feature can merge, abstain band is native to the theory (F-S's three-decision
  rule, 1969). m/u table above is the conservative starting point. Confidence: medium on
  absolute weights — they're hand-set; what would change this: per-bank EM estimation of m/u
  (Splink path) once merge history exists, or a labeled slice of the eval corpus showing
  false-merge/false-split rates.
- **Splink-style EM/TF estimation → optional (quality/max profile).** Same equation with
  m,u fit unsupervised on the bank + per-value u adjustment. Zero labels, strictly better
  calibrated; costs a background EM pass. Confidence: high that it dominates hand-weights
  when data volume supports it.
- **Levenshtein as separate feature → reject.** Correlated with JW; JW's prefix bonus is the
  right bias for diminutives ("Carol"/"Caroline" JW=0.925) — one fuzzy-name channel suffices.

### Provisional constants reviewed
- Hindsight's **0.5/0.3/0.2, θ=0.6**: confirmed in code; keep weights for the *mention*
  argmax scorer but add the rarity-weighted cooc and the abstain path; do not reuse θ=0.6 for
  entity merge — replace with T_merge=+4.25 (p≈0.95) log-odds.
- **Entity alias rules**: nickname table = a name-level evidence (+5.13), never sufficient;
  subset/initials = weaker levels; all enforced by the prior gap.
- Cross-encoder / PPR / BM25F items: n/a to this question.

UNRESOLVED: none blocking. Open calibration knob: λ and the m/u table should be re-estimated
from the first real bank (EM, ~50 lines) — flagged as the optional profile rather than a
default because the defaults already encode the required conservatism.
