# b2-rrf-sensitivity — RRF deep dive and k sensitivity (orchestrator-produced; agent b2 exhausted retries)

## Formula and canonical source

`RRFscore(d) = Σ_lane  w_lane / (k + rank_lane(d))`, absent lanes contribute 0.

Cormack, Clarke & Büttcher (SIGIR'09, doi:10.1145/1571941.1572114): k=60 fixed by a
pilot investigation and held constant through validation; the constant "mitigates the
impact of high rankings by outlier systems" while keeping deep ranks non-vanishing
(unlike an exponential). On TREC and LETOR-3 meta-ranking it beat Condorcet Fuse,
CombMNZ, and every individual input ranker — with **no training data**.

## k sensitivity — computed here, not quoted

Contribution 1/(k+rank) and the top-vs-deep ratio:

| k | rank 1 | rank 10 | rank 40 | rank 100 | rank1/rank40 |
|---|--------|---------|---------|----------|--------------|
| 10 | 0.0909 | 0.0500 | 0.0200 | 0.0091 | 4.55× |
| 30 | 0.0323 | 0.0250 | 0.0143 | 0.0077 | 2.26× |
| 60 | 0.0164 | 0.0143 | 0.0100 | 0.0063 | 1.64× |
| 90 | 0.0110 | 0.0100 | 0.0077 | 0.0053 | 1.43× |

Corroboration rule (two lanes hitting the same doc vs a lone #1 hit), computed:

- k=10: lone #1 loses only to two ≤rank-10 corroborations; top-heavy — an outlier
  lane's #1 dominates rank-20+ corroborated evidence. Outlier-prone.
- k=30: two rank-20 corroborations beat a lone #1; two rank-40 do not.
- k=60: two rank-40 hits beat a lone #1 — **deep-corroboration regime**, which is the
  right shape when lane caps are ~40 and the win condition is evidence any@10.
- k=90: nearly pure vote counting; the top of a strong lane is barely distinct
  (1.43×) from rank 40 — throws away real signal.

Optimum is flat over [30, 60]: at both, corroboration outranks lone outliers, and
the ordering inside each regime differs only at the margin. k=10 is the only clearly
bad value (outlier domination); k=90 dilutes lane strength without compensating
corroboration benefit. **Default k=60 survives**; grid {10,30,60,90} on the dev slice
is a cheap confirmation, expected spread small.

## RRF vs score-informed fusion — the modern evidence

Bruch, Gai & Ingber, *An Analysis of Fusion Functions for Hybrid Retrieval*
(ACM TOIS 42(1), arXiv:2210.11934): studied RRF vs convex combination (CC) of
normalized lexical+semantic scores. Findings, contrary to the "RRF is parameter-free
and safe" folklore (Chen et al. 2022):

- CC (α·semantic + (1−α)·lexical, min-max or z normalized) **outperforms RRF in- and
  out-of-domain**; RRF discards score-distribution information by construction.
- A *parametric* view of RRF (per-retriever weights — wRRF has as many parameters as
  retrievers + 1) is sensitive to its parameters and **generalizes poorly
  out-of-domain**: tuned RRF overfits the tuning collection.
- Normalization scheme is a small detail — min-max / z-score / linear transforms are
  rank-equivalent at the optimum.
- CC is sample-efficient: α tunes on a handful of labeled queries.

Implication for our 6–8-lane mix: rank-only RRF is the right *default* (no training
data needed, robust to one noisy lane — computed: junk at lane-rank 1 contributes
1/(k+1) while three lanes @rank5 contribute 3/(k+5) ≈ 2.8× more at k=60, so one
spammy lane cannot dominate). But score-informed fusion is the better *shape* once
a dev slice exists: see b3-combsum-calibration for the calibrated-linear path — the
recommended design is calibrated CombSUM with the RRF result as its safety floor.

## Weighted RRF and the noisy-lane failure mode

- Vendor evidence AGAINST hand-set lane weights: Hindsight's own eval found naive
  weighted RRF degenerated (measured recall@20 0.97 → 0.40); they ship **flat RRF
  k=60, no lane weights**, and the only weighting they kept is a rank-divisor lane
  boost of form 1/(k + rank/w), w ∈ {2,4,8} (verified in their code). Verbatim's
  provisional {1.0/0.75/0.5} lane weights are therefore NOT evidence-backed —
  flatten to uniform w=1 unless the dev slice shows a lane earning a different
  divisor.
- The true noisy-lane failure is not a single bad hit (RRF absorbs it) but a lane
  that is *consistently* mid-rank-wrong on every query — it floods the fused list
  with weak corroborations. Mitigations, in order of preference: (a) min-rank gate —
  a lane contributes only its top-N (e.g., N=20) unless it is the only lane with
  coverage; (b) per-lane divisor weight 1/(k+rank/w) tuned on dev (Hindsight's
  proven form), not multiplicative weights; (c) drop a lane entirely if its unique-
  contribution on dev is ≤0 (measured by leave-one-lane-out ablation).
- Elasticsearch's weighted-RRF (2025) is the same per-retriever-weight idea; treat
  it as confirming the mechanism exists, not as evidence for specific weights.

## Verdicts

- Flat RRF k=60 → **ship** for default profile. Confidence high — canonical,
  training-free, vendor-verified in production (Hindsight ships exactly this), and
  the corroboration math matches our lane depth.
- k ∈ {30, 90} → **optional**, tune on dev slice; k=10 → **reject** (outlier
  domination), k=90 → weak benefit, near-vote-count.
- Hand-set lane weights {1.0/0.75/0.5} → **reject as provisional constant**:
  replace with flat weights, or Hindsight's divisor form w/(k+rank/w′) if dev
  tuning justifies per-lane strength. This is the provisional constant most likely
  to be silently hurting.
- Calibrated linear fusion (CombSUM on isotonic/logistic-normalized lane scores) →
  **ship pending dev-slice win** — Bruch: CC beats RRF and tunes on ~a handful of
  queries; b3 note has the recipe. Keep RRF as fallback when calibration data is
  absent (fresh installs, new scopes).
- Min-rank gate (top-20 per lane) → **ship**: cheap insurance for the consistent-
  noise failure mode.

## Sources

- https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf | paper | the formula, k=60 origin, outlier-damping rationale, TREC/LETOR evidence
- https://arxiv.org/html/2210.11934v2 | paper | CC>RRF in/out-of-domain, wRRF sensitivity and poor generalization, sample-efficiency of α, normalization equivalence
- vectorize-io/hindsight (via a1 note) | repo | flat RRF k=60 in production; naive lane weights measured harmful (recall@20 0.97→0.40); rank-divisor boost form
- https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf | vendor | weighted-RRF per-retriever weights exist in production search
- local arithmetic | calculated | all contribution/corroboration tables above computed directly
