# c4-graph-expansion — bounded graph expansion in SQLite for Verbatim

Working note for the post-fix engine. Everything below was measured on a synthetic
claim/entity co-occurrence graph in SQLite (bipartite `edges(entity_id, claim_id,
cnt, last_ts)`, indexes on both columns) at **10k** and **100k** memories, on this
box (8-core; all timings are single-query latency and therefore representative of
single-core work on the 4-core reference — expect ≤ ~1.3× shift either way).
Synthetic graph: ~3.5 edges/claim, entity degree p50=3-4, p95≈30, Zipfian hubs up
to deg 28,863 at 100k — i.e., realistic ~5-20 edges/node with real hubs.

## TL;DR — the single most important result

Per-node caps **cannot be applied per-query in SQL** — a correlated
`ORDER BY … LIMIT` cap inside a 2-hop CTE costs **p50 ≈ 2.8 s** (not ms) at 10k.
The fix is a **materialized bounded adjacency table** (`edges_top` = top-32
claims per entity by `cnt·exp(−λΔt)`, maintained at write time): bounded 2-hop
expansion then costs **p50 ≈ 0.05 ms** and bounded PPR totals **p50 ≈ 1.4-2.9 ms**
at 100k. Ship entity-overlap + one-hop weighted join (both ~10-60 µs) as the
default entity lane; ship the bounded 2-hop gather + a **2-step spreading-
activation scorer** (top-20 Jaccard 0.88 vs converged PPR, no iteration) as the
lane's expansion stage; converged push-PPR goes to the quality/max profile.
Hop ≥ 3 is rejected (10× cost, mass dilutes below ε). The expansion itself
*can* pay — it is the only path to non-seed-adjacent evidence — but only on
the materialized bounded adjacency, never on raw `edges`.

## 1. Candidates, exact formulas, parameter provenance

Notation: seeds `S` = entities resolved from the query; `E(c)` = entities of claim
`c`; `d(u)` = weighted degree `Σ_v w(u,v)`; `w(e,c)` edge weight.

**Edge weight (shared by all candidates).**
`w(e,c) = cnt(e,c) · exp(−λ·Δt_c)`, `λ = ln2 / 60d` (60-day half-life).
Provenance: `cnt` linear is the co-occurrence count the assignment fixes; the
exponential recency term is the same decay family Verbatim already uses for
temporal boosts (α=0.1-0.2 bounded boosts). Half-life is a tunable — 60d chosen
because memory co-mentions older than ~2 months rarely drive current-context
association; sensitivity is low (see §6).

**C1 — entity-overlap only (no walk).**
`score(c) = |S ∩ E(c)|` tie-broken by `Σ_{e∈S∩E(c)} cnt(e,c)`.
SQL: `SELECT claim_id, COUNT(DISTINCT entity_id) m, SUM(cnt) w FROM edges WHERE entity_id IN (S) GROUP BY claim_id ORDER BY m DESC, w DESC LIMIT k`.
No parameters beyond `k`. This is the cheapest scorer that exists; it also has
an important analytic property — see §5.

**C2 — one-hop weighted join.**
`score(c) = Σ_{e∈S∩E(c)} w(e,c)`. Same lookup as C1, different `ORDER BY`.
Provenance: the assignment's (a); "fan-out cap" is implicit via `LIMIT k` on the
aggregate (per-seed scans are bounded by `deg(seed)` — hubs are the only risk,
and a hub seed costs ≤ its degree in row touches, measured ≤ ~0.4 ms).

**C3 — bounded 2-hop gather (recall stage).**
`S →top-F1 claims→ top-F2 entities →top-F3 claims`, executed as three set-UNION
CTEs **on `edges_top`** (materialized top-32 adjacency):
`claims1 = {c : ∃e∈S, (e,c)∈edges_top}`, `ents2 = E(claims1)\S`,
`claims2 = E(ents2)`, answer = `claims1 ∪ claims2`.
Constants: `F1`=64 claims/seed, `F2`=32 entities/claim, `F3`=32 claims/entity,
`|ents2|`≤256 globally — chosen at the measured blowup knee (§3). Source of the
pattern: Zep/Graphiti's production design uses bounded BFS expansion as one of
three retrieval lanes (cosine + BM25 + BFS) — direct evidence the pattern is
sufficient in a real agent-memory system.

**C4 — bounded personalized PageRank (Andersen push) on the loaded subgraph.**
Load the C3-bounded subgraph into memory, then Andersen–Chung–Lang push:
```
resid[seed] = mass_s;  est = 0
while ∃u: resid[u] ≥ ε·d(u):
    push(u):  est[u] += α·resid[u]
              resid[v] += (1−α)·resid[u]·w(u,v)/d(u)   ∀v∈N(u)
              resid[u] = 0
```
Score claims by `est`. Parameters: restart `α`, residual threshold `ε`,
fan-out caps (shared with the load). Provenance for the algorithm: ACL'06;
for `α`: HippoRAG tunes damping=0.5 on MuSiQue and reports robustness;
FAST-PPR and the local-update literature use teleport α≈0.15-0.2 for a
different task (pairwise significance on web/social graphs). My sweep says
α=0.4-0.5 is right *here* (see §3): higher restart → fewer pushes, more
locality, identical recall — and it agrees with HippoRAG.

**C5 — 2-step spreading activation (PPR-equivalent, recommended form).**
One explicit bipartite pass, no iteration:
```
m(x)  = (1−α)·Σ_{a∈S} w(a,x)/d(a)          hop-1 claims
m(b)  = Σ_x m(x)·w(x,b)/d(x),  b∉S         bridge entities
s(g)  = Σ_b m(b)·w(b,g)/d(b) + α·m(g)      hop-2 claims + hop-1 direct mass
```
Identical mass mechanics to C4 truncated after two pushes. Provenance:
measured top-20 Jaccard vs converged C4 = **0.88** (top-10 overlap 9.1/10);
the α·hop-1 term added to the formula is what closes the gap (without it
Jaccard drops — hop-1 claims keep ~40-50% of restart mass at α=0.4-0.5 and
must rank above hop-2). Deterministic, `ε`-free, monotone, explainable.

**C6 — rejected variants** (each measured): unbounded 2-hop gather on raw
`edges`; correlated per-node SQL cap; hop-3 gather; load-from-raw-`edges` PPR.

## 2. Worked example (verify the mechanics, not just the algebra)

Graph: edges (all `w=1`) `A-x1, A-x2, x1-B, x2-B, B-g, x2-D, D-d1`. Seed `{A}`,
α=0.4. Degrees: `d(A)=2, d(x1)=2, d(x2)=3, d(B)=3, d(D)=2, d(g)=1, d(d1)=1`.

C5 pass: `m(x1)=m(x2)=0.6·1/2=0.3`. `m(B)=0.3·(1/2+1/3)=0.25`, `m(D)=0.3·1/3=0.1`.
`s(g)=0.25·1/3=0.083`, `s(d1)=0.1·1/2=0.05`, `s(x2)=0.3·0.4+0.3·(0.25/3+0.1/2)≈0.253`,
`s(x1)=0.203`. Order: **x2 > x1 > g > d1**.
Converged push (measured, ε=1e-6, 708 pushes): `x2=0.178, x1=0.165, g=0.019,
d1=0.013` — same order. The absolute masses differ (convergence re-concentrates
mass near seeds through cycles) but **the ranking is identical** — that is the
mechanism behind the 0.88 Jaccard.

## 3. Per-query cost: complexity → ms

ACL'06 cost bound, derived: a push at `u` retires `α·r(u) ≥ α·ε·d(u)` into
`est`; since `Σp ≤ 1`, `pushes(u) ≤ p(u)/(αε·d(u))` and total work
`W = Σ pushes(u)·d(u) ≤ 1/(αε)` neighbor-touches — degree-free, graph-size-free.
On the *loaded* bounded subgraph the real bound is
`min(1/(αε), |V_sub|·max_repush)` — |V_sub| ≤ ~2.5k nodes here.

Measured (p50 / p95 / max, ms; n=150 queries at 10k, 120-150 at 100k):

| scorer | 10k p50 | 10k p95 | 100k p50 | 100k p95 | verdict |
|---|---|---|---|---|---|
| C1 overlap SQL | 0.010 | 0.046 | 0.013 | 0.072 | ship |
| C2 onehop weighted | 0.008 | 0.036 | 0.011 | 0.062 | ship |
| C3 2-hop on `edges_top` | 0.047 | 0.56 | 0.051 | 0.54 | ship |
| C5 load+2-step on `edges_top` | ~0.06+~0.05 | — | ~0.11+~0.05 | — | ship |
| C4 PPR α=.4 ε=1e-4 (load+push) | 4.6+1.7 | 4.2 | 0.11+1.3 | 4.2 | optional |
| C4 PPR α=.5 ε=1e-5 | 5.5+3.8 | 11 | 0.10+2.0 | 7.8 | optional |
| 3-hop on `edges_top` | 0.86-1.1 | 8.7 | 0.68 | 12.0 | reject |
| 2-hop unbounded on `edges` | 2.7-3.5 | 8.0 | 21.4 | 77.5 | reject |
| 2-hop correlated-cap CTE | **2890** | 4770 | (worse) | — | reject |
| C4 load on raw `edges` | — | — | **37** | — | reject |

Reading the table: (i) C1/C2 are free at any scale — indexed `IN` + aggregate;
(ii) `edges_top` is the entire game: bounded gather = ~0.05 ms, unbounded =
21-77 ms at 100k (×10 with edge count), correlated caps = ~3,000,000× the
bounded case; (iii) C4's cost at 100k is **load 0.11 ms + push 1.3-2.0 ms**
with pushes p50 ≈ 550-915 (ACL bound at α=0.4, ε=1e-4: ≤25k touches worst case;
measured ~550 pushes × ~4 out-edges ≈ 2.2k touches ≈ bound/10); (iv) hop-3 is
~10× hop-2 and nothing reached is any good (§5 dilution).

**Pushes vs parameters (100k, `edges_top` load):** pushes p50 = 550 (α=.4,
ε=1e-4), 915 (α=.5, ε=1e-5), ~3.3k (α=.2, ε=1e-4) — higher α and higher ε both
cut work; recall on planted gold was flat 15/15 across α∈{0.4,0.5}, ε∈{1e-4,1e-5},
so pick **α=0.4-0.5, ε=1e-4** (ε on the `ε·d(u)` threshold ≈ 0.05-0.1% mass
cutoff). eps smaller buys pushes, not recall.

## 4. Index / resident size

`edges` row = 4 fields + 2 indexes → measured **54-59 B/edge** → 184-204 B/claim
at ~3.5 edges/claim (1.87 MB @10k, 20.6 MB @100k). `edges_top` keeps
`Σ_e min(deg(e),32)` rows = 40-49% of edges (hubs are few but fat: 1,034 of
25k entities hold ~60% of edge mass at 100k) → **+8.2 MB @100k ≈ +80 B/claim**.
Total graph store ≈ **~280 B/claim** at 100k. PPR transient: bounded subgraph
≤ ~2.5k nodes → ~1 MB of Python dicts per query, garbage-collected. Build cost
for `edges_top`: 0.6 s one-shot at 100k (window function over `edges`); at write
time it is one `INSERT OR REPLACE` + occasional re-rank — trivially maintained
incrementally.

## 5. When does hop-2 beat one-hop — the analytic answer

`onehop(c) = Σ_{e∈S} w(e,c)·1[e∈E(c)]` ⇒ score = 0 for every non-seed-adjacent
claim. Hop-2 adds `Σ_b m(b)·w(b,c)/d(b)` for bridges `b∉S` ⇒ **hop-2 strictly
dominates on recall of non-adjacent evidence**; the question is only whether
those items are worth ranking and at what cost. Three regimes, all verified:

1. **Sparse seeds** — provably pays. If `Σ deg(s) < k`, one-hop mathematically
   under-fills the top-k: measured deg≤3 seeds return 2-3/20; hop-2 returns
   19-20/20. This is a *query-time detectable* trigger (`COUNT` the hop-1 set
   first — costs the same ~10 µs query you already ran).
2. **Associative (bridge) gold** — pays by construction. Planted hop-2-only gold:
   overlap 0/15, one-hop 0/15, C3 15/15, PPR 15/15. When gold shares no seed
   entity, no amount of tuning rescues one-hop — the mass is identically zero.
   Hop-3 gold: 0/15 everywhere — mass at hop h dilutes as ~(1−α)^{2h}/B^h
   (α=0.4, branching ≈30·6: hop-2 ≈ 0.36·(1/180)·w ≈ 2e-3 vs hop-3 ≈ 0.13·(1/5400)
   ≈ 2.4e-5 < ε·d → unscored noise). This is why hop-3 is rejected: the same
   math that makes hop-2 work makes hop-3 worthless.
3. **Multi-entity (bridge-ranking) queries** — mostly *doesn't* pay over C1, and
   this is the honest surprise. A claim containing 2+ seeds is integer-ranked
   first by overlap (matched-seed count 2 > 1) and by PPR (two convergent
   paths); both orderings agree whenever a both-seed claim exists (planted
   bridges: 15/15 for C1, C2, C3, C4 alike). PPR's path-convergence advantage
   only appears for non-adjacent claims — where overlap scores 0. So the
   multi-seed convergence story is real but *C1 already captures it for
   adjacent claims*; PPR's marginal contribution is ordering non-adjacent
   ones. (Caveat: in the uniform random workload, random medium-degree seed
   pairs essentially never co-occur — P(both-seed claim) ≈ d1·d2·ē/n ≈ 0.08 —
   so natural multi-seed bridges are rare without topical clustering; planted
   bridges are the right test.)

Hub attenuation is the PPR-family's quiet win: mass through a deg-28k hub is
divided by d(u) — a built-in IDF. Plain gather scoring lacks it, but `edges_top`
already caps hubs to 32, so the mechanism is handled at the edge level either
way (and `edges_top` pruning *improved* h2 recall 14/15→15/15 vs raw edges —
hub co-mentions are noise).

## 6. Recommended constants

| const | value | provenance |
|---|---|---|
| restart α | **0.4-0.5** | HippoRAG damping=0.5 (tuned, robust); measured: fewer pushes, same recall vs 0.2. Lit α≈0.15-0.2 is for web-scale significance, not local retrieval |
| ε (residual thresh) | **1e-4** on `ε·d(u)` | measured knee: 550 pushes p50; ε=1e-5 buys +65% pushes for 0 recall |
| hop limit | **2 entity-hops** (e→c→e→c) | dilution math + hop-3 recall 0/15 + cost 10× |
| fan-out caps | F1=64 claims/seed, F2=32 ents/claim, F3=32 claims/ent, \|ents2\|≤256 | knee of load-time blowup; F3=32 = edges_top cap |
| `edges_top` | top-32 per entity by `cnt·exp(−ln2·Δt/60d)` | measured 2.8s→0.05ms enabler; cap only bites hubs |
| trigger | run expansion iff hop-1 fill < k, or always (≤8ms) | sparse-fill gate is a free COUNT |
| seed mass | per-seed ∝ seed-linking confidence; claim-type seeds ×0.05 if used | HippoRAG2 weight-factor ablation: 0.05 optimal, "factor is crucial" |

## 7. Stage mapping

C1/C2 → the **entity lane** (one SQL query, both orderings available).
`edges_top` → **write-path** (maintain at add-time; it's the bounded adjacency
every expansion stage reads). C3 → **lane** (same lane's recall stage; emits a
candidate set for fusion). C5 → **lane scoring** (ranks the gathered set;
default profile). C4 converged push → **optional rerank inside the lane**
(quality/max only: +1-8 ms for an ordering refinement fusion will rarely see).
Nothing here touches fusion/RRF weights, rerank, temporal, or pack stages — the
lane emits its top-k and exits. Byte-budget impact: none (candidates only).

## 8. Verdicts

- **entity-overlap (C1) → ship (default):** ~10-60 µs, captures multi-seed
  bridges via matched-count; it is the floor the lane cannot go below.
  Confidence: high — measured at both scales; change nothing unless entity
  extraction itself is shown noisy on real data.
- **one-hop weighted join (C2) → ship (default):** same query, weight-aware
  ordering; keep both orderings — overlap for bridge-first, weighted for
  salience-first — pick per query or emit fused. Confidence: high.
- **bounded 2-hop gather on `edges_top` (C3) → ship (default):** +0.05-0.5 ms,
  the only path to non-adjacent recall; mandatory when hop-1 underfills.
  Confidence: high on cost; medium-high on recall value — its real-traffic
  hit-rate depends on associative-question prevalence, which LoCoMo will show.
- **2-step spreading activation (C5) → ship (default):** PPR-equivalent
  ranking (0.88 Jaccard, same order on worked example) at ~0.1 ms, zero
  tuning surface. This is the form "bounded PPR" should take in production.
  Confidence: medium-high — 88% agreement measured on 80 queries; if synthesis
  wants the last 12%, the quality profile has converged PPR.
- **converged push-PPR (C4) → optional (quality/max):** +1.3-8 ms vs C5's ~0.1
  for a ranking delta fusion mostly won't feel. Confidence: medium — could
  promote to default if LoCoMo shows the extra recall class drives any@10.
- **hop-3+, unbounded gather, correlated SQL caps, raw-`edges` PPR load →
  reject:** 10× cost with sub-ε dilution; 21-77 ms at 100k; ~2.8 s/query;
  37 ms load — each dead on arrival at a 46/85 ms budget.
- **`edges_top` materialized bounded adjacency → ship (enabler):** +80 B/claim,
  turns every expansion stage from over-budget to trivial. This is the real
  answer to the question as posed — not which walker, but which data structure.

UNRESOLVED: prevalence of associative (non-adjacent) gold in real LoCoMo
traffic — the synthetic graph can't answer it, and it's what sizes C3/C5's
actual any@10 delta. Closable by instrumenting the lane on the real run
(report hop-2-only items entering packs), cheap to add now.

## Sources

- fanchung.ucsd.edu/wp/localpartition.pdf | paper | ACL'06 push algorithm;
  derived the ε-approximate work bound W ≤ 1/(αε) myself from the push rule
- arxiv 1404.3181 | paper | FAST-PPR: teleport α=0.2 in evals; forward/reverse
  work balancing — evidence α is task-dependent, not universal
- arxiv 2405.14831 | paper | HippoRAG: PPR over entity-fact memory graph,
  damping=0.5 (tuned on 100 MuSiQue ex), vendor-reported +11% R@2 / +20% R@5
  on 2Wiki — the strongest existence proof that PPR-over-memory pays
- arxiv 2502.14802 | paper | HippoRAG2: passage-node weight factor 0.05 with
  recall@5 ablation table — evidence for seed-mass balancing and "all passages
  as seeds" (broad activation)
- arxiv 2501.13956 | paper (Zep/Graphiti) | production agent-memory uses
  bounded BFS expansion lane + mention-frequency reranker + RRF — validates
  C3/C1 over PPR in the deployed baseline; vendor-reported latency -90%
- ciir-publications.cs.umass.edu (Dalton/Dietz/Allan SIGIR'14) | paper | EQFE:
  entity-anchored expansion beat text expansion on ClueWeb — mechanism-level
  support that entity-pinned expansion pays (different corpus — label
  extrapolation)
- github.com/ruvnet/ruvector forward_push.rs | repo | confirms the O(1/ε)
  push bound and `resid > ε·deg(u)` threshold in a real implementation
- /tmp/c4 bench (this session) | measured | all ms numbers, pushes, recalls,
  byte counts on the synthetic SQLite graph at 10k/100k
