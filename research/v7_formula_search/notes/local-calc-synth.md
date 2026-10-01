# local calculations — synthetic dialogue slice + arithmetic cross-checks (all CALCULATED)

Slice harness: `eval/v7/scratch/synth_slice.py` — 20 invented sessions, 730 turns
(20 suppressed → eligibility-filtered, proving eligibility-before-rank end-to-end),
235 evidence-bearing questions across single-hop/temporal/multi-hop/entity/openish.
All turns/questions generated from templates; evidence ids registered at emission
time (no text-grep ground truth). every number below is CALCULATED on this fixture —
directional evidence about mechanisms, not a prediction of LoCoMo absolute scores.

## Retrieval systems compared (any@10/any@20)

| system | all | notable category |
|--------|-----|------------------|
| plain BM25 (text only) | 0.260 / 0.302 | entity 0.56, pet 0.47 |
| BM25F{text,speaker,entities,tlabel} | 0.413 / 0.502 | reside 0.15→(see spk) |
| **speaker-mask + BM25** | **0.532 / 0.626** | pet_kind 0.93, multihop 0.70 |
| speaker-mask + BM25F | 0.532 / 0.621 | ≈ same as mask+BM25 |
| mask + BM25F + temporal boosts | 0.532 / 0.634 | reside_now 0.50→0.36 (HURT) |
| BM25F + hashing-dense via RRF | 0.302 / 0.396 | hashing lane dilutes BM25F |
| lanes → RRF k∈{10,30,60,90} | 0.017–0.034 flat | all < BM25F (weak-lane dilution) |
| weighted RRF {1,.75,.5,.5,1} | 0.255 / 0.311 | ≈ flat RRF (0.247/0.323) |
| lanes → min-max CombSUM | 0.451 / 0.583 | beats every RRF variant |
| RRF60 + bounded boosts | 0.247 / 0.328 | boosts ≈ neutral |

## Findings (calculated, on this slice)

1. **Scope-restricted retrieval dominates everything else.** Resolving the
   question's named speaker (full name OR alias) to a candidate scope, then plain
   BM25 inside it, doubled baseline any@10 (0.26→0.53). Field weights *inside*
   the scope barely mattered (spk_bm25f ≈ spk_bm25). The scope decision — not the
   scorer — is the lever. Implication: entity/speaker resolution onto a
   candidate restriction is the highest-value "formula" found anywhere in this
   search; BM25F field tuning is second-order.

2. **CombSUM(min-max) > RRF at every k** (0.45 vs 0.25), reproducing Bruch et
   al. 2024's CC>RRF on our problem shape. Rank fusion threw away the real
   score gaps my lanes produce. Caveat: my synthetic lanes are weaker/more
   correlated than production's 14 — treat as corroboration, not proof.

3. **Weighted RRF ≈ flat RRF** (0.255 vs 0.247). Lane weights {1,.75,.5}
   changed nothing meaningful — supports flattening the provisional weights
   (Hindsight also measured naive weights degenerate: recall@20 0.97→0.40).

4. **Recency multipliers don't fix "current-X" — conditioned or not.** Global
   recency+tlabel boost dropped reside_now 0.50→0.357; the entity-conditioned
   variant (`spk_bm25f_ctemp`, boost only on entity/speaker-matched candidates)
   scored identically (0.528/0.634, reside_now still 0.357). The failure is
   structural, not a conditioning bug: a bounded multiplier can't express
   "the latest *assertion* of (entity, attribute)". The fix is **slot-key
   supersession** — pick the latest valid fact per (entity, attr) at
   eligibility/structure level — the mechanism Hindsight (rewrite links),
   Zep (bi-temporal truncation), and Mem0 (state_key) all ship. Bounded
   boosts remain useful as tie-breakers (Hindsight alphas confirmed) but are
   NOT the current-X fix.

5. **A weak dense lane fused by RRF dilutes a strong fielded score**
   (bm25f+hash → 0.30 vs bm25f 0.41). Rank fusion cannot express "this lane is
   weaker" — needs either score-informed fusion with weights, or min-rank
   gating of weak lanes.

6. **The "job" paraphrase gap is systematic**: job_now = 0.000–0.13 across
   ALL systems including dense-hash ("current job" vs "work at Casa Verde"
   share no n-grams). Exactly the open-domain failure the measured baseline
   shows (0.196). Confirms hashing-embeddings cannot close paraphrase gaps —
   the offline semantic tier needs real static embeddings (see b9/e3) or a
   small ONNX bi-encoder.

## CE latency model (independent cross-check of e1)

MiniLM-L-6-class cross-encoder (23M params, L=6, d=384), per-pair FLOPs ≈
16·d²·T·L (FFN) + 8·d²·T·L (qkvo) + ~12·d·T²·L (attention) ≈ 2.3 GFLOPs at
seq 96, 4.1 at seq 160. On 4-core int8 ONNX at 15–60 GFLOP/s effective:
**38–274 ms/pair**. Pool k table (serial ~25ms/pair assumed mid-range):

| k | serial p95 | batched~×0.6 |
|---|-----------|--------------|
| 16 | 0.40 s | 0.24 s |
| 32 | 0.80 s | 0.48 s |
| 64 | 1.60 s | 0.96 s |
| 128 | 3.20 s | 1.9 s |
| 300 | 7.50 s | 4.5 s |

→ k=16 already exceeds the 150ms default budget unless per-pair lands at
<10ms (needs verified int8 throughput + short seq). **CE is quality-profile
only** unless the k-curve gain is large at k≤16; smallest-k rule = measure
any@10@k for k∈{8,16,32} and pick first k within 0.005 of best.

## PPR-vs-one-hop — measured on a SQLite edge-table fixture (100k nodes, ~350k edges)

In-memory SQLite, edges(src,dst,w) with src index, 20 seed nodes (e2 self-cover):

| method | cost | yield |
|--------|------|-------|
| 1-hop join `dst where src in seeds` | **0.1 ms** | 68 nodes |
| 2-hop (same join on ≤500 hop-1 nodes) | **0.2 ms** | +213 edges |
| forward-push PPR α=0.2, rmax=1e-3, per-node SQLite fetch | **2.4 ms** | 485 pushes, 483 scored nodes |

→ **Latency is NOT the objection to PPR** (2.4 ms fits easily). The real
objections: (a) **eligibility leak** — a PPR walk propagates score *through*
ineligible nodes unless the dst set is intersected with the eligible bitset
at every push (b4's warning: ineligible nodes leak connectivity into eligible
scores); (b) no shipped memory system uses PPR — Hindsight does bounded
1-hop tanh expansion, Zep does unweighted BFS, Mem0g does BFS; (c) graded
spread only matters when hop-2+ evidence exists (~15% of the mix).

Verdict: **1-hop join → ship; bounded 2-hop (fan-out ≤500) → ship; PPR →
reject-for-default**, keep as a quality-profile weighting on the hop-2
candidate set pending dev-slice evidence that multi-hop lift exists.

## Exact dense scan at 100k (cross-check of b10/e3)

100k×384d fp32 = 154MB resident (int8: 38.5MB). Exact cosine via blocked
numpy: ~38M MACs ≈ 8–25ms on 4 cores — well inside budget. **No ANN needed
at ≤100k**; exact int8+fp32-rescore is the shape. Matryoshka-style truncated
64d int8 vectors = 6.4MB/100k — also viable if quality holds.

## Static-table budget (cross-check of e3/b9)

model2vec-class: ~30k vocab × 256d fp32 ≈ 31MB → int8 ≈ 7.7MB — acceptable
hash-pinned artifact. Encode = vocab lookup + mean-pool ≈ 10–50µs/query.
vs hashing: hashing is free but subword-only (job→work fails, measured).

## e5 self-cover — coreference policies on an invented fixture (CALCULATED)

`scratch/coref_fixture.py`: 60 dialog pairs = [named turn, pronoun-continuation
turn]; truth = entity the continuation actually refers to (registered at
emission; 70% two-entity prev turns, 20% single-entity, 10% self-ref "i").

| policy | resolved | wrong attaches | abstained |
|--------|----------|----------------|-----------|
| A prev-turn inherit-all | 53/60 (88%) | **47** | 7 |
| B strict sieve (unique-entity prev only) | 13/60 (22%) | 0 | 47 |
| **C pronoun-class sieve** | **40/60 (67%)** | **0** | 20 |

C = resolve "it/its"→unique non-person in prev AND person pronouns→unique
person in prev, independently; self-reference ("i") never inherits; abstain on
ambiguous-person (2+ people in prev). The object channel alone recovers most
of what B loses — questions name objects too ("what did she name the cat").

Verdict: **ship C** — deterministic, 0 measured wrong-attach, recovers ~2/3 of
pronoun continuations (≈13% of turns on this mix become entity-searchable vs
~4% for B); the remaining 33% needs discourse-level resolution — out of scope
for a deterministic sieve (optional LLM tier only).

## e4 partial self-cover — fact-index oracle on the synth fixture (CALCULATED)

Persisted gen-time fact map `synth_factmap.json` (name → attr → ids/latest_ids);
oracle lookup = resolve question's speaker (name or alias) + question's attr,
return latest_ids for wants_latest else ids.

Result: **hit=71, miss=42, no-slot=122 of 235** — any-rate among covered
questions 0.628, but only ~48% of questions even have a registered
(entity,attr) slot and coverage including misses caps ~30% any. Correlated
noise caveat: first-name collisions across sessions and visit/entity attr
mapping imperfections lower the measured ceiling — direction stands.

Verdict contribution: a fact index is coverage-limited — it's a strong
*lane/field*, never the backbone. Consistent with a6's measured index-time
merge > rank-time merge: fold deterministic facts into the BM25F `facts`
field (+160B/mem) rather than a peer lane, and let the scope restriction
(bigger measured lever) carry the entity structure.
