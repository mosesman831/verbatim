# Verbatim V7 formula search — decision report

Date: 2026-09-22. Method: 34 disjoint research agents (literature, ranking
formulas, memory structure, OSS code reads, new combinations) + local
measurement on a 235-question synthetic dialogue slice
(`eval/v7/scratch/synth_slice.py`, eligibility-enforced, evidence-registered
corpus; all its numbers marked **calculated**). Vendor = self-reported numbers;
paper = peer-reviewed/published; calculated = our arithmetic/measurement.

Measured baseline being beaten: 40-line Okapi BM25 any@10 **0.596**, any@20
0.677 on the local LoCoMo cat 1–4 slice; Verbatim V3 current 0.546/0.590;
open-domain 0.196 vs BM25 0.370; p50/p95 ≈ 46/85 ms on ~590 turns. Hardware:
4-core CPU, 10k–100k memories, p95 tens-of-ms default / ≤150 ms quality.

---

## 1. The one-page decision

**Recommended formula tag set** (default install, no network, no API key):

```
retrieval_policy = "eligible-first pooled retrieval v7"
bm25f            = FTS5 bm25(tbl, weights) candidate lane  +
                   Python BM25F rescore on the ELIGIBLE set,
                   idf = ln(1 + (N_E - df_E + 0.5)/(df_E + 0.5)),
                   k1 = 1.2, b = 0.6 (grid-search [0.6–2.0]×[0.3–0.8]),
                   fields {text 1.0, speaker 1.5, entities 2.5,
                           facts 1.8, tlabel 0.4}    [facts = doc-expanded field]
fusion           = RRF k=60 flat (no lane weights) if no dev labels;
                   calibrated CombSUM (per-lane isotonic PAV ≤32 breakpoints)
                   when ≥ ~150 judged queries exist   [ship-behind-gate]
rerank_features  = none default; bounded multiplicative boosts only:
                   s' = s·(1+0.2·(recency−0.5))·(1+0.2·(temporal−0.5))·(1+0.1·(proof−0.5))
                   applied ONLY to temporally/entity-matched candidates
temporal         = composite: bi-temporal intervals (said_at, valid_from,
                   valid_until i32-min + far-future sentinel), newest-wins
                   truncation at write; query-time resolver (anchor at
                   question ts) → eligibility prefilter when unambiguous;
                   recency r=0.5^(d/120) folded to ±10% bounded boost;
                   slot-key supersession for "currently"; date-range
                   surfaced in pack headers
graph_ppr        = REPLACED by bounded 2-hop expansion on a MATERIALIZED
                   adjacency table edges_top (top-32 claims/entity,
                   +80 B/claim): per-node caps can't be enforced inside a
                   query (correlated ORDER BY…LIMIT = 2.8 s p50) —
                   edges_top gather = 0.05–0.5 ms; 2-step spreading
                   activation ≈ converged PPR (0.88 top-20 Jaccard,
                   α=0.4–0.5, ε=1e-4); converged PPR → max profile only
alias            = deterministic canonicalization + Fellegi–Sunter sieve:
                   M = −4.25 + Σ log2(m/u) per evidence (exact +4.32,
                   alias-nick +5.13, cooc-strong +4.79, cooc-contradict
                   −3.55, time-overlap +2.00), rarity-weighted partners
                   w_p = max(0, 1 − df_p/0.01N); merge ≥ +4.25,
                   reject < −2.0, abstain → pending_alias
semantic tier    = DEFAULT: static-embedding lane — model2vec/potion-base-8M
                   table quantized int8 (~8.2 MB hash-pinned artifact,
                   25.6 MB vectors/100k), exact-cosine scan ~8–12 ms@100k,
                   fused with weight ~0.6–0.8 via calibrated sum;
                   hashing lane stays as a free lexical lane
cross-encoder    = OPTIONAL profile only: MiniLM-L2-class ONNX int8
                   (~4 MB, ~6 ms/pair) pool k=16 ~104–127 ms p95;
                   L-6 class (~11–16 ms/pair) is max-profile at k≤32;
                   k=300 rejected (3.4 s)
```

**Why this shape, in one paragraph.** Every independent code read (Hindsight
TEMPR, Zep/Graphiti, Mem0) and every calibrated-fusion paper converge on:
restrict to the *eligible, scoped* candidate set first, score that set with
fielded BM25F using *eligible-set* statistics, fuse complementary lanes by a
normalized-score convex combination (RRF k=60 as the uncalibrated floor),
apply small bounded multiplicative boosts for recency/temporal/proof signals,
and expand one bounded hop through an entity/fact edge table. On my synthetic
slice the single biggest measured lever is scope restriction (speaker/entity):
any@10 0.26→0.53 — larger than the entire BM25-vs-Verbatim measured gap. The
second is replacing rank fusion with score fusion (+0.18 any@10 over RRF,
corroborating Bruch 2024 and my calibration agent's 0.92 vs 0.72 sim).

**Complexity.** Default path per query: 1 FTS5 MATCH (cap 16 lowest-df terms,
top ~500) → eligible-set bitset AND (0.02 ms warm / 13 ms cold @100k) → Python
BM25F rescore of ≤500 (~1–5 ms) → optional semantic exact-cosine lane
(~10–25 ms @100k int8) → fuse (≤1 ms) → bounded 1-hop expansion (~1–10 ms) →
boosts + pack (<1 ms). Total ≈ 5–40 ms p95 at 100k — inside budget. Memory:
bitsets are ~0 standing bytes; facts field +160 B/mem; embeddings 38 MB
int8 @100k (optional profile); no CE in default path.

---

## 2. Ablation table

| # | candidate | what it changes | evidence / metric | latency | memory | license | verdict |
|---|-----------|-----------------|-------------------|---------|--------|---------|---------|
| 1 | FTS5 `bm25(tbl,w…)` as candidate lane | lexical gen | canonical BM25F shape (b1, code-verified), global idf ⇒ gen-only | ~1–5 ms | 0 | SQLite | **ship** |
| 2 | Python BM25F rescore on eligible set, idf=ln(1+x) | final score | b1: formula fixed across Lucene/FTS5/Anserini; ln(1+x) kills negative-idf edge measured −0.419 | ~1–5 ms | ~0 | n/a | **ship** |
| 3 | eligible-set df via int-bitsets | stats | b1 calculated 13.2 ms cold / 0.02 ms warm @100k | above | ~0 | n/a | **ship** |
| 3b | b is the live BM25 lever (tune b first; k1 near-inert at tf≈1 short turns) | constants | d4 ranx/pyserini verified: k1=0.9/b=0.4 Anserini provenance; ±12–18% score shift per 2× length dev | 0 | 0 | n/a | **ship note** |
| 4 | facts as BM25F *field* (doc expansion) not a lane | write+retrieval | a6 (paper): index-merge beat rank-merge R@10 0.862 vs 0.754; +9.4% rel recall | +0.3 ms | +160 B/mem | n/a | **ship** |
| 4b | typed-facts union lane (SPO + byte-pinned quote, w=1.0, fact→turn dedup) | lane | e4: facts-only ceiling ~0.85× raw coverage (LoCoMo obs vs dialog R@k) — must union; union ceiling ~0.66–0.72 vs ~0.60 turn-only | +0.3–3.3 ms | +250–500 B/mem | n/a | **ship (alt form to #4 — dev-slice picks)** |
| 4c | LLM T2 extraction vs rules | write | e4: enable iff Δany@10 ≥ +3pts (McNemar z~3, n=1536) and rule coverage <0.6 | write-path | — | n/a | **optional (quality)** |
| 5 | RRF k=60 flat | fusion | Cormack (paper): robust k∈[30,500]; Hindsight code: flat, weights measured 0.97→0.40 recall@20; my sim: weights≈0 | 0.2 ms | 0 | n/a | **ship (floor)** |
| 6 | calibrated CombSUM (isotonic/Platt per lane) | fusion | b3 sim any@10 0.92 vs RRF 0.72; Bruch (paper) CC>RRF 9/9; saturates ~10 labels/lane | 0.2 ms | ~256 B/lane | n/a | **ship gated on dev slice** |
| 7 | MemGAS entropy lane weights w∝1/H | lane weights | a8 sweep (paper): router is largest ablation component | 0.02 ms | 0 | n/a | **ship as wRRF option** |
| 8 | weighted RRF {1,.75,.5,.5,1} | — | my sim: 0.255 vs flat 0.247 ≈ noise; Bruch: wRRF poor OOD | — | — | n/a | **reject → flat/CC** |
| 9 | min-rank gate on weak lanes | fusion | my sim: weak hashing lane dilutes bm25f 0.41→0.30 | 0 | 0 | n/a | **ship (score-fusion subsumes)** |
| 10 | speaker/entity scope restriction | lane→scope | my sim **+0.27 any@10** (0.26→0.53); a2 Mem0 entity-boost; biggest single lever | <1 ms | 0 | n/a | **ship** |
| 11 | bounded boosts α=0.2/0.2/0.1 on matched candidates | rerank | d1 code-verified (Hindsight exact constants); my sim: recency multiplier doesn't fix latest-X conditioned or not → slot supersession is the real fix (c3/c5) | <1 ms | 0 | n/a | **ship as tie-breaker** |
| 12 | bi-temporal validity intervals + truncation | write+elig | a3 Zep + Mem0g + EverMemOS convergence (a8); newest-wins = 1 UPDATE | <1 ms | +16 B/fact | n/a | **ship** |
| 13 | query-time temporal prefilter (unambiguous only) | eligibility | a6 (paper) +6.8–11.3% rel TR recall; resolver ~6 µs; empty-set fallback mandatory | ~0.01 ms | 0 | n/a | **ship** |
| 14 | date-range surfacing in pack | pack | a3: Zep's biggest measured temporal win | ~0 | ~10 B/item | n/a | **ship** |
| 15 | bounded 2-hop via MATERIALIZED edges_top (top-32 claims/entity) | graph lane | c4 measured: in-query caps impossible (2.8 s p50) → materialize (+80 B/claim); gather 0.05–0.5 ms; planted hop-2 gold 15/15 vs 0/15 one-hop | 0.05–0.5 ms | ~58 B/edge | n/a | **ship** |
| 16 | full PPR on edge table | graph | a8: pure-Python 1.17 s @100k; a1: no PPR in Hindsight at all | 5–40+ ms | — | n/a | **reject (default)** |
| 17 | entity merge 0.5·name+0.3·cooc+0.2·t, ≥0.6, trigram≥0.3 | write path | a1 code-verified constants BUT c1 repro fails both directions (merges homonyms 0.84; fails Bob→Robert 0.586) | ~0 | — | n/a | **reject → Fellegi–Sunter** |
| 18 | access-driven strength R=e^(−t/S), S+=1 on hit | rerank | a8: MemoryBank + 2 systems converge; 8 B UPDATE | ~0 | 8 B/mem | n/a | **ship (bounded boost)** |
| 19 | pinned-span observation atoms + obs lane | write+lane | a5: deterministic Observer → short dated spans; obs→source edge α=0.10 | +1–3 ms | ~270 B/mem | n/a | **ship (V7 obs layer)** |
| 20 | rare-term clue ŵ=argmax 1/df | query seed | a8 O-Mem (paper): seeds lanes/graph from one posting list | 0.1–2 ms | 0 | n/a | **ship** |
| 21 | group→member max-pool (segment granularity) | units | a8: 4 systems independently chose segment units | <1 ms | 0 | n/a | **ship (pack-stage)** |
| 22 | static-embedding lane potion-base-8M int8 (~8.2 MB artifact) | semantic | b9: 82% of MiniLM MTEB-ret at 11 µs/enc, NanoBEIR 0.503 vs BM25 0.452; e3: +2–7 pts est, open-domain +0.10–0.18; my sim: job→work fails under hashing | ~8–12 ms | 8.2 MB + 25.6 MB/100k | MIT | **ship (default semantic tier)** |
| 23 | hashing dense lane fused by RRF | — | my sim: dilutes bm25f 0.41→0.30; can't bridge paraphrase (job_now 0.00) | — | 38 MB | n/a | **free lexical lane / out of fusion** |
| 23b | exact fp32 cosine vs ANN | layout | b10 measured: 1.19 ms mmap packed vs 145 ms per-row SQLite blobs @100k; int8+rescore recall@10=1.0; HNSW recall collapses on thin-shell corpora | 1–12 ms | 38.8 MB int8 | n/a | **ship mmap-packed int8; no ANN** |
| 24 | cross-encoder rerank | rerank | e1: L6 ~11–16 ms/pair ⇒ k_max 8–17 @150ms; L2 ~6 ms/pair ⇒ k=16 fits (104–127 ms); b6: CE→CE distill 96–100% retention | ~60–500 ms | 4–23 MB | Apache ok | **L2 k=16 quality / L6 k≤32 max** |
| 25 | CE pool cap 32 → 300 | — | d1 code-verified: 32 was predict batch; pool=300 = 3.4–5.6 s | — | — | n/a | **reject at 300; ship k-curve {8,16,32}** |
| 26 | ColBERT late interaction | semantic | b7: +5–9 MRR over MiniLM but 0.95 GiB int8/100k + needs PLAID | 5–10 ms (scored) | 0.15–0.95 GiB | MIT | **reject (default); optional max** |
| 27 | SPLADE sparse expansion | lexical | b8: +7–8 nDCG BEIR; license CC BY-NC-SA on naver checkpoints; Apache doc-side variant exists | 1–3 ms | postings +% | NC on best | **reject default / optional** |
| 28 | GBDT/LambdaMART reranker | rerank | b4: only model beating RRF in both sims (+0.08–0.16) but 18–62 ms pure-Py, ≥500–1000 q data | 7–60 ms | 26–277 KB | MIT | **optional (quality/max)** |
| 29 | pointwise logistic reranker | rerank | b4 sim: loses to RRF 92–100% seeds independent noise; gate it | 0.2 ms | 200 B | n/a | **optional gated** |
| 30 | LLM query-time plan/verify/summary | — | a3/a8: violates no-model default; a5: rewritten text breaks byte-pin | — | — | n/a | **reject default** |
| 33 | Fellegi–Sunter merge sieve w/ rarity-weighted partners | write | c1: M=−4.25+Σlog₂(m/u); family-hub partners →~0 weight (two Carolines sharing 'mom'+'david' → −1.48 abstain not merge) | ~0 | aliases tbl | n/a | **ship** |
| 34 | pronoun-class coref sieve ≤3-turn lookback | write+entities | c2: ~62% dialogue pronouns are 1st/2nd person → speaker sieve ~free; deterministic dialogue coref only 30–67% → abstain | ~0 | 0 | n/a | **ship** |
| 35 | slot-keyed consolidation obs=(e,p)+proof pins | write+pack | c5 measured: NO Jaccard threshold separates paraphrase (0.44) from contradiction (0.60–0.67) — slot-key mandatory; watermark freshness 0.002ms | write | ~270 B/obs | n/a | **ship** |
| 36 | dual-time anchor precedence q_ts>session>wall + i32-min columns | temporal | c6 measured: window lane 0.1–0.4 ms@100k B-tree; interval containment per-candidate predicate 0.055ms/40 — never a lane (37% selectivity); rtree rejected 25ms | 0.1–0.4 ms | +32 B/fact | n/a | **ship** |
| 37 | rank_const=1 top-heavy fusion (Σ1/rank) | fusion variant | d3 graphiti code-verified: fixes 'strong lane diluted' but one garbage lane's #1 wins — wrong for noisy lanes | 0 | 0 | n/a | **reject (calibrated-sum instead)** |
| 38 | Jaccard-only dedup/merge threshold | — | c5: paraphrase 0.44 vs contradiction 0.60–0.67 overlap | — | — | n/a | **reject** |
| 31 | per-scope docfreq tables | stats | b1: 2.9–11.5 MB, can't express dynamic purposes | — | MBs | n/a | **reject <1M memories** |
| 32 | saturate-then-combine BM25F | — | b1: 13.6 vs 3.99 inflation worked example | — | — | n/a | **reject** |


## 3. Mapping onto SPEC_V7 §32 tags

| §32 tag | decision | provisional constants verdict |
|---------|----------|-------------------------------|
| `bm25f` | ship FTS5 gen + eligible Python rescore | k1=1.2/b=0.6–0.75 **confirm-range** (grid); field weights {1.0, 1.5, 2.5, facts 1.8, 0.4} **revise** (add facts field); idf=ln(1+x) **new** |
| `retrieval_policy` | eligible-first; scope-restrict; FTS5→rescore→fuse→expand→boost→pack | "eligibility before rank" **confirmed by everything** |
| `boosts` | 1+α·(s−0.5) conditioned on entity/temporal match | α=0.2/0.2/0.1 **confirmed verbatim** (Hindsight identical); add conditioning (my sim: unconditioned recency −0.14) |
| `graph_ppr` | **replaced** by bounded 2-hop on materialized edges_top + 2-step spread | PPR hop/edge weights **reject → α=0.4–0.5, ε=1e-4 spread constants** |
| `rerank_features` | none default; boosts only; gated logistic optional | feature-rerank weights **reject default** (needs ≥300 judged q) |
| `alias` | deterministic canon + Fellegi–Sunter merge sieve | alias rules **replaced**: 0.5/0.3/0.2 additive fails both directions (c1: merges homonyms at 0.84, fails Bob→Robert at 0.586) → M = −4.25 + Σ w_i log-odds, rarity-weighted partners, merge ≥ +4.25 / reject < −2 / abstain→pending_alias (calibrate weights per corpus) |
| `temporal` | composite: validity intervals + resolver prefilter + recency r=0.5^(d/120) as ±10% fold + slot supersession + surfaced ranges | "365-day decay" **confirmed** as max(0.1, 1−d/365) rerank multiplier (not edge weight); naive recency multiplier **rejected** as current-X fix (my sim) |
| `coref` | pronoun-class sieve: "it"→unique non-person, person-pron→unique person, ≤3-turn lookback, abstain on ambiguity | **new**: recovers 67% of pronoun continuations at 0 wrong-attach (my fixture); prev-turn-all rejected (47 noise links) |
| `coref_sieve` | same as `coref` — tag TBD by spec | — |
| (new) `consolidation` | obs=(entity,predicate) + proof pins + watermark | **new** — replaces any text-synthesizing consolidation |
| (new) `semantic_tier` | potion-base-8M int8 default lane | hashing:subword-ngram **demoted** to free lexical lane; exact-cosine packed-int8 layout |

## 4. Provisional constants — confirm / replace / reject

| constant | verdict | evidence |
|----------|---------|----------|
| RRF k=60 | **confirm** (flat) | Cormack; Hindsight code; robust k∈[30,500] |
| lane weights 1/0.75/0.5 | **replace** | Bruch OOD fragility; my sim ≈noise; → flat or MemGAS 1/H |
| CE pool cap 32 | **replace** | 32 was batch size; real cap 300; ship k∈{8,16,32} curve → quality only |
| boost alphas 0.2/0.2/0.1 | **confirm** | identical in Hindsight code |
| PPR hop/edge weights | **reject** | no shipped system uses PPR here; BFS does the job |
| BM25F field weights | **revise** | add facts field; magnitudes are priors pending harness |
| alias rules | **replace** | c1: 0.5/0.3/0.2 fails both directions → Fellegi–Sunter sieve |
| hashing embeddings | **demote** | can't bridge paraphrase (measured job→work 0.00) |

## 5. Unknowns and the experiment that closes each

1. **Calibrated CombSUM vs flat RRF on the real 14-lane set** — Exp: isotonic
   per-lane fit on ≥150 labeled dev queries, held-out any@10. If <+0.02 ship
   RRF flat. (Synth says CC +0.18; sims agree; needs real lanes.)
2. **Facts-field vs facts-lane form** — a6 measured index-merge > rank-merge;
   e4 measured union-lane ceiling +5–12 pts. Exp: ablate both forms on the
   1536-q slice, pick higher cat-2/4.
3. **Smallest CE pool** — e1 modeled k_max≈8–17 @150 ms for L6; L2 k=16 fits.
   Exp: any@10 at k∈{0,8,16,32} on quality profile + one onnxruntime
   benchmark of L2-int8 on the target box to pin true throughput.
4. **2-hop expansion lift on LoCoMo multi-hop** — c4's planted-hop-2 fixture
   proved the mechanism exists; does real cat-1 evidence sit at hop-2? Exp:
   expansion off/1/2 on cat-1 questions; if 2-hop <+0.02 ship 1-hop.
5. **Entity/speaker-scope coverage** — synth says the lever is huge (0.26→0.53)
   but ~all synth questions name the entity; LoCoMo may not. Exp: resolver
   coverage on dev slice; if <60% scope becomes a *boost* not a restriction.
6. **Static-embed lane real delta on open-domain slice** — b9/e3 estimate
   +2–7 pts overall, +0.10–0.18 open-domain; measure on the 1536-q slice.
7. **Fellegi–Sunter weight calibration** — c1's weights came from the
   assignment's own probe cases; needs a merged/not-merged pair set from the
   real corpus to set m/u ratios.
8. **Whether the temporal resolver's empty-fallback cost is acceptable** —
   LongMemEval showed LLM resolvers hallucinate ranges; ours abstains.
   Exp: measure fraction of temporal questions where resolver fires; if <50%,
   prefer surfacing ranges without prefiltering.
9. **Proportional-R@10 vs any@k divergence** — a7: LoCoMo cat-1 averages 3.13
   evidence turns; Verbatim's any@k is hit-rate, the paper metric is
   proportional recall. Exp: report both on the harness before claiming wins.

## 6. Bibliography (role marked)

- Cormack, Clarke, Buettcher SIGIR'09 RRF — *paper* — k=60 robustness, CombMNZ comparison. https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- Bruch, Gai, Ingber ACM TOIS'24 (arXiv 2210.11934) — *paper (vendor-funded)* — convex normalized-score fusion > RRF 9/9, OOD fragility of tuned weights.
- Hindsight TEMPR repo+paper — *vendor + code* — flat RRF, boost alphas, entity merge, BFS deltas, pool-300 correction.
- getzep/graphiti v0.30.2 — *code* — bi-temporal edges, RRF rank_const=1, BFS lane, date-range surfacing.
- Mem0 platform v3 + SDK — *vendor + code* — additive fusion denominator, access-decay band, entity boost, temporal-intent classifier (+4.1 LoCoMo vendor claim).
- LongMemEval ICLR'25 (arXiv 2410.10813) — *paper* — fact-augmented keys +9.4% rel recall, index-merge > rank-merge, time-aware indexing ablation.
- LoCoMo ACL'24 + locomo10.json — *paper + data* — category-id swap verified, proportional R@10, DRAGON retrieval numbers, CC BY-NC 4.0.
- Chronos arXiv 2603.16862 — *vendor* — dual-index event calendar, ablation deltas, no code.
- Mastra OM — *vendor + code* — Observer/Reflector → pinned-span atom mapping, priority ladder.
- MemGAS + MemoryBank + O-Mem + EverMemOS + 17-system sweep — *papers* — entropy lane weights, access decay, rare-term clue, segment granularity, validity intervals.
- Robertson & Zaragoza BM25 primer; Lucene/Anserini/FTS5/rank_bm25 sources — *papers + code* — formula invariance, k1/b defaults, FTS5 BM25F verification, ln(1+x) idf.
- TREC-13/Pérez-Iglesias BM25F — *paper* — per-field saturation variant, sparse-field weight magnitudes.
- Hofstätter'20; RocketQAv2; RankDistil; Zhuang GAM — *papers* — CE distillation retention ladder, Margin-MSE, linear students.
- ColBERTv2 + PLAID — *papers + code* — MaxSim constants, residual codec sizes, pruning cost.
- SPLADE v2/v3 + OpenSearch — *papers + code* — FLOPS reg, doc-side variant, license split.
- RCRank'16; EPV literature; Yahoo LTR; MSLR; ProbFuse; Savoy — *papers* — sample-size floors, learned-fusion gains.
- Fellegi & Sunter (1969) entity-resolution theory — *paper* — log-odds evidence weights, rarity weighting, abstain band.
- Dense X proposition indexing (arXiv 2312.06648); MemInsight (arXiv 2503.21784); Mem0 J paper — *papers/vendor* — facts-vs-turns ceiling and union-form evidence (e4).
- SQLite rtree/btree sources — *code* — i32-min column arithmetic, 0.1–0.4 ms window queries, rtree rejection (c6).
- synth_slice.py + local-calc-synth.md — *calculated* — all slice numbers, CE latency model, PPR-vs-BFS arithmetic, exact-scan sizing.


Notes per topic: `eval/v7/notes/<topic>.md` (+ `_raw_*.txt` dumps).
