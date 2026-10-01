# e4-fact-index-ceiling — working note

Question: what is the ceiling of a quote-pinned typed-fact index vs raw turn index for Verbatim retrieval (any@10, LoCoMo categories 1–4)?

Bottom line up front: **facts-only is a strict ceiling-reducer (~0.85× raw coverage, independently published); facts-PLUS-turns union is the only viable shape, with a defensible any@10 ceiling of ≈0.66–0.72 vs the ~0.60 turn-only baseline — i.e. +5 to +12 pts. Turn the T2 fact lane on as a union member; keep it never a replacement.**

---

## 1. Evidence inventory (vendor vs independent, kept separate)

| Source | Type | Retrieval-metric evidence | Answer-metric evidence |
|---|---|---|---|
| LoCoMo paper Table 3 (Maharana et al. 2024, arXiv:2402.17753) | **independent** | R@k raw dialog vs extracted "observations" (assertions/facts): k=5 58.8 vs 49.6; k=10 67.5 vs 57.1; k=25 79.9 vs 66.0; k=50 84.8 vs 71.1 | F1 flips at small k: obs 41.4 vs dialog 31.7 @k=5; 38.8 vs 34.6 @k=10. Summaries: R@10 90.7 but F1 31.5 |
| Dense X Retrieval (Chen et al., EMNLP 2024, arXiv:2312.06648) | **independent** | Proposition-level index vs 100-word passage index, same retriever: avg R@20 +10.1 (unsupervised: SimCSE 49.1→61.7, Contriever similar); +2.2 (supervised: GTR 78.4→79.2). Largest gains on long-tail entities (EntityQuestions) | EM@100-token reader budget: +5.9 to +7.8 unsupervised, +4.9 to +6.9 supervised |
| Mem0 (arXiv:2504.19413) | **vendor (mem0.ai)** | none reported at recall level | J overall: 66.88 vs best RAG-over-raw-chunks 60.97 (k=2×256 tok) → +5.9 pts (~+9.7% rel) at 1,764 retrieved tokens vs ≤16,384 |
| Hindsight (arXiv:2512.12818) | **vendor (Vectorize)** | none at recall level | LoCoMo J: 83.18 (OSS-20B), 85.67 (OSS-120B), 89.61 (Gemini-3 answerer) vs Memobase 75.78, Zep 75.14, Mem0 66.88; LongMemEval 39→83.6 vs same-backbone full-context. Architecture confirms Verbatim's shape: 4 parallel lanes (semantic, BM25, graph, temporal) → RRF k=60 → ms-marco-MiniLM-L-6-v2 CE → token-budget pack |
| MemInsight (arXiv:2503.21760, Amazon) | **independent** | LoCoMo recall@5: turn-index + extracted attribute annotations 44.9 vs DPR-over-raw 26.5 → +18.4 pts (+69% rel); multi-hop 31.4→63.6, open-domain 15.4→53.4 | F1 30.1 vs 28.7 @k=5 |

Reading: every retrieval-recall comparison where facts **replace** raw text shows coverage loss (LoCoMo: −9.7 pts R@5, −10.4 R@10, −13.7 R@50). Every granularity/augmentation comparison where fine-grained units **coexist** with raw text shows a gain (Dense X +10–13 R@20 unsupervised; MemInsight +18 recall@5; Mem0 +6 J end-to-end). **Extraction loses coverage, granularity wins density; keep both.**

## 2. Candidate formulas, parameters, provenance

Let r_t = turn-lane any@k (measured: BM25 0.596@10, 0.677@20 on 1536 q); p = P(evidence turn yields ≥1 fact whose byte-pinned quote covers the evidence span); r_f = P(fact hits top-k | fact exists); κ = P(turn also hits | fact hits).

**F1. Facts-only ceiling:** C_fo = p · r_f
- p: LoCoMo obs/dialog coverage ratio anchors selective extraction at ≈0.84–0.85 (57.1/67.5 @10; 71.1/84.8 @50). Rule/oracle SPO extraction on declarative evidence: est. 0.80–0.90. LLM selective extraction (Mem0-style): 0.6–0.8 (Mem0 kept ~6.8% of conversation tokens, intentionally lossy). Quote-pin reduces p slightly vs freeform extraction because anaphora-dependent facts can't be rewritten — those must be carried by typed S/O fields (F4).
- r_f: granularity lift vs turn units, from Dense X: +10–13 R@20 abs. on unsupervised retrievers (the BM25 analog). Estimate r_f ∈ [0.75, 0.88] where r_t ≈ 0.60.
- Worked: C_fo = 0.8 × 0.82 ≈ 0.66 at p=0.8; 0.6 × 0.82 ≈ 0.49 at p=0.6. Facts-only can at best match, never exceed, the union — below p≈0.75 it is below raw turns alone. Sim: 0.54/0.68/0.79 at p=0.6/0.8/1.0.

**F2. Union ceiling (facts + turns, per-question best):** C_u = r_t + p · r_f · (1−κ)
- κ ∈ [0.6, 0.85]: same evidence tokens drive both lanes — overlap is high but not total; facts recover via dedup, term concentration, typed S/O joins on weak-lexical queries.
- Worked (conservative, κ=0.8): C_u = 0.60 + 0.8·0.82·0.2 ≈ 0.73. Worked (optimistic, κ=0.65): 0.60 + 0.8·0.82·0.35 ≈ 0.83.
- Cross-checks: sim oracle-union = 0.78–0.82; MemInsight augmentation-union +18.4 recall@5; Mem0 extraction vs raw-chunk RAG +5.9 J. Realistic band 0.66–0.75; point estimate ≈0.70.

**F3. RRF-fused (what Verbatim ships, not oracle):** C_rrf ≈ C_u − δ_f, δ_f ≈ 4–10 pts rank-dilution. Sim RRF k=60, w=1.0/1.0, depth-40 → 0.65–0.70 vs oracle-union 0.78–0.82. To recover union inside RRF: facts lane weight 1.0 (strong lane) and dedup fact-vs-turn at pack via fact→turn_id.

**F4. Typed entity-exact join:** adds ε_e ≈ +1–3 pts on possessive/case/anaphora queries — query "Caroline pottery" joins on subject=caroline regardless of quote surface ("I picked up pottery"), refunding the byte-pin's anaphora loss. Unpriced in published numbers — Verbatim-internal estimate.

**F5. BM25 length-normalization mechanics (why r_f > r_t on covered set):** term weight at tf=1: w = 2.2/(1 + 1.2(0.25 + 0.75·dl/avgdl)) (k1=1.2, b=0.75). Fact dl=10=avgdl → w=1.00; turn dl=90, avgdl=30 → w=0.55. At LoCoMo's ~30-token turns the effect shrinks — dominant gains are dedup + term-concentration, not raw length-norm. Caveat: the assignment's "10-token facts vs 100-token turns" exaggerates; turns average ~30 tokens.

**F6. Dedup multiplier:** evidence stated r times → r competing turn rows vs 1 fact row. Rank effect small in sim (turn any@10 0.591→0.577 as dup 1→4); real value is pack budget — one ~12-tok fact vs r ~30–90-tok turns frees slots for other evidence.

**F7. Detection threshold:** n=1536, McNemar on ~230 discordant pairs: +3-pt true lift ≈ 46 net wins → z ≈ 3.0 (p<0.01); +2 pts → z ≈ 2.0, borderline. Ship threshold = Δany@10 ≥ +3 pts; +1.5–2.9 inconclusive → keep off.

## 3. Per-query cost (measured, 8-core box; 4-core ref ≈ ×1.3)

Real SQLite FTS5 (unicode61), synthetic LoCoMo-like text (/tmp/e4/fts.py):

| Component | 10k memories | 100k memories |
|---|---|---|
| turns_fts MATCH top-40 | 0.48 ms p50 / 0.52 p95 | 4.8 / 5.8 ms |
| facts_fts MATCH top-40 (~70k rows) | 0.26 / 0.28 ms | 2.4 / 2.5 ms |
| entity join (subject/object idx, LIMIT 40) | 0.01 ms | 0.01–0.02 ms |
| RRF merge, 2×40 | ~0.05 ms | ~0.05 ms |
| **facts lane added total** | **≈0.3 ms** | **≈2.5 ms (≈3.3 ms @4-core)** |

FTS5 cost grows ~linearly with corpus (0.26→2.4 ms per 10×). <4 ms at 100k — trivial vs tens-of-ms p95 and ≤150 ms CE budget. Write path is background → zero query cost.

## 4. Index/resident size (measured)

turns-only FTS5 @100k: 24.9 MB → 249 B/memory. facts-only FTS5 @250k facts: 41.3 MB → 165 B/fact → ~413 B/mem incremental at 2.5 facts/mem. Facts lane adds ~165–250 B/fact total (FTS5 ~165 B + meta ~60 B + indexes ~25 B) → **+250–500 B/memory**, i.e. ~25–50 MB at 100k memories. Trivial.

## 5. Coverage parameter p — the load-bearing estimate

p = p_has_fact × p_quote_covers. Anchors: LoCoMo's extracted-observation index retains 84–85% of raw coverage (independent); PropIndex/Dense-X-style complete proposition decomposition → ~0.9+ but assumes non-selective extraction and context-rewritten props (byte-pin forbids rewrite). Estimates: p_oracle ≈ 0.80–0.90; p_LLM-selective ≈ 0.60–0.80; p_rules-on-dialog ≈ 0.55–0.75. UNRESOLVED-residual: exact p on Verbatim's corpus unmeasurable without LoCoMo data (CC BY-NC) — hence the decision rule is coverage-driven, not coverage-assumed.

## 6. Verdicts

1. **Facts-only lane → reject.** Published coverage cap ≈0.85× raw; arithmetic 0.8·0.82≈0.66 ceiling at best. Confidence: high.
2. **Facts+turns union lane (typed SPO + byte-pinned quote, RRF w=1.0, fact→turn_id pack dedup) → ship (default).** Ceiling ≈0.66–0.75 (+5–12 pts); +0.3–3.3 ms; +250–500 B/mem. Confidence: medium-high; magnitude extrapolated — verify on the 1536-q eval.
3. **LLM-based T2 extraction vs rules → optional (quality/max).** On iff Δany@10 ≥ +3 pts over rules and p_rules < 0.6. Confidence: medium — vendor delta on a different metric.
4. **Entity-exact S/O joins + current-flag → ship.** ~0.02 ms; refunds anaphora gap. Est. +1–3 pts. Confidence: medium.
5. **Dedup-by-state-key → ship.** Pack-token economy (~3–8× fewer bytes per evidence item); a pack-stage win. Confidence: medium.

## 7. Coverage check vs assignment

(1) why facts win — dedup, length-norm (0.55→1.0 tf weight), entity-exact joins, current flags — F4–F6. (2) why they lose — extraction coverage p + quote-pin (no contextual rewrite). (3) ceiling — §1 table (vendor/independent separated) + F1/F2 arithmetic + sim (facts-only 0.54–0.79; union 0.78–0.82; fused 0.65–0.70). (4) decision rule — Δany@10 ≥ +3 pts (z≈3, n=1536) justifies LLM extraction on quality profile; fact lane ships by default as union member.