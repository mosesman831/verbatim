# e1-ce-kcurve-latency — Cross-encoder pool depth vs p95 on the 4-core reference machine

**Question:** for an ms-marco-MiniLM-L-6-v2-class cross-encoder (~23M params, 6 layers,
hidden 384) scoring (query, turn) pairs at seq len ~96–160 on a 4-core CPU with ONNX
int8: per-pair ms, p95(k, batch), smallest k within ~0.005 of best quality, verdict.

**Answer in one line:** the model is ~2× too slow for the default profile — a
MiniLM-L-2-class head at **k=16 / batch 32 (~87–130 ms p95)** is the largest CE that
fits the ≤150 ms budget; keep L-6 at **k=32** for the quality profile (~365–500 ms p95);
reject k≥128 for latency **and** for accuracy (deep pools actively degrade, see §5).

---

## 1. Candidate formulas — exact equations, parameters, provenance

### 1.1 Compute FLOPs per scored pair

    F(T) = 2 · P_compute · T  +  L · 4 · T² · D

- `P_compute` = parameters that see GEMM work per token = **10.78M**.
  Derived from config `microsoft/MiniLM-L12-H384-uncased` → distilled 6-layer head:
  vocab 30522, D=384, L=6, FFN=1536 → token+pos+seg embeddings = 11.92M (gathers,
  ~zero FLOPs); per layer = 4·D² (qkv+out) + 2·D·F (ffn) + LNs ≈ 1.77M; +pooler/head
  ≈ 0.15M. Total params 22.71M (matches the card's ~23M; fp32 file 86.8 MB / 22.7M ≈
  3.8 B/param ✓).
- `T` = padded pair length ∈ {96, 128, 160}. LoCoMo-style turns + query tokenize to
  ~40–110 tokens; the temsa ONNX repo uses `max_length=128` — adopt 128 worst case,
  96 nominal.
- The prompt's `2·P·T` (P = full 22.7M) double-counts the 11.9M embedding table.
  Both are computed; strict form is used for calibration.

Evaluated:

| T | 2·P·T (prompt model) | F(T) strict | attn share |
|---|---|---|---|
| 96 | 4.36 G | 2.20 G | 85 M (3.9%) |
| 128 | 5.81 G | 2.97 G | 151 M (5.1%) |
| 160 | 7.26 G | 3.76 G | 236 M (6.3%) |

### 1.2 Per-pair latency

    ms_pair = F(T) / (C · g)          C = 4 cores, g = effective GFLOP/s/core

- **g nominal = 45 GFLOP/s/core (AVX2 int8, calibrated)** — see §2. Band: 30
  pessimistic, 75 optimistic (AVX-512 VNNI, ~2× int8 MAC throughput).
- The prompt's stated `2–6 GFLOP/s/core` is treated as a sensitivity bound, not the
  operating point: measured ONNX int8 lands 8–15× above it (§2).

### 1.3 Pool latency and p95

    t_avg(k, B) = k · ms_eff(B) + ceil(k/B) · c0
    p95 ≈ 1.2 · t_avg
    ms_eff(8) = ms_pair,   ms_eff(32) = 0.85 · ms_pair,   c0 = 5 ms/batch

- `c0` = tokenize + session/dispatch overhead per call (one tokenizer pass is shared
  across the batch; ONNX input-packing is the residue).
- p95/avg = 1.2 taken from temsa (249/210 = 1.18, 644/578 = 1.11 → round 1.2).
- Batch-32 throughput +15% assumption is *conservative*: temsa shows ~flat throughput
  from 20→50 pairs/request (95.2 → 86.5 pairs/s, i.e. compute-bound already at B≈20),
  so batch size is a second-order knob; B=32 still preferred (one call, fewer stalls).

### 1.4 Depth-vs-quality model (for the k pick)

    nDCG@10(k) = A · [1 − (1 + k/β)^(−γ)]          (scaled generalized-Pareto CDF)

Elastic's rerank-depth study fits this to BEIR curves; solving their "90 % of max at
~depth 100" plus ~50 % at ~depth 10 gives β≈10, γ≈1 — the params used below for the
within-0.005-of-best estimate. Worked example in §5.

## 2. Calibration — why g ≈ 45, not 2–6

Measured anchors (independent sources):

| Source | Hardware | Model | Measured | Implied per-pair |
|---|---|---|---|---|
| temsa HF repo | i7-9750H, 6C/12T, AVX2 (no VNNI), ORT 1.22.1 | ms-marco-MiniLM-L-6-v2 **QInt8** | 95.2 pairs/s (20/req), 86.5 (50/req) | 10.5–11.6 ms |
| same, upstream fp32 ONNX | same | same, fp32 | 65–68 pairs/s | ~15 ms |
| flashrank-js `mini` | desktop x64, ONNX-quantized, incl. tokenize | same class, shorter docs | 20 docs / 97 ms | ~4.9 ms |
| sbert model card | V100 GPU | same | 1800 docs/s | 0.55 ms |

Derive g from temsa (strict F, 6 cores):
`g = F(T) · 95.2 / 6` → 34 GFLOP/s/core if docs were T=96, 46 if T=128.
So **real ONNX-int8 GEMM throughput on AVX2 is ~35–95 GFLOP/s/core**, not 2–6 —
int8 `pmaddubsw/pmaddwd` (AVX2) and `vpdpbusd` (VNNI) do 2–4× the MACs/cycle of fp32
FMA, and transformer GEMMs keep utilization high because all k pairs share the weight
matrices (one pass over ~11 MB of int8 layer weights per batch). The 2–6 figure would
imply 125–371 ms/pair at T=128 — contradicted by two independent measurements
(10.5 ms and ~5 ms). **Assumption stated: g = 45 GFLOP/s/core on the reference
4-core (AVX2 baseline, no VNNI credit).** Sensitivity: g=30 → +50 % latency, g=75
(VNNI) → −40 %.

Scaled to 4 cores: 95.2 pairs/s × 4/6 ≈ **63 pairs/s ≈ 15.9 ms/pair at T=128** —
this measured-anchored number sits inside the strict-FLOPs range (F(128)=2.97 G /
(4·45 G) = 16.5 ms) — model and measurement agree.

## 3. Worked example (sub-question 1 + mechanics)

Query "coffee shop downtown" against 3 fused turns (T ≈ 60, 90, 128 → padded batch
B=3, T=128):

- F(128) = 2·10.78e6·128 + 6·4·128²·384 = 2.76 G + 0.15 G = **2.97 GFLOP/pair**
- batch work = 3 × 2.97 = 8.9 GFLOP; on 4 cores @ 45 → 180 GFLOP/s → 49.5 ms + 5 ms
  fixed = 54.5 ms avg → **p95 ≈ 65 ms** for 3 pairs (~18 ms/pair incl. overhead).
- 10-pair pool k=16 (T=96 nominal): 16 × 2.20 G = 35.2 G → 196 ms avg → 235 ms p95.

## 4. p95 table — k × batch, 4-core (sub-question 2)

Nominal scenario `ms_pair = 11 ms` (T=96–128 mixed, g=45). Values = avg / p95 ms.

| k | B=8 | B=32 |
|---|---|---|
| 0 | 0 / 0 | 0 / 0 |
| 16 | 186 / 223 | 155 / **186** |
| 32 | 372 / 446 | 304 / **365** |
| 64 | 744 / 893 | 608 / **730** |
| 128 | 1488 / 1786 | 1217 / **1460** |
| 300 | 3490 / 4188 | 2855 / **3430** |

Scenario band for k=16/B=32: optimistic (short turns or VNNI, 7 ms/pair) ≈ 120 ms p95;
pessimistic (T=160, 15 ms/pair) ≈ 251 ms p95.

**Budget cut:** `k_max = floor((150/1.2 − 5)/(0.85·ms_pair))` → k_max ≈ **8–17 for
L-6** depending on scenario vs the ≤150 ms target; the useful floor for CE (must cover
the top-10 window plus spare) is ~16 → L-6 misses the budget in all but the optimistic
case. Provisional constant under review — `cross-encoder pool capped at 32`: **replace
32 → 16 for default (and only with the L-2-class head), keep 32 for quality.**

### Time complexity at 10k vs 100k memories

CE cost is **O(k), not O(N)** — it scores only the post-fusion pool, so the table is
*identical* at 10k and 100k memories: k=16/B=32 → ~186 ms p95 either way. (N affects
only the upstream lanes that *produce* the pool: FTS5 BM25 and candidates scale ~log N
/ sub-linear at these sizes; that is a different workstream's question.) One caveat:
at 100k, cold disk reads for candidate payloads add ~constant per query — outside the
CE line item.

## 5. Depth-vs-quality evidence (sub-question 3)

- **Elastic Labs (BEIR, nDCG@10, BM25 first-stage):** 72.6 % of model×dataset curves
  are Pareto-saturating (fast rise mostly **< depth 100**, plateau after); **20.2 %
  are unimodal** (peak then *decline* — their flagship example is **MiniLM-L12-v2 on
  TREC-COVID**, i.e. this exact family decays early); 7.1 % decay outright. Their
  90 %-of-max rule → ~100 depth averaged across models; compute-constrained choice
  → **~30**. Extrapolation flag: BEIR corpora ≫ Verbatim memories and docs are longer;
  direction is reliable, magnitude is not.
- **"Drowning in Documents" (Databricks, ReNeuIR '25):** across modern rerankers,
  recall *falls precipitously* at large K — CEs emit "phantom hits" (irrelevant docs
  scored highest). Deep pools are counterproductive, not merely wasted.
- **"Optimal Re-ranking Depth" (Glasgow, MS MARCO DEV + LLM judgments):** per-query
  optimal depth exists; oracle picks reduce average depth ~5× *while gaining* +7 %
  effectiveness → fixed-deep k is both slower and worse.
- **Convention:** BEIR CE evals rerank top-100; sbert retrieve&rerank examples use
  ~top-32.

Smallest k within ~0.005 of best: under the fitted Pareto (β≈10, γ≈1, §1.4) with a
typical CE lift A ≈ 0.03–0.10 nDCG, `deficit(k) = A·(1+k/10)^(−1)` → k=64 leaves a
0.5–1.5 % lift deficit (~0.0004–0.0014 nDCG); k=32 leaves ~1.9 % (~0.0006–0.002);
k=128 ~0.7 %. For a **MiniLM-class** head specifically, the curve peaks earlier
(unimodal risk), so **k=32 is the defensible smallest-in-band pick; k=64 is the
conservative pick** — both far inside the ~100 saturation zone, and both on the safe
side of the decay regime. Marked **extrapolation** — validate on the LoCoMo eval when
the post-fix engine lands.

## 6. Index / resident size

The CE adds **zero bytes per memory** — it is a single global artifact (hash-pinned
local file, satisfies the "no hosted service" rule): int8 ONNX weights ≈ **22.1 MB**
(temsa artifact; fp32 86.8 MB if unquantized). Runtime resident ≈ 40–70 MB RSS
(weights + ORT arena + tokenizer), amortized once per process, same at 10k/100k
memories. No index build on the write path — add-ack stays model-free.

## 7. Verbatim stage

Augments the **optional rerank stage**, post score/rank fusion, pre context-pack:
take the fused list's top-k, score (query, turn) pairs, re-order, feed the byte
budget. Byte-pin unaffected (CE only re-orders retained items; it generates nothing).
Apply CE scores as `final = fused_score + w_ce · σ(ce_logit)` or pure re-order —
recommend pure re-order within pool then RRF-consistent merge, to keep eligibility/
fusion invariants.

## 8. Comparison table

| Candidate | Params | ms/pair (4C, nom) | k=16 B32 p95 | k=32 B32 p95 | TREC DL19 nDCG@10 | Verdict |
|---|---|---|---|---|---|---|
| No CE (k=0) | — | 0 | — | — | BM25 baseline | ship (status quo default) |
| MiniLM-L-2-v2 CE | 15.4M (3.7M compute) | ~6 | **~104–127 ms** | ~202–254 | 71.01 | **ship for default @ k=16** |
| TinyBERT-L-2-v2 CE | 4.4M | ~2.5 | ~55 ms | ~100 ms | 69.84 | ship-alt if tighter budget needed |
| MiniLM-L-6-v2 CE | 22.7M (10.8M) | ~11–16 | ~186–300 | ~365–500 | 74.30 | **optional (quality) @ k=32**; reject default |
| L-6 @ k=64/128 | same | ~11–16 | — | 730–1980 | ≈ plateau | optional-max only; diminishing |
| L-6 @ k=300 | same | ~11–16 | — | 3.4–5.6 s | likely **worse** (decay) | **reject** |

## 9. Verdicts

- **k=0 (no CE) → ship** default baseline; confidence high. What changes it: CE quality
  win on LoCoMo > ~0.02 any@10 would argue for shipping L-2 default unconditionally.
- **L-6-class @ default → reject**: 186–300 ms p95 at the smallest useful k=16 busts
  the ≤150 ms target ~1.3–2×; only VNNI+short-doc scenarios squeeze under.
  Confidence medium-high — a 4-core VNNI box at T≤96 (~120 ms p95) is the one
  configuration that could flip it; measure on the actual reference CPU.
- **L-2-class @ k=16, B=32 → ship** for default: ~104–127 ms p95 fits, keeps CE gain
  in the top-16 window; costs ~10 % relative MRR@10 vs L-6 (34.85 vs 39.01).
  Confidence medium — hinges on the L-2 CE actually lifting any@10 ≥ ~0.01 on the
  post-fix pipeline; the 0.546→? gap vs plain BM25 (0.596) makes CE-on-BM25-lane
  worthwhile only if the pool itself is good.
- **L-6 @ k=32, B=32 → optional** (quality/max profile): ~365–500 ms p95 buys the
  smallest-k-within-~0.005-of-best per §5. Confidence medium — saturation point is an
  extrapolation from BEIR; confirm on LoCoMo eval.
- **k≥128 → reject**: ≥1.5 s p95 for <1 % expected lift; and k=300 → reject twice over
  (~3.4–5.6 s + measured quality regression at depth).
- **Batch: B=32** (single call); sensitivity is small (temsa: −9 % throughput 20→50)
  but single-call avoids dispatch stalls; confidence high.
- **Provisional constants:** replace "pool capped at 32" → default 16 (with L-2 head)
  / quality 32 (L-6). Confirm `k=60 RRF` upstream unchanged (out of scope).

## 10. UNRESOLVED

- Exact seq-len/token mix of temsa's benchmark docs (I assume T≤128, stated in §2) —
  closes by running the quantized ONNX on the actual 4-core box at real LoCoMo turn
  lengths: the single measurement `ms_pair` decides between k=8 and k=16 for default.
- The 0.005-band k is corpus-dependent (lift A unknown until post-fix eval) — closes
  with one eval sweep k ∈ {8,16,32,64} on the fixed pipeline.
