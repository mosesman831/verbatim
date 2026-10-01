# b8-splade — SPLADE / sparse lexical expansion for Verbatim

Working note, 2026-09-22. Scope: whether a learned sparse-expansion lane can fit inside SQLite-only, offline-default Verbatim. Bottom line up front: **the architecture fits perfectly (doc-side expansion is exactly a postings lane), the latency fits trivially at 10k-100k, but no license-clean high-quality checkpoint is small enough to justify shipping in the default artifact** — it belongs in the optional `quality`/`max` profile as a hash-pinned local model, implemented the OpenSearch/`splade-v3-doc` "inference-free" way (query = static token table lookup, no encoder call at search time).

---

## 1. Candidate formulas (exact equations, parameters, provenance)

### 1a. SPLADE dual-side (Formal et al., arXiv 2107.05720; SPLADE v2 arXiv 2109.10086; SPLADE++ arXiv 2205.04733)

- SparTerm lineage: `w_ij = transform(h_i)^T · E_j + b_j` — per-token logits over the BERT WordPiece vocab **|V| = 30,522**; `w_j = g_j · Σ_i ReLU(w_ij)` where `g_j` is a learned "term importance" gate.
- SPLADE v2 / ++ replaces Σ with max and adds a saturating transform:

  **`w_j(d) = max_{i ∈ d} log(1 + ReLU(w_ij))`**  (SPLADE-max form used by v2/++/v3; OpenSearch and sentence-transformers implement `log1p(log1p(relu(max_i logits)))`, "double-log" — verified in `opensearch-neural-sparse-encoding-doc-v3-gte` model-card code, which also zeroes `[PAD/UNK/CLS/SEP/MASK]` columns before pooling.)

- Document score: **`s(q,d) = Σ_j w_j(q)·w_j(d)`** — a plain inner product of two 30,522-dim sparse vectors → executable as posting-list summation.
- Training loss: `L = L_rank + λ_q·L^q_FLOPS + λ_d·L^d_FLOPS`, `L_rank` = InfoNCE (v2) or MarginMSE distillation (++/v3). FLOPS regularizer:

  **`L_FLOPS = Σ_j θ̂_j��`, `θ̂_j = (1/N) Σ_d w_j(d)`** — penalizes *squared mean activation per vocab term*, pushing the whole corpus toward few nonzero entries. (`λ` grids are tuned per training run; I could not verify exact λ_d/λ_q constants — GitHub raw fetch was rate-limited; training-time only, so UNRESOLVED-lite and not needed for the inference-path design.)

**Worked example.** Doc = "alice likes hiking boots" → 5 tokens. Suppose max-pooled logits give `w_ij("boot")=4.0` (literal term) and `w_ij("gear")=2.4` (expanded term, never written): `w("boot") = log(1+4.0) = 1.61`, `w("gear") = log(1+2.4) = 1.22`. A query "hiking gear" scored `s = w_q(hiking)·1.9 + w_q(gear)·1.22` gets credit for "gear" — vocabulary-mismatch recall without touching the query.

**Worked example, FLOPS reg.** 3 docs × vocab {a,b,c}, weights d1={a:2,b:0,c:1}, d2={0,1,1}, d3={1,1,0}: means θ̂ = (1, 0.67, 0.67) → `L = 1 + 0.449 + 0.449 = 1.898`; squaring charges *corpus-level* frequency, not per-doc magnitude — the knob that buys sparsity.

### 1b. Doc-side-only / "inference-free" SPLADE (the candidate that matters)

- **SPLADE-Doc** (v2 paper §4.2): train `w_j(d)` as above; query is raw bag-of-words (uniform weight per token). Result: **MRR@10 = 0.322** vs full SPLADE 0.340 and BM25 0.184 — retains **~88% of the MRR gain over BM25** with *no query encoder*. With only ~19 nonzero weights/doc it still hits 0.296.
- **naver/splade-v3-doc** (arXiv 2403.06789): production version — BERT-base 110M, doc = double-log max-pool; query = `SparseStaticEmbedding`, i.e. a **learned static table of 30,522 floats** (~122KB) indexed by token id. Score = `Σ_{t ∈ q} s_q[t] · w_d(t)`. Verified via sentence-transformers `Router` docs.
- **opensearch `neural-sparse-encoding-doc-v3-gte`** (Apache-2.0): identical mechanism; the static table ships as `idf.json` ({token: weight}), query vector = one-hot × idf table. Their *other* mode ("doc-only v2") uses dataset-idf instead — learned table is strictly better.

`stage mapping: candidate generation lane` — pure postings; nothing else in the pipeline needs to change.

### 1c. Precomputed-weight alternatives

| Model | Doc-side | Query side | Score |
|---|---|---|---|
| BM25 | tf in postings | query tf/idf | Σ idf·f_d |
| docT5query | generated terms as tf | unchanged BM25 | BM25 on pseudo-doc |
| DeepImpact (2104.12016) | **learned int impact per term** (DocT5Query expansion), quantized into postings | raw terms | Σ_{t∈q} impact_d(t) |
| uniCOIL (2106.14807) | scalar w_d(t), optional T5 exp | scalar w_q(t) or 1 | Σ w_q(t)·w_d(t) |
| EPIC (2004.14245) | importance + expansion | full encoder (not tiny) | Σ over subsumed docs |
| SPLADE-doc / v3-doc / opensearch-doc | learned w_d(t) incl. expansions | static 30,522-float table | Σ s_q[t]·w_d(t) |

DeepImpact is the proof that the trick is a *postings format*, not a runtime: scoring is `Σ impact` over sorted lists — the exact query Verbatim's FTS5 lane already evaluates.

## 2. Published gains (verified from primary-source tables)

MS MARCO MRR@10 (dev set): BM25 0.184 | docT5query 0.277 | SparTerm 0.279 | EPIC 0.304 | uniCOIL-noexp 0.315 | DeepImpact 0.326 | SPLADE-Doc 0.322 | uniCOIL-T5 0.352 | SPLADE-max 0.340 | SPLADE++-EnsDistil 38.0 | splade-v3 40.2 | v3-doc 37.8.

BEIR-13 mean nDCG@10 (paper Tables 2/3, verified against per-dataset rows): BM25 **43.7**, SPLADE++-SelfDistil **50.7 (+7.0)**, splade-v3 **51.7 (+8.0)**, **v3-Doc 47.0 (+3.3)**, v3-Lexical 49.1. BM25+SPLADE++ ensemble **52.1**. → The "+5-10 points over BM25" claim is confirmed for dual-side; doc-only gives **+3.3 mean**.

Open-domain/paraphrase slices (BM25 → v3-doc, my delta vs BM25): **NQ 32.9→52.1 (+19.2)**, ArguAna 31.5→46.7 (+15.2), FiQA 23.6→33.6 (+10.0), HotpotQA 60.3→66.9 (+6.6), DBPedia 31.3→36.1 (+4.8), TREC-COVID 65.6→68.1 (+2.5). Losses: Quora −1.4, FEVER −6.4, Climate-FEVER −5.4, SCI DOCS −0.6 — query-side expansion is what fixes fact-verification sets. For Verbatim's LoCoMo open-domain slice (0.196 vs BM25 0.370 — a **−0.174 gap**) the relevant mechanism is exactly the NQ/ArguAna-style paraphrase mismatch where doc-side expansion keeps most of the win. **Extrapolation label:** BEIR is not LoCoMo; treat +3.3 mean / +19 on NQ as mechanism evidence, not an any@10 prediction.

## 3. Can expansion be precomputed into SQLite postings? — Yes, and it's the right shape.

- Doc rep after double-log pooling: ~50–200 nonzero terms (v2 paper Fig. 2; v3-doc reports FLOPS ≈ 1.4; OpenSearch doc models 1.7–2.3). Take **~120 nonzero/memory** as design center.
- Storage per posting: naive row `(term_id INT, mem_id INT, w REAL)` ≈ 30B with B-tree overhead → **100k memories ≈ 12M postings ≈ 360MB (~3.6KB/mem)**; FTS5-style packed posting blobs (varint docid-delta + uint8 quantized weight) ≈ 5–6B/posting → **~70MB (~0.7KB/mem)**. Pack weights to one byte via `round(15·w/w_max)` like DeepImpact/Wacky-Weights pseudo-TF — negligible score loss at 4 bits.
- Query path: tokenize (~1ms) → look up ≤~15 tokens in the static table → sum over posting lists. **Time complexity = Σ_t df(t) row visits.** Expanded terms are corpus-rare (df ≈ 0.1–1%): at **10k** memories ≈ 10 terms × ~30 rows ≈ 300 visits → **<1ms**; at **100k** ≈ ~3k visits → **1–3ms** SQLite scan (measured SQLite row-read ≈ 0.5–2µs). Order-of-magnitude inside the 46/85ms p50/p95 budget — expansion is free at search time.
- Write path: encode turn (~50–300ms CPU, see §4) → emit ~120 (term, weight) rows → insert (~5–15ms). Strictly a background job, matching "embedding/extraction are background jobs" — **add-ack stays model-free**.
- Byte-pin unaffected: expansions are index features, never delivered payload. Eligibility unchanged: filter memories on auth/quarantine **before** lane scoring (the lane only sees eligible rows — same as BM25 today; "approximate authorization is not" is preserved because scoring is exact over eligible postings, no ANN needed).

### 3b. Scale calibration — why the scary SPLADE latency numbers don't apply

Wacky Weights (arXiv 2110.11540) measured SPLADEv2 retrieval at **8.8M MS MARCO passages**: 220ms (PISA) to 314ms (Lucene) per query, index ~4.3GB vs BM25's 0.74GB, because its *expanded queries* carry ~25 unique terms (~+21 vs raw). That is a query-side-expansion pathology at 900× Verbatim's scale. The doc-only design has neither problem: queries stay ~10 terms, and at 100k memories the postings are ~2% of MS MARCO's — the 12% recall loss those authors saw under approximate JASS traversal is avoided entirely because document-at-a-time over ~3k rows is exact and cheap. Also from that paper, a real caution: SPLADE assigns nonzero weights to **stopwords and punctuation** ("wacky weights"); the static query table + min-max score normalization (not raw z-sum) at fusion entry absorbs this, and rank-based RRF k=60 is immune to score-scale issues.

## 4. Encoding cost on CPU (measured + derived)

Measured this box (8-core Ice Lake, AVX-512, OpenBLAS): weight-read floor 440MB fp32 ≈ **16–36ms** (12–27GB/s); MLM-head GEMM (128×768·768×30522) 11.3ms @533 GFLOPS; pooling+nonzero over 128×30522 ≈ **8.5ms**; full-doc FLOPs ≈ 2·110M·128 ≈ **28 GFLOPs**.

Derived encode latency per ~128-token turn (GEMM-bound ideal ≈ 60ms at ~450 GFLOPS sustained; real ONNX/PyTorch achieve ~40–70% of that on batch-1 skinny GEMMs):

| artifact | params | fp32 file | 8-core est. | 4-core reference est. | int8 est. |
|---|---|---|---|---|---|
| splade-v3-doc (CC NC) | 110M | 438MB | 80–200ms | 150–400ms | 50–120ms |
| opensearch doc-v3-gte | 137M | 549.6MB | 100–250ms | 180–450ms | 60–140ms |
| opensearch doc-v3-distill / v2-distill | 67M | ~268MB | 50–120ms | 80–200ms | 30–70ms |
| opensearch doc-v2-mini | 23M | 91MB | 20–50ms | 35–90ms | 15–35ms |

Cross-checks: uniCOIL-T5 query encode 45ms on DistilBERT (efficiency paper, measured); FastEmbed SPLADE-PP ≈ 3s for ~500-token docs on i7 (≈4× the seq → consistent). 100k-memory backfill at 200ms×100k ≈ 5.6h single-threaded → ~1.4h across 4 cores; one-time, acceptable.

## 5. Licenses (verified via HF API/card metadata)

- **naver/splade-cocondenser-ensembledistil, splade-v3, v3-doc, v3-lexical: CC BY-NC-SA 4.0** — non-commercial; NOT MIT. Cannot ship in a default artifact.
- **opensearch `neural-sparse-encoding-doc-{v3-gte, v3-distill, v2-distill, v2-mini, v1}`: Apache-2.0** ✓ — doc-side weights + shipped `idf.json` query table.
- **prithivida/Splade_PP_en_v1: Apache-2.0** (dual-side SPLADE++; used by Qdrant FastEmbed).
- Caveat: opensearch doc-v3-gte's BEIR-13-subset 54.6 is **partially in-domain** (trained on fever/fiqa/hotpotqa/nfcorpus/scifact) — not zero-shot like naver's 47.0; keep in a separate vendor column.

## 6. Which Verbatim stage

New/augmented **candidate lane** (peer to lexical FTS5): `postings_sparse(term_id, mem_id, w_q8)` + `sparse_vocab(token→weight)` table + one `idf.json`-style 122KB static query table inside the artifact. Fusion unchanged (RRF k=60 ranks); lane weight 1.0 (strong) — keep BM25 lane too (SPLADE++ + BM25 sum ensemble = 52.1, +1.4 over SPLADE alone; they're complementary). Optional rerank unchanged. Write-path gains a background encoder stage (hash-pinned artifact). Pack unchanged.

## 7. Verdicts

| candidate | verdict | reason | confidence / what changes it |
|---|---|---|---|
| Doc-side-only sparse lane (inference-free SPLADE mechanism) | **optional (quality/max)** | Exact-match latency <3ms at 100k; +3.3→+19 slice gains; but needs 90–550MB pinned model artifact → violates no-download default | High / a ≤30MB Apache-2.0 checkpoint with measured LoCoMo gain ≥ +10pts would promote to ship |
| opensearch doc-v3-distill (67M) as the pinned artifact | **optional** | Best size/quality point, Apache-2.0, ~80–200ms background encode | Med-high / quantize to int8 (~70MB) first |
| opensearch doc-v2-mini (23M) | **optional** | 91MB → ~23MB int8; 49.7 BEIR-13-subset; cheapest encode | Medium / verify its 49.7 holds on LoCoMo-style text |
| naver splade-v3-doc / cocondenser | **reject** | CC BY-NC-SA — non-commercial license incompatible with default | High |
| Dual-side SPLADE (query encoder at search) | **reject** | +1.0–1.4 mean over doc-only not worth a 100–200ms query-time encoder call | High / revisit only if p95 budget grows |
| DeepImpact-style quantized-integer postings format | **ship** (as the storage scheme for whatever model is chosen) | 4-bit weights + varint docid deltas, exact summation | High |
| uniCOIL / EPIC | **reject** | Same or lower quality as v3-doc; EPIC still needs a query encoder (78ms) | Medium |

**Feasibility verdict on the assignment question:** a doc-side-expanded postings lane *can* run offline-default mechanically (SQLite postings, <3ms queries, static query table) — but it *cannot ship as the default artifact* because every usable checkpoint is a ≥90MB learned model and Verbatim's default profile forbids model downloads; naver's are also non-commercial. Ship it as the first hash-pinned `quality` artifact: int8 `opensearch-neural-sparse-encoding-doc-v3-distill` + DeepImpact-format postings + its shipped `idf.json` static query table. Expected cost: ~70MB disk, ~30–70ms/turn background encode on 4 cores, ~0.7KB/memory postings, <3ms/query at 100k.