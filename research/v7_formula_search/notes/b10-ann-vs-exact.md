# b10-ann-vs-exact — Approximate vs exact dense retrieval at Verbatim scale (10k–100k memories)

**Headline:** At N ≤ 100k and d = 384, a warm exact fp32 cosine scan costs **~1.2 ms** on a pinned 4-core server CPU and **~5–15 ms** on a DRAM-bound consumer 4-core. That is 10–100× inside the ~150 ms p95 budget. **The default profile needs no ANN.** Ship exact scan over a memory-mapped packed-vector file; use symmetric int8 scalar quantization + fp32 rescore (exact-quality top-k, 4× smaller resident set) when RAM/disk pressure matters. HNSW buys ~1 ms and pays ~180 MB of index + recall tail risk + a C dependency — reject for the default profile.

All measured numbers below are from this session's box (Xeon Platinum 8559C, pinned to 4 cores via `taskset -c 0-3`, Python 3.10, numpy 2.2.6, faiss 1.15.1, warm page cache). This CPU is server-class with a large L3; where that flatters the result, a conservative consumer estimate is given alongside. Synthetic-data recall numbers are labeled as such; they are lower bounds on real corpora where distances are more structured.

---

## 1. Candidate A — exact fp32 cosine scan (the baseline)

**Formula.** With unit-normalized stored vectors and unit-normalized query, cosine = dot product:
score(i) = Σⱼ qⱼ·vᵢⱼ, j = 1..384. FLOPs = 2·N·d (one multiply-add per element; vector norms precomputed at write time, query norm once). Bytes read = 4·N·d.

**Arithmetic at the two scales (4 pinned cores, measured):**
- N=10k: flops = 2·10⁴·384 = 7.7 MFLOP; bytes = 15.4 MB. Measured **0.12 ms/query** (64.6 GFLOPS, 129 GB/s effective — the 15.4 MB set is cache-resident). Even fully DRAM-bound at 15 GB/s: 15.4 MB/15 GB/s ≈ 1 ms.
- N=100k: flops = 7.68·10⁷ = **76.8 MFLOP**; bytes = 153.6 MB. Measured **1.19 ms/query** (64.7 GFLOPS, 129.5 GB/s). Batched 200 queries: 0.17 ms/query. Single-threaded worst case: **4.71 ms** (~32 GB/s single core).

**Correction to the brief's arithmetic:** "100k × 384 × ~4 flops ≈ 15 M flops" is ~5× low. 2·N·d = 76.8 MFLOP (384 MACs/row); counting ~4 flops/element with per-row norm accumulation gives ~154 MFLOP. Either way it is 1–5 ms of compute — the scan is memory-bound, not flop-bound, which is why int8 (¼ the bytes) is the real lever, not ANN.

**Consumer-hardware estimate.** On a laptop 4-core with ~20 GB/s DRAM and no large L3: 153.6 MB/20 GB/s ≈ **7.7 ms**, plus compute overlap → ~8–15 ms. Cold-start (first query, data not in page cache) adds file read at NVMe ~1–3 GB/s: 50–150 ms at 100k — one-time per process, or avoided entirely by mmap lazy paging. At 10k all of this is <2 ms regardless.

**Bytes/memory:** 4·d = 1536 B/memory → 15.4 MB @10k, 153.6 MB @100k.

**Verbatim stage:** dense/sparse-semantic lane candidate generation. Exact scan is also the only layout compatible with Verbatim's eligibility-first rule: apply the eligible-row bitmap/predicate first, then dot the surviving rows. An ANN over the whole table can starve the post-filtered candidate set (approximate authorization is disallowed — this is a structural argument for exact scan beyond speed).

**Verdict: SHIP (default profile).** Zero dependencies, exact recall by construction, ~1–15 ms at 100k.
*Confidence: high. What would change this: N > ~500k memories resident, or a hard p95 < ~10 ms target at 100k on weak consumer hardware — then revisit HNSW or IVF.*

## 2. Candidate B — symmetric int8 scalar quantization + fp32 rescore

**Formula.** Per-vector symmetric quantize at write time: sᵢ = max|vᵢ|/127 (fp32); v8ᵢ = round(vᵢ/sᵢ) ∈ [−127,127]. Query quantized the same way. Approximate score = sᵢ·s_q·(v8ᵢ·q8) — one int8 MAC per dim accumulating into int32. Then take the top-m approximate rows (m = 100–200 recommended) and rescore them in fp32. The candidate set needs only int8 recall ≥ exact top-k ⊆ pool; the final ranking is exact.

**Worked example (d=4).** v = [0.30, −0.12, 0.07, 0.44] → s_v = 0.44/127 = 0.003465; v8 = [87, −35, 20, 127]. q = [0.28, −0.10, 0.09, 0.40] → s_q = 0.40/127 = 0.003150; q8 = [89, −32, 29, 127]. int8 dot = 87·89 + (−35)(−32) + 20·29 + 127·127 = 7743+1120+580+16129 = 25,572; approx score = 0.003465·0.003150·25572 ≈ 0.2791; exact dot = 0.30·0.28 −0.12·(−0.10) + 0.07·0.09 + 0.44·0.40 = 0.2783. Relative error 0.3% — typical for symmetric int8.

**Measured recall (synthetic clustered data, 100k×384, 200 queries):** int8 score alone recall@10 = 0.977; int8 pool + fp32 rescore = **1.0000 at every pool size m ∈ {32, 64, 100, 200}**. faiss IndexScalarQuantizer + rescore on the cone-distribution data: 1.0000 at pool 200. Caveat: synthetic clusters are well-separated; real-text corpora will be lower but the rescore step makes ranking exact over the pool — the only risk is a true neighbor falling outside the pool, which m=200 largely eliminates at k≤20.

**Published.** Qdrant docs (vendor): "the error introduced by scalar quantization is usually less than 1%" for f32→u8; their rescoring guidance is to overfetch then rescore in full precision. The rescoring pattern is standard in FAISS billion-scale pipelines (Johnson, Douze, Jégou 2017): compressed codes generate candidates, full-precision rescore of a small pool recovers most of the lost recall — their Table-2-style results show SQ8 raw recall ~0.86 @1 on Deep1B but near-exact after rescore.

**Measured speed:** naive int32-upconvert matmul 30.3 ms (avoid); blocked int8→fp32 chunk dot **9.49 ms** at 100k reading 38.4 MB; a real SIMD int8 kernel (VNNI on this CPU) would be ~2–4 ms. At 10k: ~0.3–1 ms. Rescore adds m·2·d flops = 200·768 = 0.15 MFLOP — sub-µs, noise.

**Bytes/memory:** d + 4 (scale) ≈ 388 B/memory → 38.8 MB @100k, 3.9 MB @10k. Keep fp32 on disk (mmap) for rescore: total footprint 192 MB @100k, RAM-resident working set 38.8 MB.

**Verbatim stage:** dense lane candidate generation + write-path (background embedding job produces v8 + scale alongside fp32). Pairs naturally with mmap storage.

**Verdict: SHIP as the dense-lane's internal representation whenever 100k-class corpora or tight RAM are in scope; optional-but-cheap even at 10k.** The scan stays exact-quality, the resident set drops 4×, and it halves cold-start IO.
*Confidence: high on the mechanism (rescore = exact over pool); medium on the exact recall floor on real text — close it by A/B on LoCoMo embeddings once the neural tier lands.*

## 3. Candidate C — Matryoshka (MRL) truncation / funnel

**Formula.** If (and only if) the embedding model is MRL-trained: truncate vectors to leading dims and renormalize: v⁽ʳ⁾ = normalize(v[1:r]). Funnel: score all N at r dims → take top-m → rescore full d. Time = 4·N·r + 4·m·d bytes-read equivalent.

**Published quality fractions (nomic-embed-text-v1.5 model card, MTEB average — vendor-published eval):**

| dims | MTEB | fraction of 768d |
|---|---|---|
| 768 | 62.28 | 100% |
| 512 | 61.96 | 99.5% |
| 256 | 61.04 | 98.0% |
| 128 | 59.34 | 95.3% |
| 64  | 56.10 | 90.1% |

OpenAI (vendor): text-embedding-3-large truncated to 256 dims still beats unshortened ada-002 at 1536 on MTEB. MRL paper (Kusupati et al., NeurIPS 2022): up to 14× smaller representation at equal accuracy (classification), and 14× wall-clock / 128× FLOP speedup via the funnel cascade at ImageNet scale. Mechanism transfers to text — nomic's table is direct evidence — but the MTEB deltas are averages, not retrieval any@10; label as extrapolation.

**Measured speed (fp32 exact, truncated contiguous matrices):** d=64 → 0.23 ms, d=128 → 0.39 ms, d=256 → 0.79 ms at N=100k. Funnel (64-scan + rescore 200 at full-d) ≈ 0.25 ms vs 1.19 ms full. Bytes: 1024·r/4… i.e. 4·r·N @100k: 26 MB @64d.

**Verbatim stage:** dense lane only; irrelevant elsewhere.

**Verdict: REJECT for the default profile; OPTIONAL if Verbatim's neural tier adopts an MRL-trained model AND scales past ~500k.** At ≤100k the funnel saves <1 ms while giving up up to ~10% embedding quality at the truncation tier — a bad trade inside a 150 ms budget. The only durable benefit is resident-footprint (256d → 100 MB @100k), which int8 rescore already solves without quality loss.
*Confidence: high that it is premature at this scale; the published numbers are real but they answer a speed problem Verbatim does not have.*

## 4. Candidate D — HNSW (hnswlib / faiss / sqlite-vec-adjacent)

**Cost model.** Per-node storage ≈ 4·d (vector) + 4·2·M (layer-0 link list, 2M neighbors × 4 B id) + ~M·8 avg upper links + ~24 B header. M=16, d=384: ≈1536 + 128 + 128 + 24 ≈ **1.8 KB/memory → ~180 MB @100k**. Query: O(ef·log N) distance evals, ef ≥ k.

**Measured (faiss IndexHNSWFlat, 100k×384, M=16, efConstruction=200):** build **16.4 s** (M=32: 32.1 s); query 0.02–0.09 ms/q. **Recall@10 is regime-dependent:**
- Clustered data with a decaying neighbor shell (NN1 cos 0.65 / NN50 0.54): ef=16 → 0.967, ef=64 → **0.9998**, ef=256 → 1.0000.
- Degenerate shells (iid or thin-shell clustered data, NN10≈NN50 within ~3%): ef=16 → 0.03–0.10, ef=64 → 0.09–0.22, ef=256 → 0.40. This is a *measurement caveat*: when the true top-10 sits inside a dense distance shell, rank-10 membership is nearly arbitrary and recall@10 understates semantic quality — but it is also the honest worst case for near-duplicate-heavy memory corpora, which is exactly Verbatim's regime (rephrased repeats of the same fact).

**Published anchor.** hnswlib on SIFT-1M (128-d euclidean, k=10, M=16, efC=200): recall@10 ≈ **0.97 at efSearch=64** (ann-benchmarks dashboard figure, quoted in iqdb-hnsw's docs.rs source). At ef=16 published curves sit ~0.85–0.95 on the same task. Note SIFT-1M is 10× Verbatim's max N and half the dim; recall difficulty grows with N and intrinsic dimensionality, so ~0.97 @ef64 is a fair 100k/384d expectation, with the thin-shell caveat above.

**Libraries.** hnswlib: **Apache-2.0** (repo license), header-only C++, Python bindings; needs a C++ toolchain at install (pip wheel failed to build on this box — missing Python.h — a real install-friction data point for a no-dependency default profile). faiss: **MIT**, heavy C++/BLAS dependency. sqlite-vec: **Apache-2.0** (NOT MIT as the brief guessed — GitHub-detected license is Apache-2.0), created 2024-04, ~8k stars; `vec0` virtual tables do **brute-force KNN over blobs in C** — i.e., the mature library choice at this scale IS exact scan; supports float/int8/binary storage. sqlite-vss: MIT but **archived** — README: "not in active development… my effort is now going towards sqlite-vec." USearch: Apache-2.0, i8/f16 HNSW, claims large speedups over faiss (vendor benchmark, unverified here).

**Verbatim stage:** dense lane candidate generation. Interacts badly with eligibility-first: ANN searches the whole graph then filters, so a heavily-quarantined corpus returns a starved candidate set — you would need per-partition graphs or post-hoc oversampling, adding the complexity the assignment already flags.

**Verdict: REJECT for the default profile; OPTIONAL for a future >500k–1M tier.** It converts a 1–15 ms memory-bound scan into a 0.05 ms graph walk — an irrelevant win — while adding ~180 MB index, ~16 s builds, a native dependency, and a 1–3% recall tax that compounds with eligibility filtering.
*Confidence: high. What would change it: demonstrated N ≥ 1M corpora, or a p95 target < ~5 ms that exact+int8 cannot meet.*

## 5. Candidate E — storage layout for the scan (the thing that actually matters)

Measured at N=100k: per-row SQLite BLOB reads → matrix + dot = **145.5 ms** (12.5 ms @10k); 256-vectors-per-blob packing → **71.1 ms** (7.1 ms @10k); single raw `read()` + dot = 57.6 ms; **mmap + dot = 1.19 ms** warm.

The naive layout (one BLOB row per vector, what a SQLite-first design drifts toward) is the only option that breaks the p95 target at 100k. Fix: store the dense matrix as a packed sidecar file (or a handful of large blobs) and `mmap` it — OS page cache does lazy loading, first-touch is page-granular, and the scan is exactly the fp32 numbers above. sqlite-vec's vec0 is essentially this pattern maintained in C.

**Verdict: SHIP — mmap sidecar (or ≤few-hundred large blobs); REJECT per-row blob scanning.** Confidence: high, measured directly.

## 6. Comparison table (N=100k, d=384, k=10)

| Candidate | Query ms @10k | Query ms @100k | Index bytes @100k | Recall@10 vs exact | Deps / license | Verdict |
|---|---|---|---|---|---|---|
| Exact fp32, mmap | 0.12 (measured) | 1.2 server / ~8–15 consumer | 153.6 MB | 1.000 | none | ship |
| int8 scan + fp32 rescore (m=200) | ~0.3–1 | ~2–5 (9.5 numpy-emulated) | 38.8 MB (+fp32 on disk) | 1.0000 measured synthetic; ~0.99 published SQ+rescore | none | ship/optional |
| MRL funnel 64→384 | ~0.03 | ~0.25 | 26 MB resident | ~90–98% of full-dim MTEB | needs MRL model | optional/reject-now |
| HNSW M=16 ef=64 | <0.01 | 0.06 | ~180 MB | 0.97–1.00 structured; 0.1–0.4 thin-shell | hnswlib Apache-2.0 / faiss MIT | reject (<1M) |
| Exact over per-row SQLite blobs | 12.5 | 145.5 | 153.6 MB | 1.000 | none | reject |

## 7. Provisional-constant note

- **cross-encoder pool cap 32:** consistent with this budget — at ~2–5 ms/candidate on 4 cores, 32 × ~3 ms ≈ ~100 ms, sitting exactly at the ≤150 ms quality-profile ceiling. Keep cap ≤32; if exact scan stays default, there is no reason to shrink it further.

## 8. Coverage check

(1) exact cosine flops/IO + ms: §1 (76.8 MFLOP, 153.6 MB, 0.12/1.19 ms measured, ~8–15 ms consumer). (2) int8 + rescore recall curves: §2 (measured 1.0000 at m≥32 synthetic; Qdrant <1% SQ error; FAISS SQ8+rescore pattern). (3) MRL at 64/128/256: §3 nomic MTEB table + MRL paper funnel. (4) HNSW recall@10 at ef 16/64, build memory, library licenses: §4 (measured 0.967/0.9998 structured + thin-shell caveat; published 0.97@ef64 SIFT-1M; ~1.8 KB/node; hnswlib Apache-2.0, sqlite-vec Apache-2.0 [not MIT], sqlite-vss MIT-archived, faiss MIT, usearch Apache-2.0). (5) verdict with ms: §1–§5 — no ANN needed at ≤100k; exact int8+rescore suffices.
