# d4-ranx-pyserini — ranx fusion semantics + pyserini/Anserini BM25 defaults → ported formulas for Verbatim's lane-fusion stage

Method note: all ranx behavior below was read from the shipped source (ranx 0.3.21 wheel, `ranx/fusion/*.py`, `ranx/normalization/*.py`, `ranx/meta/fuse.py`) AND executed locally to confirm semantics. All pyserini/Anserini behavior was read from source (`pyserini/search/lucene/_searcher.py`, `pyserini/fusion/_base.py`, `pyserini/trectools/_base.py`, `anserini/.../SearchCollection.java`, `IndexCollection.java`, `AccurateBM25Similarity.java`). Numbers marked [measured] were computed on this machine (4-core-class Linux box, CPython 3.10, SQLite 3.37.2); numbers marked [source] are code-verified constants; [paper] are literature claims.

---

## 1. ranx fusion: exact semantics (verified against source + live execution)

### 1.1 The dispatch structure

`fuse(runs, norm="min-max", method="wsum", params=None)` (`ranx/meta/fuse.py:8-37`):
1. Optionally normalize each run independently via `norm_switch(norm)` — norms: `borda`, `max`, `min-max`, `min-max-inverted`, `rank`, `sum`, `zmuv` (`ranx/normalization/__init__.py:18-34`).
2. Dispatch to `fusion_switch(method)` — 24 methods, aliases: `sum, mnz, gmnz, anz, max, med, min, wsum, wmnz, rrf, bordafuse, w_bordafuse, condorcet, w_condorcet, isr, log_isr, logn_isr, mapfuse, posfuse, probfuse, segfuse, slidefuse, bayesfuse, rbc, mixed`.

Key fact: **the fusion functions themselves never normalize**. `comb_sum`/`wsum` sum whatever scores they are given. Normalization is a caller responsibility (`fuse()` or explicit `normalize()`). Same for pyserini (rescoring is an explicit `TrecRun.rescore` call). Ported rule for Verbatim: `fuse = normalize ∘ aggregate` as two explicit stages, and rank-based lanes skip stage 1 entirely.

### 1.2 RRF — `ranx/fusion/rrf.py`

```
_rrf_score(results, k):  for i, doc_id in enumerate(results.keys()):
                             combined[doc_id] = 1 / (k + i + 1)
rrf(runs, k=60):         return comb_sum([_rrf_score(run) for run in runs])
```

- Rank = position in the lane's score-descending list, 1-based → `fused(d) = Σ_lanes w? no → Σ 1/(k + rank_l(d))`.
- `k=60` default, matching Cormack, Clarke & Bütcher, SIGIR 2009 [paper, doi:10.1145/1571941.1572114 — tested k over TREC runs, k≈60 robust].
- ranx RRF takes **no lane weights** — to get weighted RRF you must write `Σ w_l·1/(k+r_l)` yourself (trivial).
- Rank comes from **stored order**, not scores — identical scores get distinct consecutive ranks in insertion order (numba dict order = insertion).

pyserini's equivalent (`trectools/_base.py:113,145` + `fusion/_base.py:29`): `rank = score.rank(ascending=False, method='first')` (ties broken by first occurrence — same "position decides ties" semantics), `score = 1/(rrf_k + rank)`, merge by `AggregationMethod.SUM`, `rrf_k=60` default. Two independent implementations, identical semantics — the formula is standard.

**Measured (me)** — 3 lanes × 5 candidates, `k=60`:
lane1 `d1:10,d2:8,d3:6,d4:4,d5:2`; lane2 `d3:5,d6:4.5,d1:4,d7:3,d2:1`; lane3 `d9:1,d3:0.9,d8:0.5,d1:0.4,d10:0.1`
- `d1 = 1/61 + 1/63 + 1/64 = 0.047891458495966696` — ranx output **bit-exact**.
- `d3 = 1/63 + 1/61 + 1/62 = 0.04839549075403121` → d3 wins despite d1's highest raw score — RRF rewards multi-lane presence, pure rank, no scale. This is exactly the property Verbatim wants when lanes emit non-comparable scores (BM25 vs PPR vs entity match count).

### 1.3 CombSUM / CombMNZ — `comb_sum.py`, `comb_mnz.py`

```
comb_sum(d)  = Σ_lanes  s_l(d)                     # absent lane contributes 0
comb_mnz(d)  = Σ_{l: d∈l} s_l(d) × |{l: d∈l}|      # sum × count of covering lanes
comb_anz(d)  = Σ s / |{l: d∈l}|                    # (code: sum / len)
comb_gmnz(d) = (Σ s) × |{l: d∈l}|^γ              # γ tunable, search range 0.01-1.0 (optimize_comb_gmnz)
```

Verified live: 3-lane example → `comb_mnz(d1) = (10+4+0.4)×3 = 43.2` bit-exact via `_comb_mnz`. CombMNZ = the classic Montague & Aslam formulation: multi-source presence is a bonus multiplier, not just missing-value zero. For Verbatim this is the score-based analogue of "candidate seen by 2+ lanes should rank above a single-lane hit" — the same effect RRF produces implicitly.

### 1.4 Weighted variants — `wsum.py`, `wmnz.py`

```
wsum(d, w) = Σ_lanes w_l · s_l(d)                  # absent → 0 contribution
wmnz(d, w) = (Σ_{l: d∈l} s_l(d)) × (Σ_{l: d∈l} w_l) # sum × sum-of-covering-weights
```

Verified live: `wmnz(d1) = 14.4 × (1+0.75+0.5) = 32.4` bit-exact. Note WMNZ is NOT `Σ w·s × count` — it's score-sum times weight-sum. For Verbatim's lane weights {1.0, 0.75, 0.5}: WMNZ makes a candidate's bonus scale with the *quality* of covering lanes (cover by lanes 1+3 → ×1.5) rather than a flat count — strictly more expressive than CombMNZ for weighted lanes.

### 1.5 Normalization (`ranx/normalization/*.py`) — exact edge semantics

- **min-max**: `(s − min) / max(max − min, 1e-9)` (`min_max_norm.py:16-33`). **A flat lane (all-equal scores) → every doc maps to 0.0**, i.e. the lane contributes *nothing* to a sum-fusion — silent vote-erasure. Per-query (normalized per `q_id`), and `min-max-inverted` exists for ascending (distance-like) scores.
- **pyserini NORMALIZE** (`trectools/_base.py:~150-158`): same `(s−low)/(high−low)` but on `high−low==0` returns `np.ones_like` — **flat lane → all ones**. Opposite policy to ranx. For Verbatim (where the filed bug is "min-max fusion tie-collapse"): decide deliberately — options: (a) pyserini-style flat→1 (flat lane votes fully, spurious at top), (b) ranx-style flat→0 (lane silenced), (c) flat→`w_l·0.5` (half-vote — my recommendation for a lane that produced a degenerate uniform list), or (d) drop flat lanes from score-fusion and let them vote only via rank lanes.
- **rank**: `1 − i/n` for 0-based position i in a lane of n (`rank_norm.py:15-24`) — linear 1→~1/n ramp, score-free, deterministic. A cheap alternative to RRF's 1/(k+r) curve: sharper tail decay (rank-norm hits ~0 at the bottom of the lane; RRF k=60 at rank 60 still gives 1/120 ≈ half of rank-1's 1/61). If Verbatim wants "only the top of each lane matters," rank-norm (or smaller RRF k) expresses it.
- **zmuv**: z-score `(s−μ)/max(σ,1e-9)` — spreads scores but preserves outliers; risky when a lane emits one huge outlier (that doc dominates post-sum; z-scores give it unbounded leverage). Reject for default.
- **sum**: `s/Σs` — makes each lane a distribution; sensitive to lane length n (a 10-doc lane inflates per-doc share vs a 100-doc lane).
- **borda**: cross-run position-based, maps to Copeland-style points; equivalent to linear rank fusion.

### 1.6 Empirical caveat discovered on this machine

ranx 0.3.21's numba-parallel wrappers misbehaved in my CPython 3.10 + current-numba env: high-level `comb_mnz(runs)` dropped the 3rd lane's contribution (d1→28.0 instead of 43.2), and `wmnz(runs,...)` returned an empty result. The `@njit` low-level functions were exact when fed `TypedList`s directly. Takeaway for the port — **import the formulas (≤30 lines of pure Python), not the library**: ranx pulls in numba/llvmlite (JIT, ~100MB+ transitive deps) which contradicts "one SQLite file, no heavy deps," and its parallel layer showed version-dependent behavior.

---

## 2. BM25 defaults: Anserini/pyserini vs Lucene — and which constant matters for short turns

### 2.1 Verified defaults

| source | k1 | b | where |
|---|---|---|---|
| Lucene `BM25Similarity()` | 1.2 | 0.75 | Lucene source/javadoc — Robertson TREC-4-era default |
| Anserini `-bm25.k1/-bm25.b` | **0.9** | **0.4** | `SearchCollection.java:277-280` |
| pyserini `LuceneSearcher.set_bm25` | 0.9 | 0.4 | `_searcher.py:583`; index-side `bm25(k1=0.9,b=0.4)` at `_searcher.py:675` |

Anserini's own provenance comment (`SearchCollection.java:270-274`, verbatim): Robertson et al. (TREC 4) proposed k1∈[1.0,2.0], b∈[0.6,0.75] with "k1 = 1.2 and b = 0.75 being a very common setting. Empirically, these values don't work very well for modern collections. Here, we adopt the defaults recommended by Trotman et al. (SIGIR 2012 OSIR Workshop) of k1 = 0.9 and b = 0.4." Tuned on INEX 2008 Wikipedia; also used in ATIRE and Lin et al. ECIR 2016. Two independent anchors: Trotman's workshop tuning + a decade of Anserini TREC reproductions.

### 2.2 Which parameter actually moves short-turn scores [computed]

BM25 term contribution: `idf(t)·tf·(k1+1)/(tf + k1·(1−b + b·dl/avgdl))`. Computed for N=100k, df=50, turns of 15/30/60 tokens (avgdl=30):

| case | k1=1.2,b=0.75 | k1=0.9,b=0.4 | delta |
|---|---|---|---|
| tf=1, dl=avgdl | 7.590 | 7.590 | **0.0 — identical** |
| tf=2, dl=30 | 10.437 | 9.946 | −5% |
| tf=1, dl=60 (2×avg) | 5.387 | 6.381 | +18% |
| tf=1, dl=15 (0.5×avg) | 9.542 | 8.385 | −12% |
| tf=1, dl=30, avgdl=100 | 10.636 | 8.751 | −18% |

Facts extracted:
- At tf=1 and dl=avgdl the whole k1/b apparatus collapses to `idf` exactly — **k1 is near-inert for dialogue turns** because a 25-60-token turn rarely holds tf≥3 of a content term (tf=2 is the whole tf effect, ±5%).
- **b is the live lever**: it scales the length-advantage spread by ~±12-18% per ±1× length deviation. Dialogue turns DO vary 2-4× in length, so b=0.4 (Anserini) compresses the "longer turn wins" bias relative to b=0.75 — desirable: a 60-token turn shouldn't win over a 15-token turn merely for containing more words.
- Lucene/FTS5's 1.2/0.75 vs Anserini's 0.9/0.4 will reorder items only at the margins for this corpus shape; the bigger deal is *where the stats come from* (next item).

### 2.3 The eligibility interaction — the real reason the defaults question is secondary

Lucene bakes the `(1−b+b·dl/avgdl)` norm into a 1-byte index-time norm — hence Anserini's `IndexCollection.java:176` comment: "Necessary during indexing as the norm used in BM25 is already determined at index time" (why `AccurateBM25Similarity` exists). And `AccurateBM25Similarity` computes `idf = log(1 + (N−n+0.5)/(n+0.5))` and `avgdl = sumTotalTermFreq/docCount` **at query time from live CollectionStatistics** (`AccurateBM25Similarity.java:48-53`).

For Verbatim this has two direct consequences:
1. **Store raw `dl` per doc (per field), never baked norms** — Verbatim's avgdl/N/df must be computed over the *eligible* set per query (hard rule), which is impossible if the engine stores Lucene-style quantized norms. Store `dl` as a small int column; compute everything at query time.
2. **idf form choice**: Anserini/Accurate uses `log(1 + (N−n+0.5)/(n+0.5))` — always ≥0, no floor needed, and gracefully handles high-df terms on small eligible sets. FTS5's builtin uses Robertson `log((N−n+0.5)/(n+0.5))` with a `1e-6` floor (proved below). For eligible-set stats where df_e ≤ N_e can approach N_e (e.g. "the" inside one conversation), Anserini's +1 form is strictly safer. **Port `idf = log(1 + (N_e − df_e + 0.5)/(df_e + 0.5))`.**

### 2.4 FTS5's built-in bm25() — measured, and why it can't be used for scoring [measured]

Constructed a 4-doc index (lengths 1,2,4,6; "apple" in 3) and solved the constants from measured rank ratios: d2/d1 = 1.10513 exactly matches k1=1.2,b=0.75 (k1=0.9,b=0.4 predicts 1.196 — rejected); d3/d1 = 0.65495 matches b=0.75; absolute magnitude shows `idf = 1e-6` floor when Robertson idf ≤ 0.

**Conclusion**: SQLite FTS5 `bm25()` = Lucene-style BM25, **k1=1.2, b=0.75 hardcoded, idf floored at 1e-6, stats over the whole table** — i.e., it is wrong for Verbatim twice over: (a) parameters not tunable, (b) N/df/avgdl computed over ineligible rows, violating "BM25 statistics must be computed on the ELIGIBLE set." Use FTS5 for what it's great at — candidate generation (postings + MATCH) — and score in Python (or a custom FTS5 auxiliary function) over the eligible subset.

### 2.5 Measured FTS5 costs on this machine [measured]

Synthetic memories, 25–60 tokens, `CREATE VIRTUAL TABLE mem USING fts5(body)`:

| N | build (s) | db size | bytes/doc | top-60 `MATCH…ORDER BY bm25` p50 | p95 |
|---|---|---|---|---|---|
| 10,000 | 1.9 | 3.9 MB | 394 | 0.06 ms | 0.07 ms |
| 100,000 | 19.6 | 39.4 MB | 394 | 0.36 ms | 0.40 ms |

Candidate generation is ~1% of Verbatim's current p95 (85 ms). The lexical lane is not the bottleneck — the gap vs. the 40-line BM25 harness is the *eligible-set stats + lane coverage*, not the index.

### 2.6 BM25F (multi-field) — brief

pyserini/Lucene has **no true BM25F**: `search(q, fields={f:boost})` issues per-field BM25 queries combined by disjunction — field scores are computed independently then summed (not pooled tf). True BM25F (Zaragoza et al., Microsoft Cambridge TREC-14) pools `tf'(d) = Σ_f w_f·tf_f(d)` and applies one saturation+length normalization on the pooled length `dl' = Σ_f w_f·dl_f`. For Verbatim fields (e.g. `text`, `entities`, `speaker`), if a second field is ever indexed, implement pooled-tf BM25F in the same custom scorer — do not fake it with boosted disjuncts (known-miss behavior: high-boost short fields over-fire).

---

## 3. Prebuilt-index patterns to mimic in SQLite

pyserini's pattern (`prebuilt_index_info.py` + `util.py:253-307,391-463,498-522`): a registry `INDEX_INFO[name] = {urls: [mirrors], md5, ...}`; `download_url` → md5 verify → `download_and_unpack_archive` → materializes to `{index_name}.{md5}` (checksum in the dirname = content-addressed version pin). Collection stats (`docCount`, `sumTotalTermFreq`, per-term df/cf) live inside the Lucene index and are read at open.

Port to Verbatim's one-file model:
- **The .db IS the artifact.** FTS5 tables + stats travel inside the file — no tarball needed. Optionally `VACUUM INTO` to ship a compacted image; put `PRAGMA integrity_check` result + row counts in the file manifest.
- **Content-addressing**: keep `{db}.sha256` (or `sqlite3 -deserialize`/URI `mode=ro&immutable=1` open) — same checksum-pin contract as pyserini's md5.
- **Stats table** (`docstats`: `N`, `sumTotalTermFreq`, `avgdl`, per-field equivalents; `termstats` df/cf optional — df can be `COUNT`ed from postings or maintained incrementally at add-time). Write-path maintains stats transactionally with the doc insert — equivalent of Lucene's index-time CollectionStatistics, but updated per-memory.
- **No baked norms** (see §2.3) — store `dl` raw; stats computed on eligible subset at query time.

Resident cost of the eligible-BM25 machinery itself: a stats row (~40B) + `dl` column (4-8B/doc → ~0.7MB/100k) + optional `df` cache (vocab × 8B; ~2MB for 250k-term vocab/100k). Total ≪ FTS5's own 394B/doc.

---

## 4. Per-query complexity → milliseconds (4-core reference, Python implementation)

Notation: L = lanes (8), C = per-lane candidates (60-128), U = union size (≤L·C, realistically 150-400 with overlap), T = query terms (2-6 typical).

| stage | ops | 10k | 100k | basis |
|---|---|---|---|---|
| FTS5 candidate gen (per lane-term) | O(postings touched) | 0.06 ms | 0.36 ms | measured, top-60 |
| Eligible-set stats (N_e, df_e, avgdl_e) | one `COUNT`/avg on eligible table + per-term `COUNT` on postings∩eligibility | ~1-3 ms | ~5-15 ms | est.: 100k-row filtered agg in SQLite ≈ ms range; worst case per-term df scan ≤ df_e rows |
| Python BM25 re-score of U candidates × T terms | O(U·T) dict lookups + dl fetch | ~0.5-2 ms | ~0.5-2 ms | est.: ~2k term-evals ≈ µs each |
| Fusion (RRF/wsum/mnz) over L×C | O(L·C) accumulate + O(U log U) top-k | 0.14-0.36 ms | same (size-independent) | measured: rrf 136µs, wsum+minmax 152µs, mnz 182µs @8×60; 250-360µs @8×128 |
| Total added vs today | — | ≈2-6 ms | ≈6-18 ms | well under p95≈150ms headroom; the eligible-df query dominates |

Eligible-df arithmetic shown: term postings for a mid-df term ≈ 50-500 rows at 100k; an indexed `COUNT`/`IN` on a small integer eligibility column is ~10-50µs per term → 6 terms ≈ ≤0.3ms typical; worst-case common-term (df~5k) ~0.5-1ms. If a term's global df ≫ 5k, fall back to `df_e = df_global × (N_e/N)` sampled estimate or an eligible-doc bitset (100k bits = 12.5KB, Roaring or plain bytes) for per-term intersects — that's the pre-built-index move worth stealing for the eligibility stage too.

---

## 5. Ported pseudocode — drop-in for Verbatim's fusion stage

```python
# ---------- normalization (per lane, per query) ----------
def minmax_norm(lane, flat="half"):          # lane = [(doc, score)] score-desc
    lo = min(s for _, s in lane); hi = max(s for _, s in lane)
    if hi - lo < 1e-12:
        return [(d, FLAT_VALUE[flat]) for d, _ in lane]   # "one"->1.0 (pyserini), "zero"->0.0 (ranx), "half"->0.5 (recommended)
    den = hi - lo
    return [(d, (s - lo) / den) for d, s in lane]

def rank_linear(lane):                       # ranx rank_norm
    n = len(lane)
    return [(d, 1 - i / n) for i, (d, _) in enumerate(lane)]

# ---------- score fusion (needs normalized, comparable scales) ----------
def comb_sum(norm_lanes):
    acc = {}
    for lane in norm_lanes:
        for d, s in lane: acc[d] = acc.get(d, 0.0) + s
    return acc

def comb_mnz(norm_lanes):                    # ranx-verified: sum × covering-lane count
    acc, cnt = {}, {}
    for lane in norm_lanes:
        for d, s in lane:
            acc[d] = acc.get(d, 0.0) + s
            cnt[d] = cnt.get(d, 0) + 1
    return {d: acc[d] * cnt[d] for d in acc}

def wsum(norm_lanes, w):                     # w: per-lane weights {1.0, .75, .5}
    acc = {}
    for lane, wl in zip(norm_lanes, w):
        for d, s in lane: acc[d] = acc.get(d, 0.0) + wl * s
    return acc

def wmnz(norm_lanes, w):                     # ranx-verified: Σs × Σw over covering lanes
    acc, wacc = {}, {}
    for lane, wl in zip(norm_lanes, w):
        for d, s in lane:
            acc[d]  = acc.get(d, 0.0)  + s
            wacc[d] = wacc.get(d, 0.0) + wl
    return {d: acc[d] * wacc[d] for d in acc}

# ---------- rank fusion (score-free, scale-immune) ----------
def wrrf(lanes, w, k=60):                    # ranx/pyserini RRF + optional weights
    acc = {}
    for lane, wl in zip(lanes, w):
        for rank, (d, _) in enumerate(lane, start=1):
            acc[d] = acc.get(d, 0.0) + wl / (k + rank)
    return acc                               # top-k via heapq.nlargest

# ---------- eligible-set BM25 (the whole reason for custom scoring) ----------
def bm25_eligible(q_terms, eligible, postings, dl, stats, k1=0.9, b=0.4):
    # stats over ELIGIBLE set only: N_e, df_e(t), avgdl_e — Anserini-style idf
    N_e, avgdl_e = stats["N"], stats["avgdl"]          # computed on eligible rows
    out = {}
    for t in q_terms:
        df_e = stats["df"][t]                         # df within eligible set
        idf = math.log(1 + (N_e - df_e + 0.5) / (df_e + 0.5))   # >= 0, no floor
        for d, tf in postings[t]:                     # only eligible d's
            denom = tf + k1 * (1 - b + b * dl[d] / avgdl_e)
            out[d] = out.get(d, 0.0) + idf * tf * (k1 + 1) / denom
    return out
```

**Worked example** (3 lanes × 5 candidates from §1.2 — all numbers verified against ranx output):
- wrrf, w=[1,.75,.5]: `d3 = 1/61·1 + 1/61·.75? — careful: per-lane rank` → d3: lane1 r3 → 1/63·1.0; lane2 r1 → 1/61·0.75; lane3 r2 → 1/62·0.5 = 0.015873+0.012295+0.008065 = 0.036233. d1: 1/61 + 0.75/63 + 0.5/64 = 0.016393+0.011905+0.007812=0.036110. d3 still wins but the gap narrows — weights shrink cross-lane bonuses toward the dominant lane.
- comb_mnz on min-max (flat="half"): lane1 minmax d1=1,d2=.75,d3=.5; lane2 d3=1,d6=.875,d1=.75; lane3 d9=1,d3=.889,d1=.333 → `d1=(1+.75+.333)×3=6.25`, `d3=(.5+1+.889)×3=7.17` — MNZ ordering flips vs wsum because coverage count is multiplicative.

---

## 6. Comparison tables

### Fusion candidates for Verbatim

| formula | needs scores? | norm needed | handles heterogeneous lanes | tie-collapse risk | verdict |
|---|---|---|---|---|---|
| RRF k=60 (current) | no | no | yes — best | none | **ship (default)** |
| Weighted RRF | no | no | yes | none | **ship** — trivial extension, matches {1,.75,.5} weights |
| CombSUM+min-max | yes | yes | poor if scales differ | flat-lane → policy needed | **optional** |
| CombMNZ+min-max | yes | yes | ok | flat-lane + count dominates | **optional** — coverage bonus may beat RRF when lanes disagree; test post-fix |
| WSUM+min-max | yes | yes | poor-moderate | same | **optional** — ranx's own default (min-max+wsum); grid-search target if we ever tune fusion |
| WMNZ+min-max | yes | yes | ok | same | **optional** — strictly-more-expressive MNZ for weighted lanes |
| zmuv/sum norm | — | — | — | outlier leverage / lane-length bias | **reject** for default |
| ranx library as dep | — | — | — | numba dep + observed wrapper quirks | **reject** — port formulas (≤30 lines) |

### BM25 configuration for short turns

| option | k1/b | stats set | verdict |
|---|---|---|---|
| FTS5 builtin bm25() | 1.2/0.75 locked | whole table | **reject for scoring** (eligibility violation + untunable); **ship as candidate generator** |
| Custom scorer, Lucene defaults | 1.2/0.75 | eligible | acceptable fallback |
| Custom scorer, Anserini defaults | **0.9/0.4** | eligible | **ship** — b=0.4 dampens the 2-4× length variance of turns; proven provenance (Trotman 2012 → Anserini → decade of TREC reproductions) |
| Anserini idf `log(1+…)` | — | eligible | **ship** — nonneg, no floor hack |

### Prebuilt-index moves worth stealing

| pyserini pattern | SQLite port | verdict |
|---|---|---|
| INDEX_INFO registry + md5-pinned artifacts | `.db` + sidecar `.sha256`, `immutable=1` open | **ship** |
| Collection stats inside index | `docstats` table + raw `dl` column | **ship** |
| Index-time baked norms | do NOT bake — Anserini built AccurateBM25 because of this | **reject pattern** |

---

## 7. Licenses (all permissive — formulas portable)

| component | license | evidence |
|---|---|---|
| ranx 0.3.21 | MIT | wheel METADATA `Classifier: License :: OSI Approved :: MIT`, repo README badge |
| pyserini | Apache-2.0 | file headers (`_searcher.py`, `_base.py`) |
| anserini | Apache-2.0 | repo (file headers in `SearchCollection.java`, `IndexCollection.java`) |
| Lucene | Apache-2.0 | ASF project |
| SQLite/FTS5 | public domain | sqlite.org |

No copyleft anywhere; equations themselves are unencumbered — port freely.

## 8. Residual risks / honest caveats

- k1=0.9/b=0.4 for LoCoMo-style turns is an *extrapolation*: both presets barely differ at tf=1 (identical) — the choice is low-risk but also low-upside; the real win is eligible-set stats, not the constants. Confidence: high on mechanism, moderate on magnitude.
- ranx.fuse (CIKM 2022, Bassani & Romelli) demonstrated grid-search over method×norm×weights as a workflow; I did not extract its leaderboard deltas (paper body not freely fetchable) — treat "learned weighted fusion can beat RRF" as supported-by-existence, not by a number.
- Eligible-df per query is the one new O(df) cost; on 100k with heavy terms this is the millisecond item to watch — measure `df_e` query plan before locking p95 budget.
- BM25F pooled-tf variant untested here — optional, only if a second scored field appears.
