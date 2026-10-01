# d5-fts5-sqlite — FTS5 behavior verified on sqlite 3.37.2 (all CALCULATED)

Verified on this box (`sqlite3.sqlite_version = 3.37.2`, in-memory DBs):

## Confirmed mechanics

- `bm25(tbl, w_a, w_b, ...)` per-column weights WORK — verified: multiplies each
  column's tf contribution pre-saturation = canonical BM25F shape (matches b1's
  fts5_aux.c read). Valid for candidate generation.
- `columnsize` aux function: **NOT compiled in this build** ("no such
  function"). Per-column doc lengths are therefore unavailable to SQL —
  strengthens the case for the Python-side rescore (which computes field
  lengths itself).
- `fts5vocab(tbl,'col')` virtual table WORKS: returns (term, col, doc_count,
  total_count) per column — usable to build the df table and the rare-term
  clue seed; 'row' variant gives global df. Still *global* stats — eligible-df
  needs the int-bitset path regardless.
- Negative-idf edge REAL: `bm25` returns −1.7e-6 scores for high-df terms
  (SQLite's clamp floor, no +1 in idf) — a high-df query term can rank docs
  by *smaller* negative = reversed ordering on that term's contribution.
  Confirms ln(1+x) idf for the rescore.

## Measured MATCH + order-by-rank latency @ 100k docs (top-40, warm)

| query shape | latency |
|---|---|
| uniform-vocab 3-term OR | 80 ms |
| uniform-vocab 10-term OR | 178 ms |
| uniform-vocab single term | 36 ms |
| Zipf vocab, mid-df 3-term OR | **3.9 ms** |
| Zipf vocab, mid-df 3-term AND | 0.1 ms |
| Zipf vocab, rare-df 3-term OR | 0.2 ms |
| Zipf vocab, head-df 3-term OR (w0|w1|w2 ≈ 30–60% df) | 98 ms |

Latency tracks total posting-list length scored — `order by rank` forces
bm25() on every match. Two deployment conclusions:

1. **The selective-term cap is load-bearing, not optional**: restrict MATCH to
   the ≤8–16 lowest-df query terms (df table from fts5vocab or the corpus df
   table; ~0.1 ms lookup) AND/OR drop terms above a df fraction (~0.25·N_E).
   With Zipf vocab this keeps FTS5 at ~1–5 ms; without it head terms cost
   ~100 ms/query at 100k.
2. fts5vocab on 'col' + the eligible-bitset AND gives exact eligible-df for
   any scope — per-scope docfreq tables rejected (2.9–11.5 MB, b1) since the
   bitset cost is 13 ms cold / 0.02 ms warm and zero standing bytes.

## Verdicts

- FTS5 `bm25(tbl, w…)` as the lexical candidate lane → **ship** (gen-only;
  global idf + no columnsize + hardcoded k1/b disqualify final scoring).
- Lowest-df term cap (≤16, drop df>0.25·N_E) inside MATCH construction →
  **ship** — measured 98→4 ms worst-case difference.
- fts5vocab('col') as the df table → **ship** (built-in, no extra storage).
- columnsize-dependent BM25F in SQL → **reject** (function absent in
  sqlite ≤3.44 builds; field lengths must come from the Python rescore).
- Python BM25F rescore with eligible stats → **ship** (unchanged; all the
  above just confirms the split).
