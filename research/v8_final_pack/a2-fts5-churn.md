# V8-07.04/07.06 — FTS5 shadow-statement churn: origin, prototypes, verdicts

Corpus: locomo10 dev store, `/tmp/v7w/mem.db` (stock) and `/tmp/v7proto/mem.db` (proto).
4143 indexed FTS rows, eligible set = `kind='turn'` units = 2877 rows.
All statement counts measured with `conn.set_trace_callback` inside real `search()` calls;
sample = 40 dev queries, `limit=10`, `consistency='session'`.
Parity baseline re-run with `--timeout-ms 30000` to remove deadline-truncation noise from snapshot comparisons.

## Task 1 — Origin attribution (trace per shadow shape)

`set_trace_callback` on the thread-local reader connection, one real `search()` per query,
stack-frames attribute each shadow statement to its caller.

| Shape | stmts/q (p50, base) | Caller | Trigger |
|---|---|---|---|
| `unit_fts_docsize` (`SELECT sz … WHERE id=?`) | ~3,418 (75%) | `graph.py:323 _derive_seeds` — `ORDER BY rank` on a ≤16-term OR-MATCH forces bm25() evaluation (→ per-row docsize read) on **every matched row**, not just the LIMIT-10 output | ~2,560 |
| same shape | (25%) | `temporal.py:359 _fts_scores` — `ORDER BY bm25(unit_fts)` over the OR-MATCH set | ~860 |
| `unit_fts_content` (`SELECT … FROM unit_fts_content WHERE fts_row_id=?`) | ~706 | `lexical.py:294 _fetch_fields` — per-candidate vtable column reads (~700 nominated docs/q × 1 stmt) | ~706 |
| `unit_fts_idx` (`SELECT pgno … segid=? AND term<=?`) | ~328 | `_collect_postings`/`_match_rowids` per-term MATCH probes + stem/tri lanes (`stem_idx` ~87+~3411 total-family, `tri_idx` ~202+~1386) | per-term × channel |
| `PRAGMA 'main'.data_version` | ~4–5/q | FTS5 vtable open freshness check, once per vtable per query | — |
| `tbl:units` probes | ~330 | `graph.py:406 unit_row` — one `SELECT … WHERE unit_id=? AND generation<=?` per seed/visited unit | per-unit |

Totals (base, 2000ms budget): mean total 5,291/q; p50 4,250; p95 9,904. Shadow fraction ≈ 93%.

## Task 2 — maintained `unit_doclen` arm

Prototype: `unit_doclen(unit_id, generation, scope_id, field, len)` WITHOUT ROWID, PK
`(scope_id, unit_id, field, generation)`; written inside the projection tx
(`units_jobs.py`, 5 fields/unit via `_tokenize`); backfilled into `/tmp/v7proto` for all
20,715 field entries. Read path `_unit_lens` = one batched `IN`-chunk read per candidate
page, generation-pinned via a batched `unit_fts_rows` generation lookup
(replaces `_field_lens`'s docsize decode).

- **Drift: 0/20,715** field entries differ from `docsize` `sz` (tokenize-vs-unicode61 identical on this corpus).
- **Scoring parity (fixed):** an early version read the FTS row generation from the wrong
  source (`by_rowid` recs carry no `generation`) → all-zero lens → +19–58% score inflation.
  Fixed by fetching generation from `unit_fts_rows` per rowid (1 batched stmt). After fix:
  **scores byte-identical 40/40**, **items byte-identical 40/40** (full arm, 30s timeout).
- **Statement delta (UDL-only arm):** p50 4,331 vs base 4,250 — **≈ neutral (+81)**. The
  `_field_lens` docsize read was already `IN`-batched, so the win is architectural, not
  statement-count: the rescore no longer touches any FTS5-internal shadow at all, which is
  what unblocks the docsize-killing arms. Cost: +1–2 plain-table reads per page
  (`gen_of` + doclen).
- **lex-lane p50:** stock 37.2ms → proto 45.0ms (+21%) — Python-side aggregation in
  `_unit_lens`/`_vocab_rowids` outweighs the SQL savings inside the lane; recovered at
  whole-query level by the lanes that lose their rank-sort churn.

## Task 3 — collapsed-MATCH arm (V8-07.06)

Per-term MATCH replaced by fts5vocab `'instance'` postings (`unit_fts_vocab`, plus
`_stem_vocab`/`_tri_vocab` created in proto schema) — single `term IN (…)` aggregate
`GROUP BY term,doc[,col]` per channel. (`col` is a column **name**, not index; tf is
aggregated across columns, matching FTS5's total-tf bm25.)

- **Nomination sets: byte-identical 40/40** (posts dict and nominated list both).
- Multi-token/phrase-shaped terms and identifier-channel terms fall back to the original
  MATCH (vocab can't answer phrase rows — gated by `_U61_RE.findall(fold(term))==1`).
- Statement drop for the postings segment alone: modest (p50 4,315 vs 4,250 — the per-term
  MATCHes were ~24/q outer statements; their real cost was the idx probe shape, absorbed
  into the vocab scan). The dominant win comes from the two rank-sort sites below.

## Task 4 — FTS5-native bm25() arm

Formula empirically recovered and verified **bit-exact** on this store:
`score(doc)=Σ_terms idf_t·tf_t·(k1+1)/(tf_t+k1·(1−b+b·DL_total/AVG_total))`,
k1=1.2, b=0.75, tf summed over all columns, `DL_total=Σsz` over 5 columns,
`AVG_total=Σ averages-record sums / nRow` (`unit_fts_data` id=1 = `[nRow, Σsz per col]`),
`idf=ln((N−df+0.5)/(df+0.5))`, non-positive idf ⇒ zero contribution.
Accumulation order must be **per-doc in MATCH-term order** to be bit-identical (alphabetical
GROUP BY order drifts ~1e-15 and flips boundary ties — fixed and re-verified: derived seed
lists 52/52 identical).

Dev-gold accuracy (40 tasks, gold rowid via ref2unit; recall@10 over the arm's top-10
restricted to the nominated set):

| Arm | recall@10 | mean gold rank | statement shape |
|---|---|---|---|
| production bm25f rescore | 19/40 | 3.05 | ~700 content reads + docsize |
| **native `ORDER BY bm25 LIMIT 300`** | 19/40 | 2.11 | ~300 docsize reads (rank sort) |
| **vocab bm25 (Python, corpus stats)** | 19/40 | 2.11 | ~4 stmts (vocab + docsize IN + avg record) |
| eligible-idf variant (N/df restricted to turns) | 19/40 | 2.05 | ~4 stmts |

Per-task top-10 gold-set disagreements: 4/40 queries swap which gold appears (recall tied).
Python-vocab replication returns the same rank order as native bm25 (identical rank lists).

**Does eligible-set restriction break idf?** Restricting N and df to eligible turns
(N 4143→2877) shifts per-term idf by median **2.5%**, max **684%** for terms living mostly
in non-eligible `sentence_window` rows (df_eligible→0 degenerates). On this corpus recall
survives, but it is a semantic change to the population stats — **not** free. Note the
production bm25f already uses eligible-restricted stats (n_eligible=2877 for avglen), so
corpus-vs-eligible stat choice is the actual semantic decision, not SQL-vs-Python.
The catastrophic `rowid IN (…) AND MATCH` plan (SQLite probes the index once per IN value,
up to ~580k stmts/q) rules out restricting native bm25 inside the query — restrict *after*
ranking, in Python.

## Task 5 — post-prototype census (all arms on)

Flags: `VB_DIRECT_CONTENT VB_UNIT_DOCLEN VB_VOCAB_POSTS VB_TEMPORAL_VOCAB VB_VOCAB_SEEDS
VB_GRAPH_BATCH VB_NO_RESPELL` (respelling kept dark for parity).

| Metric | base | proto | Δ |
|---|---|---|---|
| total stmts/q p50 | 4,250 | **507** | **−88%** |
| total stmts/q p95 | 9,904 | 1,132 | −89% |
| shadow stmts/q p50 | 3,955 | 374 | −91% |
| whole-query ms p50 | 711 | 628 | −12% |
| whole-query ms p95 | 1,796 | 1,651 | −8% |

**SB8-11 target ≤600 stmts/q: met at p50 (507)**; p95 1,132 still above — residual churn is
the stem/trigram channels' per-term MATCHes (`unit_fts_idx` 10,400; `stem_idx` 3,654;
`tri_idx` 1,386 still alive) — the same vocab-instance treatment kills them (~14k → ~300).
500ms latency target: **not met** (628ms p50) — the remaining cost is Python-side bm25f
candidate scoring over ~700 docs/q, not statements.

What died (mean counts/q): `unit_fts_docsize` 3,418→8 (−99.8%);
`unit_fts_content` 724→4.5 (−99.4%); `tbl:units` probes 550→19 (−96%);
`MATCH:unit_fts` ~14→0. What remains: stem/tri idx probes (above), `lex_df` (~23),
`entity_mentions` (~8), `unit_fts_rows` gen lookups (~6), `data_version` pragma (~5).

## Verdicts

1. **docsize dies** — both rank-sort sites (`_derive_seeds` `ORDER BY rank` = 75%,
   `temporal._fts_scores` `ORDER BY bm25` = 25%) replaced by Python bm25 over
   vocab-instance tf + batched docsize + the id=1 averages record. Verified bit-exact;
   with per-doc MATCH-term-order accumulation even ordering reproduces (seeds 52/52,
   items 40/40 identical). Remaining docsize reads (~8/q) are the vocab arms' own
   batched `IN` lookups.
2. **content dies** — `_fetch_fields` direct `unit_fts_content` batched read:
   724→4.5/q, nominated/scores/items identical.
3. **per-term MATCH dies in the lexical channel** — vocab postings keep nomination
   byte-identical (40/40). Extend the same `_stem_vocab`/`_tri_vocab` pattern to the stem
   and trigram lanes to reach the p95 target; this is the only sizable residue.
4. **unit_doclen is worth it architecturally, not for counts** — neutral statements alone,
   but it removes the rescore's dependency on the shadow schema entirely, is drift-free
   (0/20,715), and keeps scoring identical. Generation pinning must come from
   `unit_fts_rows` (batched), not `by_rowid` — universe recs don't carry generation.
5. **Native-vs-Python scoring: ship Python.** Native SQL bm25 cannot be row-restricted
   without the 580k-stmt IN+MATCH blowup; unrestricted-then-filtered native rank and the
   Python replication have identical recall@10 (19/40) and rank order — but the SQL form
   still costs ~300 docsize reads per rank sort while the Python arm costs ~4 statements.
   Population-stats choice (corpus N/df vs eligible N/df) is the real semantic lever
   (median 2.5% idf shift, up to 684% on turn-sparse terms): keep corpus stats to match
   native semantics, or eligible stats to match today's bm25f — pick deliberately.
6. **Manifest seeds (§32.6) are NOT a drop-in** — they change the seed set on 40/40
   queries (lane top-10 vs `_derive_seeds`' own entity-idf+rank top-10) and flip items on
   32/40. That is the spec-intended seeds vs a fallback approximation — a behavior change
   to decide on, not a bug fix. The vocab-seed path preserves byte-identical outputs.

### Maintained-schema patch spec (prototype layout)

- `unit_doclen(scope_id, unit_id, field, generation, len)` WITHOUT ROWID PK
  `(scope_id,unit_id,field,generation)`; written in the projection tx alongside
  `unit_fts_rows`; backfill = `_tokenize` count per field over `unit_fts_content` (0 drift).
- `fts5vocab(unit_fts,'instance')` (+ stem/tri twins) — read views, no storage.
- `_unit_lens(conn, uni, rowids, fields, scope, gen)` → per-rowid generation from
  `unit_fts_rows` (1 batched stmt) then newest doclen `≤ gen` per field (1 batched stmt).
- `_fts_scores_vocab` (temporal) + `_derive_lex_seeds_vocab` (graph): identical Python
  bm25 replication — vocab `GROUP BY` tf, batched docsize `IN`, cached averages record;
  per-doc accumulation in MATCH-term order for bit-exactness.
- `_collect_postings` vocab branch (single-token terms only; fallback to MATCH otherwise).
- `_fetch_fields` → direct `unit_fts_content` `IN`-read; `_GRAPH_BATCH` → one-shot units
  scan for `unit_row`.
- Fallback everywhere: `has_table`/`has_index` gates → original shadow paths.

Artifacts: proto tree `/tmp/verbatim-proto`, harness `/tmp/harness`
(`measure_arms.py`, `task4_accuracy.py`, `proto_schema.sql`, `res_*.json`),
verification: `drift_check.py`, `validate_bm25.py`.
