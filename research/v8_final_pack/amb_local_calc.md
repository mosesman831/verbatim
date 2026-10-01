# AMB token-budget sizing — measured on dev store (v8-local-calc)

Source: track_r postfix run (HEAD 1c86f4f), dev split, verbatim arm, 990 q / 777 answerable.
Token method: tiktoken cl100k_base (identical to AMB `count_tokens`) on AMB-style
render `## Memory {i}\n{turn_text}` joined with \n\n.

## Measured
- Pack tokens per query: median 1,200 / p90 1,500 / max 1,821 (limit=20 items)
- Tokens preceding first gold item: median 82 / p90 432 (n=527 delivered-gold)
- First-gold rank in pack: median 3 / p90 13
- 527/527 delivered-gold questions: first gold inside 4,500-token budget

## Implications for AMB adapter (V8-15.06)
1. Verbatim's full 20-item pack already fits ≤2,000 tokens → the 4,500 budget has
   ~3x headroom: the adapter can ship MORE items (limit ~30-60) or pack neighbors.
   Arm: limit {20, 40, 60} at budgets {2,000, 4,500, 9,000}.
2. Do NOT return raw_response JSON — LoCoMo build_rag_prompt dumps raw_response
   as context when present (json.dumps over whole dict) — that is how Hindsight
   hits 36,235 tokens. Return raw_response=None.
3. retrieve_time_ms = wall around memory.async_retrieve only → the ≤100ms target
   is measured on verbatim search() alone (no render, no jobs).
4. ingest: AMB gives one Document per (sample_id, session) with timestamp +
   user_id=sample_id → map to Memory.add(session_text, occurred=session_ts,
   speaker scope=user_id). ingest/doc = add-ack + drain per doc (SB8-14 ≤150ms).
5. PrecisionMemBench --mode retrieval needs Document.source_ids = the ingested
   Document.id(s) each result came from → adapter needs a doc_id→unit_id map
   persisted in store_dir (units carry source_id already — check item_ref
   propagation through units.source_id→source external ref).
6. concurrency: AMB default 4 parallel retrieves. Verbatim Engine on shared
   SQLite conn — check thread-safety; if not, set provider concurrency=1 or
   open a per-call Engine (cheap? measure init cost).
7. query_timestamp is passed to retrieve() → maps to as_of (V8-09.01) with
   dataset-provided time (LoCoMo = last session date, per V8-09.02).
8. Provider file: src/memory_bench/memory/verbatim.py implementing
   MemoryProvider (name='verbatim', kind='local', concurrency TBD,
   supports_filters=False); register in memory/__init__.py + catalog.json.
9. Reader sees "## Memory i\n<content>" blocks — put speaker+date into content
   itself (e.g. "Caroline (2023-05-07): ...") since instructions tell the reader
   to convert relative times using timestamps in context.
10. Rank quality is doubly load-bearing on AMB: first gold at median rank 3,
    p90 rank 13 — the reader reads top-down; MRR fixes feed accuracy directly.

## Concurrency (checked on box)
- SQLite conns open `check_same_thread=False` (storage/store.py:393,541,828)
- but Engine._lock guards only SESSION-barrier receipts, not the V7 search
  pipeline — concurrent retrieves share one connection without serialization.
- Adapter rec: provider `concurrency = 1` for v1 (safe, ~5min serial for 990q
  at ~300ms). Follow-up: a pool of N Engine instances per provider (each opens
  its own read conn) if wall-clock becomes painful; measure Engine init cost.
