# verbatim

Local-first agent memory. SQLite. One file. Byte-pinned.

**verbatim** is a retrieval engine for long-term conversational agent
memory. It stores facts as short sentences, each carrying a verbatim
byte-span quote of its source, and retrieves answers the agent can act on.

- **Verbatim quotes** — every stored fact carries the exact source span it
  was distilled from, so answers stay inspectable.
- **Deterministic lanes** — lexical (FTS5 BM25), event-order, temporal,
  entity, bridge, and an optional semantic lane; all fusion is explicit,
  not opaque.
- **Query lanes are generated, not hard-coded** — 28 plan generators turn a
  question into lane queries (exact, temporal, entity, lexical, semantic).
- **Conflict-aware** — every answer carries a support/conflict verdict;
  supersession is first-class.
- **Offline by default** — no network calls, no hosted model required.
  Extras are opt-in.

## Install

    pip install verbatim-memory

## Quickstart

```python
from verbatim import Memory

mem = Memory()                                    # or Memory("path/to/store.vdb")

added = mem.add("the release tag is v6.2", metadata={"topic": "release"})
mem.wait_ready(added.receipt_id)                  # bounded wait for indexing

res = mem.search("what is the release tag?")
print(res.status, res.items[0].quote)             # -> "the release tag is v6.2"

detail = mem.inspect(res.items[0].ref)            # provenance + evidence spans
mem.forget(res.items[0].ref)                      # targeted, CAS-guarded
mem.close()
```

`add` is durable when it returns, but derived work (lexical indexing,
optional embeddings) settles asynchronously — the receipt is the handle.
`search` waits up to `ready_timeout_ms` by default and reports
`res.status` (`ready`/`partial`/`pending`/`blocked`). Every result item
carries the verbatim source span, so answers stay inspectable; the store
is a single SQLite file you can copy, back up, or open with any SQLite
tool.

## How it works

Retrieval runs in three explicit stages. **Eligibility** decides which
sentences are even allowed to answer (temporal, entity, and scope filters
applied before ranking). **Lanes** score eligible candidates in parallel —
each lane sees the query through its own lens, and each exposes its score
and vetoes. **Fusion + verdict** merges lane evidence into a final
ranking, resolves conflicts (e.g., a newer fact superseding an older one),
and emits the answer with its evidence and trace. Nothing is a black box:
you can read exactly why each sentence won or lost.

## Benchmarks

Measured on the [AMB (Agent Memory Benchmark)](https://github.com/mosesman831/agent-memory-benchmark)
harness — answer model `mimo-v2.6-flash`, judge `nemotron-3.5-lightning`,
temperature 0. Full matrix, recipes, and partial-coverage notes:
[BENCHMARKS.md](BENCHMARKS.md) · [eval/amb/RECOMMENDED_ENV.md](eval/amb/RECOMMENDED_ENV.md)

| Benchmark | Verbatim | Best published | Best published score |
|---|---|---|---|
| LoCoMo-10 (all cats ≥90) | **94.94%** | MemMachine | 91.7% |
| PrecisionMemBench | **100%** | open-knowledge-format | 46.75% |
| SDEBench | **100%** | — | (no published baseline) |
| LifeBench (all 10 units) | **83.2%** | MemOS | 55.22% |
| LongMemEval-S | **87.8%** | Mastra | 92.8% |
| MSC-MemFuse-MC10 | **98.6%** | — | (no published baseline) |
| MemBench reflective | **74.7%** | — | (no published baseline) |
| PersonaMem-32k | **73.0%** | — | — |
| PersonaMem-128k | **67.9%** | Ever-EOS | ~45% |
| BEAM | **67.5%** | — | (baseline pending) |

## HTTP + MCP + TypeScript

Three surfaces wrap the same engine (extras are opt-in):

- `python -m verbatim.http` — FastAPI server (`/memory`, `/retrieve`,
  `/documents`, `/documents/{id}/verbatim`).
- `python -m verbatim.mcp` — MCP server exposing `verbatim_add` and
  `verbatim_retrieve` tools for agent hosts.
- `clients/ts` — TypeScript client mirroring the Python store API.

## Development

The `verbatim/` package is the source of truth; `src/` is legacy V1 code
kept for reference. Tests (403) run offline against tiny fixtures in
`tests/` — no server required:

    pytest -q

Optional extras: `pip install verbatim-memory[semantic]` (numpy embedding
lane), `verbatim-memory[dev]` (pytest + pyyaml for the test suite).

MIT.
