# Verbatim

Evidence-first, local-first memory for AI agents. SQLite-backed, host-neutral,
and built around one rule: **every memory traces back to byte-exact evidence.**
Verbatim never generates memory content - it stores verbatim source text,
derives claims with pointers to exact spans, and reports degraded lanes
honestly instead of silently serving weaker results.

## Install & first memory

```bash
pip install verbatim-memory
```

```python
from verbatim import Memory

mem = Memory()                                    # or Memory("path/to/store.vdb")

added = mem.add("the release tag is v6.2", metadata={"topic": "release"})
mem.wait_ready(added.receipt_id)                  # bounded wait for indexing

res = mem.search("what is the release tag?")
print(res.status, res.items[0].quote)             # -> "the release tag is v6.2"

detail = mem.inspect(res.items[0].ref)            # provenance + evidence spans

mem.forget(res.items[0].ref)                      # targeted, CAS-guarded
# or query preview -> confirmation token -> execute exactly that selection:
# preview = mem.forget(query="release tag")
# mem.forget(confirmation=preview.confirmation_token)

mem.close()
```

### Readiness, honestly

`add` is durable when it returns, but derived work (lexical indexing,
optional embeddings) settles asynchronously - the receipt is the handle.
`wait_ready(receipt)` blocks until ready/partial/blocked; `search` with the
default `consistency="session"` already waits up to `ready_timeout_ms` and
reports `res.status` (`ready`/`partial`/`pending`/`blocked`/`unavailable`)
plus `warnings` rather than pretending. `mem.status()` reports what this
build actually provisioned - never config-implied health.

## How it works

Retrieval runs in explicit stages over a fixed lane set (`lex`, `fuzzy`,
`dense`, `ent`, `time`, `graph`, `typed`, `obs`, `exact_id`, `source`).
**Eligibility** decides which sentences may answer at all (temporal, entity
and scope filters applied before ranking). **Lanes** score the eligible
candidates in parallel, each publishing its own score and vetoes.
**Fusion and verdict** merge lane evidence into a final ranking, resolve
conflicts (a newer fact superseding an older one), and emit the answer with
its evidence and trace. Nothing is a black box: you can read exactly why
each sentence won or lost, and a lane that cannot apply eligibility reports
an error with zero candidates rather than unfiltered output.

## Surfaces

- **HTTP service** - `verbatim-service --path store.vdb --user me --token SECRET`
  serves the bearer-authenticated `/v2/memory/*` routes
  (add/search/inspect/forget/status/readiness/capabilities) on loopback.
  Unauthenticated requests fail closed with 401. `python -m verbatim.service`
  is the same launcher.
- **MCP** - `verbatim-mcp` serves the consumer tool profile
  (`v5_capture`/`v5_recall`) over stdio for agent hosts.
- **TypeScript client** - `clients/ts/memory-client`
  (`@verbatim/memory-client`): zero-dependency `fetch` client mirroring the
  `/v2/memory` shapes.
- **CLI** - `verbatim` exposes the operator surface (ingest, inspect,
  status, doctor, export, purge, jobs).

## Properties

- **Evidence-preserving** - source revisions, byte-range spans, HMAC-verified
  digests, claim->evidence edges; read-time integrity fails closed
  (`STORE_CORRUPT`) rather than serve unverified content.
- **Local-first** - one SQLite file, no server, no GPU, no vector database;
  `cryptography` is the only hard dependency.
- **Host-neutral** - Python `Memory` facade, HTTP service, MCP server, and
  a Hermes memory-provider entry point; hosts are thin adapters.
- **Honest capabilities** - every operation reports healthy / degraded /
  unavailable; optional lanes (semantic embeddings, neural encoder) degrade
  explicitly instead of silently serving weaker results.
- **Deletion closure** - forget suppresses and purges through the
  dependency graph; scoped partitions fence every read.

Optional extras: `pip install verbatim-memory[semantic]` (numpy embedding
lane), `verbatim-memory[dev]` (pytest, pytest-xdist, pyyaml).

## Benchmarks

Measured on the `agent-memory-benchmark` harness (deterministic retrieval,
mimo reader + nemotron judge at temperature 0; MCQ rows unjudged):

| dataset | score |
|---|---|
| locomo `locomo10` | **94.94%** - all categories >=90 |
| precisionmembench | **100%** (P = R = 1.00) |
| sdebench `boltons` | **100%** - first measured row |
| lifebench `en` | **83.2%** - all 10 units |
| longmemeval `s` | **87.80%** |
| personamem `32k` / `128k` | **73.0%** / **67.9%** |
| beam `100k` | **67.5%** |
| msc_memfuse | **98.6%** |
| membench reflective | **74.7%** |

Published rows on the same harness: locomo 94.94% vs MemMachine 91.7%;
lifebench 83.2% vs MemOS 55.22%; precisionmembench 100% vs
open-knowledge-format 46.75%; longmemeval-s 87.80% vs Chronos 95.6% and
Mastra 92.8%. Those rows use a different reader/judge pairing than ours, so
the comparison is directional rather than strict apples-to-apples - the
longmemeval row is a genuine loss. Full matrix, per-dataset recipes,
judge-churn bounds and the freeze-verification gate live in
[`BENCHMARKS.md`](BENCHMARKS.md) and
[`eval/amb/RECOMMENDED_ENV.md`](eval/amb/RECOMMENDED_ENV.md).

## Development

```bash
python -m pytest tests/ -n auto     # full suite, parallel (recommended)
python -m pytest tests/             # same suite, serial
python -m eval.v3.run               # internal eval harness (see eval/v3/)
```

The suite runs offline against tiny fixtures in `tests/` - no server, no
network. CI (`.github/workflows/ci.yml`) runs it on Python 3.11 and 3.12,
plus a packaging job that builds the distributions, checks metadata with
`twine`, installs the wheel into a clean environment and runs
`tools/ci_smoke.py` (capture -> readiness -> recall -> inspect -> deletion
closure, plus a bearer-auth check on the HTTP surface).

Releases are cut by pushing a `v*` tag: `.github/workflows/release.yml`
builds, verifies the wheel, publishes to PyPI via trusted publishing, and
attaches the distributions to the GitHub release.

Pre-release, active development. The internal spec corpus (SPEC v1-v8.5,
REQUIREMENTS, THREAT_MODEL) is maintained privately and is not part of this
distribution. `RELEASE_MANIFEST_V4.json` records declared capability and gate
status for this codebase, including declared limitations and the measured
targets that currently miss - see `known_limitations`.

## License

MIT - see `LICENSE`.