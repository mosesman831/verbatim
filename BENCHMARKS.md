# Benchmarks

Measured verbatim results on the `agent-memory-benchmark` (AMB) harness and
adjacent suites. Every number was measured, not projected — the per-dataset
environment recipe that produced each row lives in
[`eval/amb/RECOMMENDED_ENV.md`](eval/amb/RECOMMENDED_ENV.md).

Reader/judge for judged datasets: `mimo-v2.6-flash-free` (answer) +
`nemotron-3.5-lightning-free` (judge) at temperature 0. MCQ datasets need no
judge. Deterministic retrieval mode (`VERBATIM_AMB_SEARCH_TIMEOUT_MS=0`) unless
noted.

## Board standings

| dataset | split | score |
|---|---|---|
| locomo | `locomo10` (1,540q) | **94.94% det** — integ-merge reproduces 95.0%, all categories ≥90 |
| precisionmembench | `single-turn` (77q) | **100%** — P = R = 1.00 |
| sdebench | `boltons` (61 tasks) | **100%** — first measured row |
| lifebench | `en` (3,385q) | **83.2% det mean — all 10 units verified** |
| longmemeval | `s` (500q) | **87.80% det** |
| personamem | `32k` (589q) | **73.0% det** |
| personamem | `128k` (2,727q) | **67.9% det** |
| beam | `100k` (400q) | **67.5% det** (up from 8.25% first-measured — ingest-governance fix) |
| msc_memfuse | `main` (500q, MCQ) | **98.6%** |
| membench | `first_high` + `third_high` (4,000q, MCQ) | **74.7% reflective** |

### Published-baseline comparisons (harness `external_results.json`)

| dataset | verbatim | best published row |
|---|---|---|
| locomo | 94.94% | MemMachine 91.7% |
| lifebench | 83.2% | MemOS 55.22% |
| precisionmembench | 100% | open-knowledge-format 46.75% |
| longmemeval | 87.80% | Chronos 95.6% · Mastra 92.8% |

Same datasets, different reader/judge pairing than the published rows —
directionally strong, not strict apples-to-apples.

## Regression gate (freeze verification)

Integ tree = `main` + the frozen eval-side stack (PRs #12–#15) octopus-merged;
all four legs pass within judge churn:

| board | reference | integ-merge | verdict |
|---|---|---|---|
| locomo10 det | 94.94% | **95.3%** main / **95.0%** integ | pass |
| personamem/32k det | 73.0% | **72.0%** | pass |
| longmemeval/s det | 87.80% | **87.40%** | pass |

## Bonus-lane partials (harvested at freeze cutoff)

- **longmemeval/m union: 54.88%** (90/164 deduped questions scored of 500 —
  coverage stopped mid-flight). M is far harder than S: multi-session 43.9%,
  temporal 24.1% dominate misses; knowledge-update 82.8%.
- **membench LowLevel** (19,166q splits, partial): `first_low` **87.9%**,
  `third_low` **92.2%** on scored subsets — the LowLevel tiers read far
  easier than the reflective highs.
- **membench Emotion ~26.6%** is documented **reader/inference-bound**
  (clean contexts, letter-biased reader; the paper itself marks the
  reflective tier hard) — deliberately not optimized.

## Reproduce

Each row's env block is in `eval/amb/RECOMMENDED_ENV.md`. In short:

```bash
cd agent-memory-benchmark
uv run amb run --dataset <name> --split <split> --memory verbatim
```

Budget map (the key finding — no universal default exists): lifebench / beam /
longmemeval `16000` · locomo `6500` · precisionmembench / personamem /
sdebench `4500`. QX on for locomo + longmemeval only; off for lifebench +
personamem.
