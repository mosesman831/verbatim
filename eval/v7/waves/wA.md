# Wave A brief — V7 foundation + retrieval core (2026-09-22)

30-slot wave per owner instruction (max 30 concurrent, all SWE-2 Max).
Contracts frozen in `docs/v7_contracts.md` + `verbatim/core/types_v7.py`.

## Lane A — audits (read-only, 6 workers)

Report-only; no file writes. Each returns a defect/gap register that the main
session merges into `eval/v7/audit/` and the D7 register (D7-25+).

| Worker | Scope |
| --- | --- |
| aud-read | consumer search path end-to-end (`memory/facade.py` search, `retrieval/v3/*`, `querying/*`): every SQL statement on the hot path, stage time budget, new defects beyond D7-01..24 |
| aud-write | `Memory.add` -> ingest -> job enqueue -> managed drain: tx boundaries, lock holds, where T0 stages would slot in; defects |
| aud-schema | `storage/schema*.py`, FTS tables, stats, migrations, additive-DDL conventions; what `schema_v7` must coexist with |
| aud-eval | `eval/v3..v6` reusable pieces: ledger format, envelope runner, corpus format, comparator registry, mutation suite mechanics |
| aud-enrich | `enrichment/*`, `evidence/entities.py`: existing canon/alias/temporal/identifier capabilities vs V7-05.10/08/09 needs |
| aud-trust | `querying/verdict.py`, `retrieval/v3/abstain.py`, `pack.py`, `security/*`, `eval/v4/mutations.yaml`: verdict triggers today, mutation anchor conventions, security invariants V7 lanes must honor |

## Lane B — eval harness (7 workers)

| Worker | Closes | Verify |
| --- | --- | --- |
| w-ledger | V7 ledger parses all 349 req ids; gates skeleton reads artifacts only | pytest tests/eval/test_v7_ledger.py |
| w-datasets | dataset registry + LoCoMo local loader (O1-gated) + owned-twin loader | test_v7_datasets.py |
| w-twin-dialogue | seeded LoCoMo-like generator (multi-session dialogue, planted evidence, cat-like slices) | test_v7_twins_locomo.py |
| w-twin-lme | seeded LME-like generator (temporal, knowledge-update, abstention items) | test_v7_twins_lme.py |
| w-twin-extra | actions/prefs/scale generators | test_v7_twins_extra.py |
| w-trackr | Track R runner + metrics (§33) + BM25/FTS5 baseline arms + attribution | test_v7_trackr.py |
| w-profiler | stage_profile/v7 record + aggregation + mutations.yaml scaffold | test_v7_profiler.py |

## Lane C — core modules (15 workers)

| Worker | Closes |
| --- | --- |
| w-norm | `norm/v2` analyzer (§32.1) — D7-01/03 |
| w-intent | `intent/v2` classifier + facet decomposition (§32.14) |
| w-schema | `schema_v7` additive DDL + `stats_v7` incremental stats (§30, V7-06.06) |
| w-entities | canon/mentions/alias rules A1–A6 (V7-08.01–06) — D7-02 |
| w-temporal | `temporal/v2` resolver T01–T30 (V7-09.03–05) — D7-18 |
| w-algebra | interval algebra (V7-09.10) |
| w-events | `event/v1` extraction (V7-09.08, §32.10) |
| w-coref | `coref_sieve/v1` (V7-13.20) |
| w-prefs | `pref/v1` + `state_keys/v1` (§32.11/12) |
| w-fusion | RRF no-ties + feature reranker + bounded boosts (V7-10) — D7-06/07/11 |
| w-verdict | structural verdict (V7-11) — D7-04/05/12 |
| w-policy | policy table + pool profiles + deadline slices (V7-05.02/05/07) — D7-14 |
| w-pipeline | S1–S8 orchestrator (§04.2) — D7-09 lane gating removed by construction |
| w-lexical | fielded BM25F lane + trigram fuzzy lane (V7-06) — D7-08/16 |
| w-graph | edges builder + bounded PPR lane (V7-08.07–12) |
| w-tlane | temporal lane, event-index first (V7-09.06–09) |
| w-dense | contiguous matrix + dense lane honest skeleton (V7-07.04/05) — D7-17 partial |
| w-pack | pack assembly + reader view + computed items + tok/v1 (V7-12) — D7-21 partial |
| w-units | `derive_units` deterministic projection (V7-30.01, V7-13.02–05) — D7-20/21 |

## Exit criteria (V7-27.04, main session)

- every diff reviewed against this brief; focused suites green;
- full suite + mutation suite green on integrated tree;
- owned-twin Track R baseline row recorded (V7 engine still pre-integration;
  run reports current-path numbers as the pre-W2 baseline).
