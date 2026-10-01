# V5 Requirements Ledger — Summary

Generated from `SPEC_V5.md` by `tools/gen_v5_ledger.py`.
Do not hand-edit: curate `eval/v5/dispositions_v5.json` and
re-run with `--write`. Sibling machine ledger:
`eval/v5/ledger_v5.json`.

Status model (V5-00.05/27.05): `implementation_status` and
`qualification_status` are separate axes; `status` is the
derived rollup — `[x]` qualified, `[m]` locally_measured
(implemented and measured here, still unqualified), `[u]`
implemented_unmeasured, `[~]` in_progress, `[ ]` planned,
`[!]` failed, `[n]` not_applicable. `locally_measured` and
`qualified` require named executed evidence; nothing here
infers status from counts.

## Totals

- requirements: 369
- status `planned`: 20
- status `in_progress`: 0
- status `implemented_unmeasured`: 0
- status `locally_measured`: 329
- status `qualified`: 0
- status `failed`: 0
- status `not_applicable`: 20
- qualification `not_run`: 0
- qualification `locally_measured`: 329
- qualification `qualified`: 0
- qualification `failed`: 0
- qualification `not_applicable`: 20
- qualification `deferred`: 20
- scenarios: 96 (E01–E96), all `not_run`
- gates: 15 (G5-00–G5-14), all `not_run`
- requirements with ≥1 applicable scenario: 340
- requirements with a bound gate: 36
- disposition overrides: 369

## By section

| Section | Title | Reqs | Stage | Statuses |
| --- | --- | --- | --- | --- |
| §00 | Reading map, authority, and acceptance language | 7 | P0 | locally_measured=2 not_applicable=5 |
| §01 | Product thesis and what winning means | 6 | P0 | locally_measured=4 not_applicable=2 |
| §02 | Audited starting point and corrected assumptions | 5 | P0 | locally_measured=5 |
| §03 | Explicit contract amendments and inherited obligations | 7 | P1 | locally_measured=7 |
| §04 | Product profiles and feature layering | 6 | P0/P2 | locally_measured=6 |
| §05 | Identity, store bootstrap, and retention consent | 14 | P1 | locally_measured=14 |
| §06 | Public API and result contracts | 18 | P1 | locally_measured=18 |
| §07 | Add, source search, and inference without truth laundering | 13 | P1 | locally_measured=13 |
| §08 | Causal readiness and read-your-writes | 19 | P1 | locally_measured=19 |
| §09 | Managed worker lifecycle and reliability | 14 | P1 | locally_measured=14 |
| §10 | Default retrieval route: two lanes, one authority | 10 | P2 | locally_measured=10 |
| §11 | Authorization-equivalent indexes and exact statistics | 12 | P2 | locally_measured=12 |
| §12 | Exact-first acceleration and ANN admission | 10 | P2 | locally_measured=8 not_applicable=2 |
| §13 | Packing, contrary evidence, and truthful result state | 10 | P1 | locally_measured=10 |
| §14 | Corrections, supersession, and the stale-deploy demonstration | 16 | P1 | locally_measured=16 |
| §15 | Inspect, forget, and deletion closure | 10 | P1 | locally_measured=10 |
| §16 | Caches, prefetch, snapshots, and delivery races | 10 | P2 | locally_measured=9 not_applicable=1 |
| §17 | Neural quality and optional advanced memory | 10 | P2 | locally_measured=9 not_applicable=1 |
| §18 | Distribution and Mem0 migration | 13 | P3 | locally_measured=12 planned=1 |
| §19 | Hermes, two-tool MCP, framework adapters, and TypeScript | 14 | P3 | locally_measured=13 not_applicable=1 |
| §20 | Adoption performance envelopes and resource budgets | 14 | P2 | locally_measured=12 not_applicable=1 planned=1 |
| §21 | Quality, operations, usability, and the real task portfolio | 11 | P2 | locally_measured=8 not_applicable=1 planned=2 |
| §22 | Competitive execution: beat strong systems, not caricatures | 12 | P4 | locally_measured=11 not_applicable=1 |
| §23 | Statistical decision rules and claim discipline | 12 | P4 | locally_measured=11 not_applicable=1 |
| §24 | Acceptance scenarios and verification matrix | 7 | P0 | locally_measured=5 planned=2 |
| §25 | Release gates and honest claim levels | 6 | — | locally_measured=1 not_applicable=1 planned=4 |
| §26 | Implementation plan after separate approval | 6 | P0 | locally_measured=3 not_applicable=2 planned=1 |
| §27 | Traceability, amendment mapping, and approval decisions | 6 | P0 | locally_measured=6 |
| §29 | Competitor strengths and the V5 answer | 4 | P4 | locally_measured=2 planned=2 |
| §30 | Grounded memory operations: extraction that cannot lie | 25 | P1 | locally_measured=22 planned=3 |
| §31 | Multi-signal ranking under exact eligibility | 10 | P2 | locally_measured=8 not_applicable=1 planned=1 |
| §32 | Token efficiency, long history, and grounded consolidation | 11 | P4 | locally_measured=10 planned=1 |
| §33 | Performance engineering program | 12 | P2 | locally_measured=10 planned=2 |
| §34 | Multi-agent, multi-namespace, and shared memory | 5 | P3 | locally_measured=5 |
| §35 | Learning from use without learning authority | 4 | P5 | locally_measured=4 |

## By stage

| Stage | Reqs | Exit gates |
| --- | --- | --- |
| P0 | 38 |  |
| P1 | 146 | G5-00, G5-01, G5-02 |
| P2 | 104 | G5-02, G5-03 |
| P3 | 32 | G5-04, G5-05, G5-06, G5-07, G5-13 |
| P4 | 39 | G5-08, G5-09, G5-14 |
| P5 | 4 |  |
| P6 | 0 | G5-10, G5-11 |
| (unbound) | 6 | |

## Gates

| Gate | Name | Status | Bound reqs |
| --- | --- | --- | --- |
| G5-00 | Authority continuity | not_run | 0 |
| G5-01 | Consumer lifecycle | not_run | 0 |
| G5-02 | Retrieval correctness | not_run | 0 |
| G5-03 | Adoption performance | not_run | 0 |
| G5-04 | Default selection | not_run | 0 |
| G5-05 | Semantic quality | not_run | 0 |
| G5-06 | Distribution and DX | not_run | 0 |
| G5-07 | Host/transport parity | not_run | 0 |
| G5-08 | Agent utility | not_run | 0 |
| G5-09 | Competitive execution | not_run | 0 |
| G5-10 | Leadership | not_run | 0 |
| G5-11 | Independent reproduction | not_run | 0 |
| G5-12 | Operational release integrity | not_run | 0 |
| G5-13 | Memory-operations quality | not_run | 25 |
| G5-14 | Token efficiency and long history | not_run | 11 |

## Scenarios (E01–E96)

| Scenario | Stage | Primary sections | Anchored reqs | Status |
| --- | --- | --- | --- | --- |
| E01 | P1 | §03, §06 | 25 | not_run |
| E02 | P1 | §06, §07, §08 | 50 | not_run |
| E03 | P1 | §05, §08 | 33 | not_run |
| E04 | P1 | §05 | 14 | not_run |
| E05 | P1 | §05, §06 | 32 | not_run |
| E06 | P1 | §05, §09 | 28 | not_run |
| E07 | P1 | §07 | 13 | not_run |
| E08 | P1 | §07 | 13 | not_run |
| E09 | P1 | §07 | 13 | not_run |
| E10 | P1 | §07 | 13 | not_run |
| E11 | P1 | §07 | 13 | not_run |
| E12 | P1 | §08 | 19 | not_run |
| E13 | P1 | §08 | 19 | not_run |
| E14 | P1 | §08 | 19 | not_run |
| E15 | P1 | §08 | 19 | not_run |
| E16 | P1 | §08 | 19 | not_run |
| E17 | P1 | §09 | 14 | not_run |
| E18 | P1 | §09 | 14 | not_run |
| E19 | P1 | §09 | 14 | not_run |
| E20 | P1 | §09 | 14 | not_run |
| E21 | P1 | §09 | 14 | not_run |
| E22 | P1 | §07, §09 | 27 | not_run |
| E23 | P2 | §04, §10 | 16 | not_run |
| E24 | P2 | §10, §13 | 20 | not_run |
| E25 | P2 | §10, §11, §12 | 32 | not_run |
| E26 | P2 | §11 | 12 | not_run |
| E27 | P2 | §11 | 12 | not_run |
| E28 | P2 | §10, §11 | 22 | not_run |
| E29 | P2 | §11 | 12 | not_run |
| E30 | P2 | §11 | 12 | not_run |
| E31 | P2 | §11 | 12 | not_run |
| E32 | P2 | §12 | 10 | not_run |
| E33 | P2 | §12 | 10 | not_run |
| E34 | P2 | §12 | 10 | not_run |
| E35 | P2 | §12 | 10 | not_run |
| E36 | P1 | §03, §13 | 17 | not_run |
| E37 | P2 | §13 | 10 | not_run |
| E38 | P2 | §13 | 10 | not_run |
| E39 | P1 | §14 | 16 | not_run |
| E40 | P1 | §14 | 16 | not_run |
| E41 | P1 | §14 | 16 | not_run |
| E42 | P2 | §14 | 16 | not_run |
| E43 | P1 | §15 | 10 | not_run |
| E44 | P1 | §15 | 10 | not_run |
| E45 | P1 | §15 | 10 | not_run |
| E46 | P2 | §11, §15 | 22 | not_run |
| E47 | P2 | §16 | 10 | not_run |
| E48 | P2 | §16 | 10 | not_run |
| E49 | P2 | §16 | 10 | not_run |
| E50 | P2 | §17 | 10 | not_run |
| E51 | P3 | §17 | 10 | not_run |
| E52 | P3 | §17 | 10 | not_run |
| E53 | P3 | §17 | 10 | not_run |
| E54 | P3 | §18 | 13 | not_run |
| E55 | P3 | §18 | 13 | not_run |
| E56 | P3 | §18 | 13 | not_run |
| E57 | P3 | §18 | 13 | not_run |
| E58 | P3 | §19 | 14 | not_run |
| E59 | P3 | §19 | 14 | not_run |
| E60 | P3 | §19 | 14 | not_run |
| E61 | P5 | §19 | 14 | not_run |
| E62 | P2 | §20 | 14 | not_run |
| E63 | P2 | §06, §13 | 28 | not_run |
| E64 | P2/P3 | §21 | 11 | not_run |
| E65 | P3 | §21 | 11 | not_run |
| E66 | P4 | §22 | 12 | not_run |
| E67 | P4 | §22 | 12 | not_run |
| E68 | P4 | §21, §22, §23 | 35 | not_run |
| E69 | P4 | §23 | 12 | not_run |
| E70 | P4 | §22, §23 | 24 | not_run |
| E71 | P6 | §23 | 12 | not_run |
| E72 | All | §25, §26, §27 | 18 | not_run |
| E73 | P3 | §30 | 25 | not_run |
| E74 | P1 | §30 | 25 | not_run |
| E75 | P3 | §30 | 25 | not_run |
| E76 | P1 | §30 | 25 | not_run |
| E77 | P1 | §30 | 25 | not_run |
| E78 | P1 | §30 | 25 | not_run |
| E79 | P2 | §30 | 25 | not_run |
| E80 | P3 | §30 | 25 | not_run |
| E81 | P3 | §30 | 25 | not_run |
| E82 | P3 | §30 | 25 | not_run |
| E83 | P2 | §31 | 10 | not_run |
| E84 | P2 | §31 | 10 | not_run |
| E85 | P2 | §31 | 10 | not_run |
| E86 | P2 | §31 | 10 | not_run |
| E87 | P5 | §31 | 10 | not_run |
| E88 | P3 | §34 | 5 | not_run |
| E89 | P3 | §34 | 5 | not_run |
| E90 | P3 | §34 | 5 | not_run |
| E91 | P3 | §34 | 5 | not_run |
| E92 | P4 | §32 | 11 | not_run |
| E93 | P4 | §32, §33 | 23 | not_run |
| E94 | P4 | §32 | 11 | not_run |
| E95 | P2 | §33 | 12 | not_run |
| E96 | P5 | §35 | 4 | not_run |

Section-level anchors are the planning map (V5-27.02);
qualification requires named assertions per requirement,
recorded as `executed_evidence` in `dispositions_v5.json`.
