# V6 Requirements Ledger — Summary

Generated from `SPEC_V6.md` by `tools/gen_v6_ledger.py`.
Do not hand-edit: curate `eval/v6/dispositions_v6.json` and
re-run with `--write`. Sibling machine ledger:
`eval/v6/ledger_v6.json`.

Status model (V6-00.06): `implementation_status` and
`qualification_status` are separate axes; `status` is the
derived rollup — `[x]` qualified, `[m]` locally_measured
(implemented and measured here, still unqualified), `[u]`
implemented_unmeasured, `[~]` in_progress, `[ ]` planned,
`[!]` failed, `[n]` not_applicable. `locally_measured` and
`qualified` require named executed evidence; nothing here
infers status from counts.

## Totals

- requirements: 69
- status `planned`: 3
- status `in_progress`: 0
- status `implemented_unmeasured`: 2
- status `locally_measured`: 60
- status `qualified`: 0
- status `failed`: 0
- status `not_applicable`: 4
- qualification `not_run`: 5
- qualification `locally_measured`: 60
- qualification `qualified`: 0
- qualification `failed`: 0
- qualification `not_applicable`: 4
- qualification `deferred`: 0
- scenarios: 32 (F01–F32), all `not_run`
- gates: 9 (G6-00–G6-08), all `not_run`
- carried V5 tail (V6-04.02/03): 20 rows (planned=20 implemented_unmeasured=0)
- requirements with ≥1 applicable scenario: 53
- requirements with a bound gate: 55
- disposition overrides: 69

## By section

| Section | Title | Reqs | Stage | Statuses |
| --- | --- | --- | --- | --- |
| §00 | Authority and what V6 is allowed to change | 8 | P0 | locally_measured=5 not_applicable=3 |
| §01 | Adoption thesis | 13 | P3 | implemented_unmeasured=2 locally_measured=11 |
| §02 | Performance thesis — small facts on the hot path | 16 | P1/P2 | locally_measured=15 planned=1 |
| §03 | Trustworthy defaults — better than extraction-only memory | 16 | P4/P5 | locally_measured=16 |
| §04 | Qualification closure — the V5 tail | 5 | P6 | locally_measured=3 planned=2 |
| §05 | Service surface — Memory over HTTP | 5 | P3 | locally_measured=5 |
| §06 | Comparators — evidence that can actually run | 5 | P6 | locally_measured=5 |
| §08 | Gates G6-00–G6-08 | 1 | — | not_applicable=1 |

## By phase

| Phase | Reqs | Exit scenarios | Exit gates |
| --- | --- | --- | --- |
| P0 | 8 |  |  |
| P1 | 6 | F17, F18, F19, F20 |  |
| P2 | 10 |  |  |
| P3 | 18 | F01, F15, F22, F23, F24 |  |
| P4 | 11 | F25, F26, F27, F28 |  |
| P5 | 5 | F29, F30 | G6-02 |
| P6 | 10 | F31, F32 | G6-05, G6-07, G6-08 |
| P7 | 0 |  |  |
| (unbound) | 1 | | |

## Gates (G6-00–G6-08)

| Gate | Name | Scenarios named | Status | Bound reqs |
| --- | --- | --- | --- | --- |
| G6-00 | Authority |  | not_run | 1 |
| G6-01 | Instant personal search | F17, F18, F19, F20, F21 | not_run | 10 |
| G6-02 | Recommended quality | F29, F30 | not_run | 7 |
| G6-03 | Trustworthy pack | F05, F06, F07, F13, F25, F26 | not_run | 12 |
| G6-04 | Adoption surface | F01, F10, F11, F12, F15, F22, F23, F24 | not_run | 19 |
| G6-05 | Local OSS comparison | F32 | not_run | 12 |
| G6-06 | Named leadership |  | not_run | 2 |
| G6-07 | Long history | F21 | not_run | 1 |
| G6-08 | Qualification closure | F31 | not_run | 10 |

## Scenarios (F01–F32)

| Scenario | Phase | Bound gates | Anchored reqs | Status |
| --- | --- | --- | --- | --- |
| F01 | P3 | G6-04 | 18 | not_run |
| F02 | — |  | 0 | not_run |
| F03 | — |  | 0 | not_run |
| F04 | — |  | 0 | not_run |
| F05 | — | G6-03 | 0 | not_run |
| F06 | — | G6-03 | 0 | not_run |
| F07 | — | G6-03 | 0 | not_run |
| F08 | — |  | 0 | not_run |
| F09 | — |  | 4 | not_run |
| F10 | — | G6-04 | 0 | not_run |
| F11 | — | G6-04 | 0 | not_run |
| F12 | — | G6-04 | 0 | not_run |
| F13 | — | G6-03 | 0 | not_run |
| F14 | — |  | 0 | not_run |
| F15 | P3 | G6-04 | 18 | not_run |
| F16 | — |  | 0 | not_run |
| F17 | P1 | G6-01 | 6 | not_run |
| F18 | P1 | G6-01 | 6 | not_run |
| F19 | P1 | G6-01 | 6 | not_run |
| F20 | P1 | G6-01 | 7 | not_run |
| F21 | — | G6-01, G6-07 | 0 | not_run |
| F22 | P3 | G6-04 | 18 | not_run |
| F23 | P3 | G6-04 | 18 | not_run |
| F24 | P3 | G6-04 | 18 | not_run |
| F25 | P4 | G6-03 | 11 | not_run |
| F26 | P4 | G6-03 | 11 | not_run |
| F27 | P4 |  | 11 | not_run |
| F28 | P4 |  | 11 | not_run |
| F29 | P5 | G6-02 | 5 | not_run |
| F30 | P5 | G6-02 | 5 | not_run |
| F31 | P6 | G6-08 | 10 | not_run |
| F32 | P6 | G6-05 | 10 | not_run |

## Carried V5 tail (V6-04.02/03)

20 unfinished V5 rows carried from
`eval/v5/dispositions_v5.json` (read-only; the V5 file is never
edited). Each MUST be re-evidenced or explicitly
re-deferred in this wave.

| V5 id | Status | Qualification | Note |
| --- | --- | --- | --- |
| V5-18.01 | planned | deferred | Deferred — metadata chooses hermes-verbatim dist / verbatim import, but the legally-verified distribution-identity + import-coexistence decision record (the pypi `verbatim` namespace is occupied by an unrelated project) is not produced; owed before public install instructions beyond the repo README. |
| V5-20.01 | planned | deferred | Deferred — no controlled reference machine is pinned; this host is a loaded shared aarch64 box (loadavg ~1.8-8.8 recorded during runs) and cannot serve as the pinned reference. environment() records actuals per run in the meantime. |
| V5-21.03 | planned | deferred | Deferred — the >=200-paired-task cross-host floor needs real Hermes + ADK installs that are absent here: not measurable in this environment. The A-12 narrower named-host path remains open. |
| V5-21.06 | planned | deferred | Deferred — pinned permitted LongMemEval/LoCoMo corpora are license-restricted and absent from this tree; the owned synthetic corpus remains the executed stratum. Restricted datasets stay blocked until permission exists. |
| V5-24.03 | planned | deferred | Deferred — property/state-machine tests generating add/retry/search/replace/hold/revoke/forget/restart histories against an authority reference model are unauthored; the scenario suites remain the executed stratum. |
| V5-24.06 | planned | deferred | Deferred — controlled isolated/quiescent worktrees for qualification runs are not established; this wave's re-executed suites ran on a loaded host (loadavg recorded per run), so they count as regression evidence, not qualification evidence. |
| V5-25.01 | planned | deferred | Deferred — the six manifest axes (implementation, conformance, host qualification, performance, statistical outcome, independent reproduction) bind at release; RELEASE_MANIFEST_V4.json demonstrates the separated-axis pattern but no V5/V6 release manifest exists. |
| V5-25.02 | planned | deferred | Deferred — the early-exposed optional-feature safety rule binds when a release runs; none executed. Optional lanes shipped today (semantic extra, learned controller) stay fail-closed or honestly degraded under their own gates. |
| V5-25.03 | planned | deferred | Deferred — the failed-gate record (owner, cause, evidence, next verification, narrower claims) binds when a release gate fails; no release gate has run. The retention machinery itself is measured (V5-20.14). |
| V5-25.04 | planned | deferred | Deferred — the publication wording rule binds an A0-qualified consumer publication; none is authorized, so the 'does not certify S0/S1 or complete M0/M5' wording has no artifact to attach to. |
| V5-26.06 | planned | deferred | Deferred — build-completion is partially exercised (predecessor regression suites re-executed green and ledgers regenerated this wave); executed public examples and actual SLO reports remain owed. |
| V5-29.01 | planned | deferred | Deferred — the comparator ladder executed measured slices (v6 report 2026-09-22) but the published section-29 row->requirement->measurement mapping table is not produced; rows without measured slices stay out of comparison copy. |
| V5-29.04 | planned | deferred | Deferred — the first executed comparison exists; the after-each-comparison loop it triggers (next mechanism change must name a measured loss on a named slice, with an ablation proving causation) is forward-looking and owed by the P7 improvement program. |
| V5-30.22 | planned | deferred | Deferred — T2/T3 proposition-emitting extraction tiers are not implemented (verbatim/enrichment/grounding.py: deterministic T1 only), so the span-pinned emission contract cannot be measured; the grounding-failure half IS measured (e82 fabricated proposition fails grounding, green). |
| V5-30.23 | planned | deferred | Deferred — the G5-13 floors are adjudication targets: delivered-grounding checked 25 hits with 0 violations, but n=25 gives an upper-95% bound far above the 0.01 floor; a larger adjudicated slice is owed. |
| V5-30.24 | planned | deferred | Deferred — no per-tier extraction cost run was produced; the comparator cost schema reports unmeasured categories (extraction, embeddings, consolidation, answer_reading, maintenance) honestly rather than estimating. |
| V5-31.04 | planned | deferred | Deferred — per-signal A2 ablation runs were not executed; the default ranking/v1 signal table ships unablated and the two-point regression rule has no run to apply to. |
| V5-32.06 | planned | deferred | Deferred — LongMemEval/LoCoMo-class category reporting needs licensed corpora absent from this environment; owned natural histories exist at small scale and BEAM-scale (10M) stays inherited and unclaimed. |
| V5-33.11 | planned | deferred | Deferred — no CI gate runs the wide-threshold perf smoke; stage timers exist and are exercised (e95 per-stage timers green) but nothing gates merges on them. |
| V5-33.12 | planned | deferred | Deferred — the exact-oracle differential exists (tests/retrieval/v3/test_source_lane.py::test_lexical_bm25_matches_naive_reference) but the per-optimization-PR differential+profile gate is not established. |

Phase-level anchors are the planning map (V6-00.06);
qualification requires named assertions per requirement,
recorded as `executed_evidence` in `dispositions_v6.json`.
