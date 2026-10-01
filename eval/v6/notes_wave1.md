# V6 eval — wave-1 measurement notes (2026-09-22)

Scope: execute the V6 measurement portfolio on a quiet tree and register
honest evidence. No new features. Artifacts: `eval/v6/report_v6.json` /
`.md` / `.repro.json`, `eval/v6/dispositions_v6.json`,
`eval/v6/ledger_v6.json`, `eval/v6/summary.md`, plus the v5 suite's own
`eval/v5/report_v5.*` from the consolidation+feedback run.

## Quiet-tree note (V6-00.07)

Two portfolio executions happened:

1. **14:57 run — DISCARDED.** Mid-run churn detected: a concurrent
   worker rewrote `verbatim/memory/facade.py` at 14:58:39 and
   `tests/v6/test_scenarios_f.py` at 15:01 (817→1464 lines), then ran
   `pytest tests/v6 -x` 15:01–15:22. The run's imported facade predated
   the rewrite and a sibling pytest competed for the box — the numbers
   are not comparable to anything and the artifact was overwritten.
2. **15:22 run — the published artifact.** Same command
   (`python -m eval.v6.run`), quiet tree verified (no concurrent
   pytest/eval/file churn during the run; only ambient hermes daemons
   on the box). `report_v6.*` is this run.

## What was measured

`python -m eval.v6.run` — 6 suites, verdict **inconclusive** (honest:
envelope misses mark the run):

| suite | verdict | headline |
|---|---|---|
| comparators | executed | verbatim_memory acc 1.000 / recall@k 1.000; verbatim_v2 0.852; naive_fts 0.889; vector_rag 0.778; no_memory 0.148. mem0_oss(+inferfalse) unavailable (mem0ai absent); graphiti_oss unavailable; holographic unavailable; zep_hosted + mem0_platform out_of_scope (hosted, unauthorized) |
| envelopes | **missed** | every measured envelope missed its latency budget on this machine — details below |
| add_ack_scaling | executed | n=512→1024, all stage alphas < 1 (max 0.845 session_receipt_tx); no superlinear stage at this span |
| twins | **passed** | auto_applied 5/5 true, false_applied 0/13, upper95_bound 0.2281 |
| neural | executed | paired gate: recall@k artifact 1.0000 vs hashing 1.0000 — not strictly better → neural_recommended **False** (published) |
| v5_tail | executed | unresolved 45/369 (42 planned + 3 implemented_unmeasured) |

### Envelopes detail (quick-but-real: A0/A0-cache/A1 at 512 memories)

| envelope | search p50/p95/p99 ms | budget p95(/p99) | verdict |
|---|---|---|---|
| A0 (cache off) | 86.3 / 125.7 / 143.6 | 25 / 75 | **MISS** |
| A0-cache (warm) | 71.5 / 104.6 / 110.9 | 8 | **MISS** + pack cache never hit (lookups=120, hits=0, stores=0 — real finding) |
| A0-neural | — | — | **unavailable** — `Memory(encoder="artifact")` constructs but `available()` is False without a provisioned artifact (V6-03.08 fail-closed, honest) |
| A1 (managed, 40 reads/16 writes) | 298.4 / 462.1 / 539.6 | 40 | **MISS** — all 40 reads returned honest `pending` (barrier budget exhausted under live drain); add-ack p95 310.1ms > 50ms **MISS**; backlog bounded (peak 536 → end 0, not unbounded); same-session visibility 6/6 satisfied but observed at ~250ms vs the 200ms budget (published, not smoothed) |
| A3 (2048 records, checkpoints 512/1024/2048) | 298.3 / 417.8 / 432.7 | 150 | **MISS** — plus named V6-02.16 defect: add-ack stage `source_state` superlinear, alpha≈1.15 (0.89ms→4.38ms over n=512→2048). Index growth alpha 0.797 (sublinear, meets), peak RSS 238 MiB (≤1536, meets), add-ack p95 41.7ms (≤80, meets) |

**A1 vs the V5 miss (honest delta):** the V5 published miss was a ~120ms
settled-profile p95 at n=256 (stage_profile_v5.md: managed settled p95
108.09ms). This wave's A1 measures 462.1ms p95 at n=512 under live
2w/10r load with a draining backlog — a *larger* miss under heavier
conditions, not a pass. The cross-store commit_notify + unblock-first
drain changes are exercised (F17/F18 green; visibility probes observed
6/6 writes) but the envelope budget is missed by ~11× on this loaded
4-core aarch64 box. The fix did not produce a measured A1 pass — that
is the data.

**A0 immediate_add_search:** p50 ≈295ms, all 4 probes `pending` —
the causal-barrier cost is real and published beside settled search
(V6-02.12).

`python -m eval.v5.run --suite consolidation --suite feedback`:

- consolidation: **passed** — observations=2 produced (positive outcome,
  V6-03.11 fixture works), pins resolve 5/5 byte-traceable,
  sole_trace=False, corroboration passed on folded hit.corroboration=2,
  grounding 25 hits / 0 violations.
- feedback: **inconclusive** — exposure_rows_recorded passed
  (source_exposure rows minted on consumer path), kill_switch passed
  (learned_active refuses CONFIG_INVALID naming the binding),
  shadow_logged_deterministic_applied passed; `feedback_bounds`
  inconclusive — a real delivered-handle feedback row could not be
  exercised (influence_rows=0 on this path; unknown-handle refusal
  verified).

`pytest` evidence run (quiet tree): 203 tests across tests/v6 (41),
tests/eval/test_v6_* (ledger/envelopes/neural/comparators),
tests/adapters/test_langchain.py, tests/test_provider_v4.py,
tests/storage/test_commit_notify.py, tests/connectors/test_service.py,
tests/retrieval/test_f4_stats.py, tests/policy/test_artifacts.py —
**202 pass, 1 fail**: `test_f31_v5_tail_re_dispositioned` fails honestly
because the V5 tail is still open (45 unresolved — spec bar, not a
regression).

`eval.v6.neural.fit_calibration` (V6-03.09): floor 0.85 fitted for
`artifact:hash-distilled:v1`, `separates=true` (reject envelope max
0.707) — the artifact's own calibration, not a copied hashing threshold.

## What remains missed (published, not suppressed)

- **A0 / A0-cache / A1 / A3 search latencies** — all miss their budgets
  on this machine (see table). A0 misses 25ms by ~5×, A1 misses 40ms
  by ~11×, A3 misses 150ms by ~2.8×.
- **A1 add-ack** p95 310.1ms > 50ms under live drain contention.
- **Pack cache** — consulted (120 lookups) but 0 hits/stores on repeated
  intents: the fingerprint never restamps to a hit. Flag stays
  default-off; V6-02.06's SHOULD-enable default is unmet.
- **A3 `source_state` stage** — superlinear alpha≈1.15, named V6-02.16
  defect.
- **Twins upper-95% bound 0.2281** vs the <0.01 gate — corpus is 13
  false twins; ~300 zero-failure twins are required before the bound
  can clear. Behavior is correct (0 false applies); the *bound* is
  honestly unmet.
- **Neural label** — artifact does not strictly improve recall
  (1.0000 vs 1.0000): `neural_recommended=False`, label stays
  unrecommended.
- **A0-neural envelope** — unavailable: no verified pinned artifact in
  the consumer route (the dev artifact exists for the paired gate; the
  facade slot needs a provisioned artifact in data_dir).
- **V5 tail** — 45/369 unresolved (V6-04.02/03 open;
  `test_f31_v5_tail_re_dispositioned` correctly fails until closed).

## What remains unmeasured / deferred

- **V6-01.06** distribution-name legal verification + clean-env tutorial
  resolution — declared name asserted, verification not exercised.
- **V6-01.13** clean-venv `pip install .` on aarch64+x86_64 — wheel
  builds; the install path is unmeasured.
- **V6-02.05** ANN — precondition now met (A0/A1 missed); no ANN work
  this wave.
- **V6-04.02/03** V5-tail re-dispositioning — measured (45 rows), not
  executed here (v5 dispositions file is a different owner).
- **V6-04.04** full predecessor regression (v3+v4+v4.5+compat+mutation)
  — this wave ran the v6-targeted subset only.
- **V6-04.05** E-scenario evidence-row updates / stale-xfail sweep.
- **Unmeasured categories** (carried in report): operator-time costs,
  hosted/priced comparator costs, deployment cost beyond local
  embedded store, competitor editions absent from this environment.

## Disposition delta

Before: 69 rows — 69 planned / 69 not_run.
After: 58 `locally_measured` (all with named executed evidence),
4 `not_applicable` (authority/registry text), 5 `planned` +
2 `implemented_unmeasured` (the 7 `not_run` rows listed above).
No row was promoted without an executed check; misses are recorded in
`report_v6.json` and in the notes above.
