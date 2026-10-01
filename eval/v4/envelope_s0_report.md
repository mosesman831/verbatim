# Workload envelope S0 — SPEC_V4 §56

Generated: 2026-09-21T04:47:38Z  |  seed 42  |  duration 52.5s

**Verdict: MISSED** (all declared targets AND zero query/capture failures required)

## Volumes (declared → observed)

| volume | spec | observed |
| --- | ---: | ---: |
| claims_total | 100 | 100 |
| claims_active | 100 | 100 |
| spans | 500 | 500 |
| episodes | 10 | 10 |
| procedures | 0 | 0 |
| claim_revisions | — | 100 |
| sources | — | 400 |
| embeddings | — | 0 |
| superseded (historical) | ≥0% seeded | 0 |
| held (quarantine) | — | 0 |

## Recall latency (ms) — measured samples only

- clients: 1, measured: 1000, failures: 0
- p50 43.65 / p95 79.19 / p99 97.32 / max 163.06 / mean 43.51

## Capture + readiness

- captures: 88 (declared 2.0/s, measured 2.01/s), failures: 0
- capture ack p50 12.85 / p95 23.88 / p99 33.81 / max 33.81
- lexical_ready states: {'succeeded': 17} — p95 204.64ms
- semantic_ready states: {'deferred': 17} — p95 210.79ms
- readiness deadlines exceeded: 0

## Load discipline

- measured queries: 1000 (warmup 100 excluded; 2 restart repetitions)
- per-repetition: [{'repetition': 0, 'measured': 500, 'failures': 0, 'p50_ms': 39.604996, 'p95_ms': 68.654933, 'p99_ms': 92.788202, 'wall_s': 20.007929492974654}, {'repetition': 1, 'measured': 500, 'failures': 0, 'p50_ms': 48.519417, 'p95_ms': 83.964781, 'p99_ms': 97.809151, 'wall_s': 23.722428307984956}]
- query mix realized: {'cross_scope': 140, 'exact': 144, 'multi_entity': 144, 'no_answer': 144, 'paraphrase': 144, 'procedural': 140, 'temporal': 144}
- drain totals: {'processed': 51, 'succeeded': 51, 'failed': 0, 'deferred': 0, 'expired': 0}
- backlog at end: {'jobs_pending': 0, 'obligations_pending': 780, 'receipts_unsettled': 0}

## Targets

| target | bound ms | measured ms | met |
| --- | ---: | ---: | --- |
| recall.p95_ms | 25.00 | 79.19 | NO |
| recall.p99_ms | 75.00 | 97.32 | NO |
| capture.ack_p95_ms | 50.00 | 23.88 | yes |

## Environment

```
{
  "cpu_count": 4,
  "db_bytes": 7274496,
  "loadavg_end": [
    4.377,
    4.218,
    4.541
  ],
  "loadavg_start": [
    4.292,
    4.158,
    4.542
  ],
  "peak_rss_kb": 56528,
  "platform": "Linux-6.17.0-1020-oracle-aarch64-with-glibc2.39",
  "python": "3.11.15",
  "sqlite": "3.53.1",
  "wal_bytes": 0
}
```

## Notes

- measured captures: 20% claim-bearing user_messages (~128B, ~1 claim each, full pipeline, lexical readiness measured) + 80% agent_notes (4KiB, write-path ack load, no claim derivation — envelope volumes stay at spec); all acks are real accepted-input timings
- M0 no-model envelope: rules recall only, embeddings unprovisioned.
