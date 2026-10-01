# Workload envelope S1 — SPEC_V4 §56

Generated: 2026-09-21T05:07:06Z  |  seed 42  |  duration 1153.6s

**Verdict: MISSED** (all declared targets AND zero query/capture failures required)

## Volumes (declared → observed)

| volume | spec | observed |
| --- | ---: | ---: |
| claims_total | 1000 | 1100 |
| claims_active | 1000 | 900 |
| spans | 5000 | 5000 |
| episodes | 100 | 100 |
| procedures | 20 | 20 |
| claim_revisions | — | 1200 |
| sources | — | 3900 |
| embeddings | — | 1000 |
| superseded (historical) | ≥10% seeded | 100 |
| held (quarantine) | — | 50 |

## Recall latency (ms) — measured samples only

- clients: 4, measured: 10000, failures: 0
- p50 427.18 / p95 713.30 / p99 820.79 / max 1048.57 / mean 408.73

## Capture + readiness

- captures: 2079 (declared 2.0/s, measured 2.00/s), failures: 0
- capture ack p50 57.01 / p95 129.75 / p99 176.84 / max 281.81
- lexical_ready states: {'succeeded': 411} — p95 883.13ms
- semantic_ready states: {'succeeded': 411} — p95 1052.03ms
- readiness deadlines exceeded: 0

## Load discipline

- measured queries: 10000 (warmup 100 excluded; 2 restart repetitions)
- per-repetition: [{'repetition': 0, 'measured': 5000, 'failures': 0, 'p50_ms': 372.730369, 'p95_ms': 565.915278, 'p99_ms': 640.20064, 'wall_s': 443.91598983097356}, {'repetition': 1, 'measured': 5000, 'failures': 0, 'p50_ms': 500.726818, 'p95_ms': 764.66026, 'p99_ms': 850.232249, 'wall_s': 595.4691831410164}]
- query mix realized: {'cross_scope': 1421, 'exact': 1450, 'multi_entity': 1432, 'no_answer': 1426, 'paraphrase': 1424, 'procedural': 1422, 'temporal': 1425}
- drain totals: {'processed': 2055, 'succeeded': 2055, 'failed': 0, 'deferred': 0, 'expired': 0}
- backlog at end: {'jobs_pending': 0, 'obligations_pending': 11568, 'receipts_unsettled': 0}

## Targets

| target | bound ms | measured ms | met |
| --- | ---: | ---: | --- |
| recall.p95_ms | 25.00 | 713.30 | NO |
| recall.p99_ms | 75.00 | 820.79 | NO |
| capture.ack_p95_ms | 50.00 | 129.75 | NO |
| capture.ack_p95_ms_ingest | 100.00 | 129.75 | NO |
| readiness.lexical_p95_ms | 2000.00 | 883.13 | yes |
| readiness.semantic_p95_ms | 10000.00 | 1052.03 | yes |

## Environment

```
{
  "cpu_count": 4,
  "db_bytes": 67547136,
  "loadavg_end": [
    4.565,
    4.574,
    4.52
  ],
  "loadavg_start": [
    3.838,
    4.108,
    4.5
  ],
  "peak_rss_kb": 166328,
  "platform": "Linux-6.17.0-1020-oracle-aarch64-with-glibc2.39",
  "python": "3.11.15",
  "sqlite": "3.53.1",
  "wal_bytes": 0
}
```

## Notes

- active claims 900 below spec volume 1000 — admissions that rejected are reported, not forced
- measured captures: 20% claim-bearing user_messages (~128B, ~1 claim each, full pipeline, lexical readiness measured) + 80% agent_notes (4KiB, write-path ack load, no claim derivation — envelope volumes stay at spec); all acks are real accepted-input timings
- Sustained load: 4 recall clients, 2 captures/s, 1 drain; deterministic hashing encoder provisions the semantic lane.
