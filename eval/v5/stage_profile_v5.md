# V5 stage profile — consumer route

Generated 2026-09-22T08:31:55Z — qualification: **locally_measured**. Scale: 256 memories, 48 profiled searches, seed=42.

Method: monkeypatch-wrapped real stage functions + sqlite trace callback; no production changes

## worker=managed

### per-stage attribution (settled searches)

| stage | n | mean ms | p50 | p95 |
|---|---|---|---|---|
| total | 48 | 41.688 | 27.219 | 169.776 |
| facade_other | 48 | 29.351 | — | — |
| source_lane | 48 | 10.381 | 10.215 | 16.306 |
| lanes | 48 | 0.627 | 0.465 | 1.605 |
| source_fuse | 48 | 0.447 | 0.448 | 0.476 |
| authorize | 48 | 0.367 | 0.26 | 1.225 |
| analyze | 48 | 0.143 | 0.144 | 0.176 |
| controller.discretize | 48 | 0.112 | 0.039 | 0.112 |
| journal | 48 | 0.111 | 0.108 | 0.183 |
| controller.plan_routes | 48 | 0.062 | 0.057 | 0.109 |
| groups | 48 | 0.028 | 0.025 | 0.057 |
| abstain | 48 | 0.027 | 0.025 | 0.032 |
| union | 48 | 0.017 | 0.017 | 0.021 |
| fusion | 48 | 0.01 | 0.01 | 0.011 |
| classify | 48 | 0.005 | 0.005 | 0.007 |

### boundary timings

| boundary | p50 | p95 | p99 |
|---|---|---|---|
| add_ack_ms | 14.33 | 54.03 | 54.03 |
| forget_ack_ms | 10.34 | 15.49 | 15.49 |
| immediate_add_search_ms | 207.92 | 243.47 | 243.47 |
| readiness_wait_ms | 16.14 | 31.18 | 31.18 |
| settled_search_ms | 21.81 | 108.09 | 113.27 |
| source_visibility_lag_ms | 15.96 | 51.92 | 51.92 |

## worker=external

### per-stage attribution (settled searches)

| stage | n | mean ms | p50 | p95 |
|---|---|---|---|---|
| total | 48 | 219.286 | 218.825 | 221.592 |
| facade_other | 48 | 210.34 | — | — |
| source_lane | 48 | 7.443 | 7.728 | 9.698 |
| source_fuse | 48 | 0.434 | 0.429 | 0.486 |
| lanes | 48 | 0.419 | 0.415 | 0.465 |
| authorize | 48 | 0.23 | 0.221 | 0.269 |
| analyze | 48 | 0.145 | 0.148 | 0.183 |
| journal | 48 | 0.084 | 0.079 | 0.118 |
| controller.plan_routes | 48 | 0.066 | 0.063 | 0.098 |
| controller.discretize | 48 | 0.042 | 0.042 | 0.045 |
| abstain | 48 | 0.025 | 0.025 | 0.029 |
| groups | 48 | 0.025 | 0.024 | 0.026 |
| union | 48 | 0.016 | 0.016 | 0.017 |
| fusion | 48 | 0.01 | 0.01 | 0.011 |
| classify | 48 | 0.007 | 0.005 | 0.006 |

### boundary timings

| boundary | p50 | p95 | p99 |
|---|---|---|---|
| add_ack_ms | 13.82 | 16.38 | 16.38 |
| forget_ack_ms | 13.36 | 24.78 | 24.78 |
| immediate_add_search_ms | 230.04 | 230.83 | 230.83 |
| readiness_wait_ms | 3000.24 | 3000.3 | 3000.3 |
| settled_search_ms | 18.61 | 21.25 | 57.58 |
| source_visibility_lag_ms | 15.39 | 41.02 | 41.02 |

## Notes

- `facade_other` on `external` is the bounded causal-barrier wait on post-settle pending obligations (ready_timeout_ms) — expected on an undrained env; on `managed` it is the real residual facade cost.
- Denominators are published; stages with no samples report nothing rather than fabricating a zero.
