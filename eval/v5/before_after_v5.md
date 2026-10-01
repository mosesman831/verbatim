# V5 before/after — optimization batch

Generated 2026-09-22T08:32:50Z — qualification: **locally_measured**

Baseline = quiet-tree run of `python -m eval.v5.run` executed before the optimization batch (settlement retries, WAL-snapshot prescan, commit-wake signaling, batched writes, deferred DDL, HMAC template, pending_count, cv triggers, checkpoint pacing). Same entry point, same scales, same seed=42.

| metric | stat | before (ms or count) | after | delta |
|---|---|---|---|---|
| a0.search_ms | p50 | 21.7 | 17.779929970856756 | -3.92 (-18.1%) |
| a0.search_ms | p95 | 24.9 | 26.039956021122634 | 1.14 (4.6%) |
| a0.search_ms | p99 | 41.8 | 39.90749199874699 | -1.893 (-4.5%) |
| a0.add_ack_ms | p50 | 11.5 | 9.85486397985369 | -1.645 (-14.3%) |
| a0.add_ack_ms | p95 | 31.6 | 15.86661400506273 | -15.733 (-49.8%) |
| a0.add_ack_ms | p99 | 45.4 | 20.402485970407724 | -24.998 (-55.1%) |
| a1.search_ms | p50 | 240.6 | 28.9359109592624 | -211.664 (-88.0%) |
| a1.search_ms | p95 | 282.9 | 119.6287180064246 | -163.271 (-57.7%) |
| a1.search_ms | p99 | 287.3 | 194.27407102193683 | -93.026 (-32.4%) |
| a1.add_ack_ms | p50 | 9.8 | 9.985543030779809 | 0.186 (1.9%) |
| a1.add_ack_ms | p95 | 23.9 | 23.833878978621215 | -0.066 (-0.3%) |
| a1.add_ack_ms | p99 | 45.7 | 41.164689988363534 | -4.535 (-9.9%) |
| a3_768.search_ms | p50 | 41.2 | 32.1609050151892 | -9.039 (-21.9%) |
| a3_768.search_ms | p95 | 73.6 | 43.09908702271059 | -30.501 (-41.4%) |
| a3_768.search_ms | p99 | 92.7 | 80.20378299988806 | -12.496 (-13.5%) |
| a3_768.add_ack_ms | p50 | 15.6 | 13.300498016178608 | -2.3 (-14.7%) |
| a3_768.add_ack_ms | p95 | 75.0 | 70.86139998864383 | -4.139 (-5.5%) |
| a3_768.add_ack_ms | p99 | 86.7 | 83.12961901538074 | -3.57 (-4.1%) |
| timers.total_ms | p50 | 214.4 | 212.71951700327918 | -1.68 (-0.8%) |
| timers.total_ms | p95 | 215.2 | 213.61471596173942 | -1.585 (-0.7%) |
| timers.total_ms | p99 | 215.4 | 232.47828398598358 | 17.078 (7.9%) |
| timers.settled_search_ms | p50 | 8.4 | 12.070099997799844 | 3.67 (43.7%) |
| timers.settled_search_ms | p95 | 9.1 | 13.185056974180043 | 4.085 (44.9%) |
| timers.settled_search_ms | p99 | 10.5 | 19.53728700755164 | 9.037 (86.1%) |
| timers.immediate_add_search_ms | p50 | 221.2 | 221.79634199710563 | 0.596 (0.3%) |
| timers.immediate_add_search_ms | p95 | 221.4 | 221.88090201234445 | 0.481 (0.2%) |
| timers.immediate_add_search_ms | p99 | 221.4 | 221.88090201234445 | 0.481 (0.2%) |
| timers.sql_per_search | p50 | 123.0 | 159.0 | 36.0 (29.3%) |
| timers.sql_per_search | p95 | 140.0 | 167.0 | 27.0 (19.3%) |
| timers.sql_per_search | p99 | 142.0 | 169.0 | 27.0 (19.0%) |

## Caveats

- Latency deltas on a shared 4-core box carry noise; treat <±15% as parity unless repeated runs disagree.
- Correctness gates (quality recall/precision, leakage, dx) must hold in BOTH columns — speed without correctness is not a win (SPEC_V5 §33.08).
