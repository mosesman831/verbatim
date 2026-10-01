# I6 — utility-budgeted refresh vs periodic full reflection (D11/D12)

- corpus: 12 slots × 2 claims, 6 periods × 3 corrections
- freshness floor matched (0 stale answers, all changed regions covered): **True**
- claims processed: periodic **288** vs budgeted **50** → 82.6% less
- bytes processed: 1740 → 297 (82.9% less)
- jobs: 6 → 9 (-50.0% less)
- vs periodic consolidate-only (no reflection charge): 65.3% less claims
- D12 owner probe: scheduled 1, deferred 0, counted 1 priority job(s)
- met: **True**

| period | changed | budgeted jobs | claims | stale |
| --- | --- | --- | --- | --- |
| 0 | 0 | 4 (priority 0) | 24 | 0 |
| 1 | 3 | 1 (priority 0) | 6 | 0 |
| 2 | 3 | 1 (priority 0) | 6 | 0 |
| 3 | 3 | 1 (priority 0) | 6 | 0 |
| 4 | 3 | 1 (priority 1) | 2 | 3 |
| 5 | 0 | 1 (priority 0) | 6 | 0 |
