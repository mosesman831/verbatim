## Track R — owned_locomo_like (split=full, seed=0)

schema `track_r/v7-a` · constants `provisional/v7-r0` · manifest `unpinned`

items=360 tasks=120 answerable=96 · k=[10, 20] · corpus digest `31bbc1e0a2ac8b8512ce24d8c3db0a36d554b9f64840980dfa395c815abe2ab6`

### Evidence recall — item granularity

| arm | cat | n | n_gold | mean\|G\| | any@10 | any@20 | all@10 | all@20 | prop@10 | prop@20 | ndcg@10 | mrr@10 | zero | abstain | p50ms | p95ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **flat_bm25** | all | 120 | 96 | 1.033 | 0.760 | 0.771 | 0.656 | 0.688 | 0.708 | 0.729 | 0.650 | 0.671 | 0.000 | 0.000 | 0.5 | 0.7 |
| flat_bm25 | abstain | 24 | 0 | 0.000 | — | — | — | — | — | — | — | — | — | 0.000 | 0.5 | 0.6 |
| flat_bm25 | 1 multi_hop | 24 | 24 | 2.000 | 0.875 | 0.875 | 0.458 | 0.542 | 0.667 | 0.708 | 0.597 | 0.732 | 0.000 | 0.000 | 0.7 | 0.8 |
| flat_bm25 | 3 open_domain | 24 | 24 | 1.000 | 0.250 | 0.292 | 0.250 | 0.292 | 0.250 | 0.292 | 0.214 | 0.201 | 0.000 | 0.000 | 0.5 | 0.7 |
| flat_bm25 | 4 single_hop | 24 | 24 | 1.000 | 0.917 | 0.917 | 0.917 | 0.917 | 0.917 | 0.917 | 0.834 | 0.806 | 0.000 | 0.000 | 0.5 | 0.6 |
| flat_bm25 | 2 temporal | 24 | 24 | 1.167 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.954 | 0.944 | 0.000 | 0.000 | 0.6 | 0.8 |

### Evidence recall — session granularity

| arm | cat | n | n_gold | mean\|G\| | any@10 | any@20 | all@10 | all@20 | prop@10 | prop@20 | ndcg@10 | mrr@10 | zero | abstain | p50ms | p95ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **flat_bm25** | all | 96 | 96 | 1.271 | 0.917 | 0.958 | 0.854 | 0.938 | 0.885 | 0.948 | 0.995 | 0.751 | 0.000 | 0.000 | 0.5 | 0.8 |
| flat_bm25 | 1 multi_hop | 24 | 24 | 1.958 | 1.000 | 1.000 | 0.750 | 0.917 | 0.875 | 0.958 | 0.966 | 0.817 | 0.000 | 0.000 | 0.7 | 0.8 |
| flat_bm25 | 3 open_domain | 24 | 24 | 1.000 | 0.708 | 0.833 | 0.708 | 0.833 | 0.708 | 0.833 | 0.579 | 0.390 | 0.000 | 0.000 | 0.5 | 0.7 |
| flat_bm25 | 4 single_hop | 24 | 24 | 1.000 | 0.958 | 1.000 | 0.958 | 1.000 | 0.958 | 1.000 | 1.134 | 0.819 | 0.000 | 0.000 | 0.5 | 0.6 |
| flat_bm25 | 2 temporal | 24 | 24 | 1.125 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.303 | 0.979 | 0.000 | 0.000 | 0.6 | 0.8 |

### Attribution (per-question, item granularity)

| arm | delivered | lane_miss | rank_shift | packed_out | abstain | unsupported | unattributed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat_bm25 | 74 | 0 | 22 | 0 | 0 | 24 | 0 |

