## Track R — owned_locomo_like (split=full, seed=0)

schema `track_r/v7-a` · constants `provisional/v7-r0` · manifest `unpinned`

items=360 tasks=120 answerable=96 · k=[10, 20] · corpus digest `31bbc1e0a2ac8b8512ce24d8c3db0a36d554b9f64840980dfa395c815abe2ab6`

### Evidence recall — item granularity

| arm | cat | n | n_gold | mean\|G\| | any@10 | any@20 | all@10 | all@20 | prop@10 | prop@20 | ndcg@10 | mrr@10 | zero | abstain | p50ms | p95ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **verbatim** | all | 120 | 96 | 1.033 | 0.729 | 0.802 | 0.635 | 0.708 | 0.682 | 0.755 | 0.613 | 0.626 | 0.000 | 0.008 | 196.8 | 367.0 |
| verbatim | abstain | 24 | 0 | 0.000 | — | — | — | — | — | — | — | — | — | 0.042 | 205.0 | 367.0 |
| verbatim | 1 multi_hop | 24 | 24 | 2.000 | 0.875 | 0.875 | 0.500 | 0.500 | 0.688 | 0.688 | 0.651 | 0.758 | 0.000 | 0.000 | 192.8 | 375.3 |
| verbatim | 3 open_domain | 24 | 24 | 1.000 | 0.167 | 0.458 | 0.167 | 0.458 | 0.167 | 0.458 | 0.115 | 0.100 | 0.000 | 0.000 | 188.1 | 331.6 |
| verbatim | 4 single_hop | 24 | 24 | 1.000 | 0.917 | 0.917 | 0.917 | 0.917 | 0.917 | 0.917 | 0.872 | 0.858 | 0.000 | 0.000 | 190.9 | 324.3 |
| verbatim | 2 temporal | 24 | 24 | 1.167 | 0.958 | 0.958 | 0.958 | 0.958 | 0.958 | 0.958 | 0.812 | 0.790 | 0.000 | 0.000 | 216.7 | 366.0 |

### Evidence recall — session granularity

| arm | cat | n | n_gold | mean\|G\| | any@10 | any@20 | all@10 | all@20 | prop@10 | prop@20 | ndcg@10 | mrr@10 | zero | abstain | p50ms | p95ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **verbatim** | all | 96 | 96 | 1.271 | 0.938 | 1.000 | 0.875 | 0.990 | 0.906 | 0.995 | 0.969 | 0.721 | 0.000 | 0.000 | 194.5 | 366.0 |
| verbatim | 1 multi_hop | 24 | 24 | 1.958 | 1.000 | 1.000 | 0.750 | 0.958 | 0.875 | 0.979 | 1.055 | 0.873 | 0.000 | 0.000 | 192.8 | 375.3 |
| verbatim | 3 open_domain | 24 | 24 | 1.000 | 0.792 | 1.000 | 0.792 | 1.000 | 0.792 | 1.000 | 0.514 | 0.299 | 0.000 | 0.000 | 188.1 | 331.6 |
| verbatim | 4 single_hop | 24 | 24 | 1.000 | 0.958 | 1.000 | 0.958 | 1.000 | 0.958 | 1.000 | 1.088 | 0.868 | 0.000 | 0.000 | 190.9 | 324.3 |
| verbatim | 2 temporal | 24 | 24 | 1.125 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.218 | 0.844 | 0.000 | 0.000 | 216.7 | 366.0 |

### Attribution (per-question, item granularity)

| arm | delivered | lane_miss | rank_shift | packed_out | abstain | unsupported | unattributed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| verbatim | 77 | 15 | 4 | 0 | 0 | 24 | 0 |

