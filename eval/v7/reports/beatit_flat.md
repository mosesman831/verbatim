## Track R — owned_locomo_like (split=full, seed=0)

schema `track_r/v7-a` · constants `provisional/v7-r0` · manifest `unpinned`

items=360 tasks=120 answerable=96 · k=[10, 20] · corpus digest `31bbc1e0a2ac8b8512ce24d8c3db0a36d554b9f64840980dfa395c815abe2ab6`

### Evidence recall — item granularity

| arm | cat | n | n_gold | mean\|G\| | any@10 | any@20 | all@10 | all@20 | prop@10 | prop@20 | ndcg@10 | mrr@10 | zero | abstain | p50ms | p95ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **verbatim** | all | 120 | 96 | 1.033 | 0.719 | 0.771 | 0.625 | 0.677 | 0.672 | 0.724 | 0.586 | 0.596 | 0.000 | 0.108 | 548.7 | 713.5 |
| verbatim | abstain | 24 | 0 | 0.000 | — | — | — | — | — | — | — | — | — | 0.167 | 546.8 | 677.5 |
| verbatim | 1 multi_hop | 24 | 24 | 2.000 | 0.833 | 0.875 | 0.458 | 0.500 | 0.646 | 0.688 | 0.541 | 0.651 | 0.000 | 0.000 | 520.1 | 1294.6 |
| verbatim | 3 open_domain | 24 | 24 | 1.000 | 0.167 | 0.292 | 0.167 | 0.292 | 0.167 | 0.292 | 0.098 | 0.077 | 0.000 | 0.083 | 540.7 | 667.7 |
| verbatim | 4 single_hop | 24 | 24 | 1.000 | 0.875 | 0.917 | 0.875 | 0.917 | 0.875 | 0.917 | 0.801 | 0.780 | 0.000 | 0.208 | 552.5 | 566.4 |
| verbatim | 2 temporal | 24 | 24 | 1.167 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.904 | 0.878 | 0.000 | 0.083 | 566.4 | 698.9 |

### Evidence recall — session granularity

| arm | cat | n | n_gold | mean\|G\| | any@10 | any@20 | all@10 | all@20 | prop@10 | prop@20 | ndcg@10 | mrr@10 | zero | abstain | p50ms | p95ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **verbatim** | all | 96 | 96 | 1.271 | 0.854 | 0.948 | 0.812 | 0.896 | 0.833 | 0.922 | 0.880 | 0.657 | 0.000 | 0.094 | 550.0 | 1200.0 |
| verbatim | 1 multi_hop | 24 | 24 | 1.958 | 0.917 | 1.000 | 0.750 | 0.792 | 0.833 | 0.896 | 0.907 | 0.729 | 0.000 | 0.000 | 520.1 | 1294.6 |
| verbatim | 3 open_domain | 24 | 24 | 1.000 | 0.542 | 0.792 | 0.542 | 0.792 | 0.542 | 0.792 | 0.441 | 0.223 | 0.000 | 0.083 | 540.7 | 667.7 |
| verbatim | 4 single_hop | 24 | 24 | 1.000 | 0.958 | 1.000 | 0.958 | 1.000 | 0.958 | 1.000 | 0.992 | 0.799 | 0.000 | 0.208 | 552.5 | 566.4 |
| verbatim | 2 temporal | 24 | 24 | 1.125 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.179 | 0.878 | 0.000 | 0.083 | 566.4 | 698.9 |

### Attribution (per-question, item granularity)

| arm | delivered | lane_miss | rank_shift | packed_out | abstain | unsupported | unattributed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| verbatim | 74 | 10 | 10 | 0 | 2 | 24 | 0 |

