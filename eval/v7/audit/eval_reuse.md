# V7 Evaluation Reuse Audit (w-aud-eval, archived)

## Ledger: V7 inherits the V6 two-axis model
`implementation_status × qualification_status → derived status`; named executed evidence required for qualified/locally_measured. V7 deltas: parse `V7-NN.MM`, H01–H104, G7-00..14, P0..P10(+P2b), SB rows, O-decisions; add `datasets`/`sb_rows`/`o_decisions`/`formula_tags` fields; carry V6 tail read-only. `deferred` valid only with owner+reason. Landed: `eval/v7/ledger.py`+`ledger_v7.json`+`manifest.py`+`gates.py` (349 reqs registered, all planned/not_run).

## Corpus format
V3 (JSONL fixtures + gold-view proxy) fused with V5 (consumer route, lifecycle through real add/forget, corpus digest, gold-field AttributeError). Landed `eval/v7/corpora.py`: CorpusItem(speaker/session_id/when/image_caption/group_id), CorpusTask(evidence_ids + evidence_session_ids dual granularity, answerable, group_id, metadata answer/query_time/unresolved). Loaders: locomo_json (O1-gated), longmemeval_json (O2), twin lazy-import via DatasetUnavailable(status-verbatim). **Adapter gap fixed in main session:** locomo-like `{turns,questions}` + lme-like `{questions:[{sessions:[turns]}]}` shapes normalized to items/tasks (corpora.py `_locomo_twin_items_tasks`/`_lme_twin_items_tasks`).

## Arms
V5 `ComparatorArm` protocol: name/pin/seed/answer/close (+replace/delete/storage_bytes/internal_lifecycle). V7 pin = 15 fields (V6's 13 + reader_prompt_digest + budget_preset). New required arms: hindsight_oss, letta_memfs, full_context, flat_bm25 (pure-Python Okapi reference), bm25_dense_rrf. Landed `eval/v7/arms.py`: VerbatimArm (real Memory, _MAX_LIMIT=64, lanes_disabled plumbing), FlatBM25Arm (k1=1.2 b=0.75), FTS5Arm.

## Metrics/attribution (§33)
any@k/all@k dual granularity, official recall_any/all, NDCG@10 binary, MRR; refusal metrics (zero-result, false abstain, correct refusal); delivered tokens on reader view (tok/v1, median+p95); per-question attribution classes delivered|not_indexed|not_retrieved|retrieved_below_cut|packed_out|reader_wrong|judge_disputed|abstained_wrongly → landed as delivered|lane_miss|rank_shift|packed_out|abstain|unsupported|unattributed in `eval/v7/attribution.py`.

## Envelopes/stats
B0–B6; stage record fields per §32.17 (snapshots/bytes_read/candidates/pool_R/coverage_digest new); α measurement ≥4 checkpoints, stage α>1.0 named defect; cluster bootstrap ≥10,000 resamples (V5 default 2000 must be raised); McNemar exact; Bonferroni across SB family; "beats" requires lower paired bound > 0.

## Mutation suite
eval/v7/mutations.yaml — all 23 §35 targets as `pending:` entries (v4 loader requires concrete anchor/replacement/killed_by — null anchors crash; promotion = move to `mutations:` + fill fields). ≥12 new V7 lane/index mutants owed for G7-00. Anchor-must-match-live-path lesson (MUT-10) is normative.

## Report/manifest
`not_run` for absent inputs, never omitted; SB rows rendered always; manifest adds dataset digests+split, tiers, model hashes, lane-policy digest, reader+judge ids+prompt digests, start/end. Landed `eval/v7/manifest.py` (formula=unselected until O9) + `eval/v7/gates.py` (GATE_INPUTS transcribed from §26, artifacts index absent → all not_run).

## First live numbers (w-trackr probe, owned twins)
owned_actions: flat_bm25 any@10 0.964 / all@10 0.744; fts5 0.946/0.674. owned_prefs: verbatim any@10 **0.274** vs flat_bm25 **0.500** — the measured gap (attribution: 123 abstain on preference intents, 100 unattributed suppressed tails). Search p50 17.8ms/p95 36.4ms at 1355 items.
