# Cat-5 premise-delivery + verdict arms — measured on the real dev store

## Method

Store: `/tmp/v7w/mem.db` rebuilt by `eval.v7.track_r --dataset locomo --split dev --arms verbatim,flat_bm25` at HEAD `1c86f4f`, snapshotted post-ingest (4143 units, generation 1, scope `ns_acf07e38a93a4da4bf65c6412add423c`) to `/tmp/v7w_probe/mem.db`. Probes are per-question — no ingest re-runs — driving the real `run_search` inside `verbatim.Memory(..., encoder="hashing")` with a facade-identical `LaneContextV7` (generation, `make_eligible`, `build_query_view`, profile `default`, MID budget, manifest limit=64). Taps on `rerank_features.score_candidates`, `verdict_v2.classify_groups`, `verdict_v2.result_verdict` capture fused pool, per-item feature values, group details, and verdict notes without changing behavior. Delivered refs = first-10 distinct corpus refs of `pack.items`, mirroring the harness.

Baseline parity vs the official `report.json`: cat-5 any@10 0.404 = official 0.404; withhold-rate unanswerable 0.216 ≈ official abstain 0.211; attribution 532/102/87/2/54/213 ≈ official. Offline weight sweeps re-score captured feature vectors linearly (score′ = score + ΣΔw·f) and re-derive first-10 distinct refs; validated against a real ent_idf=0 pass — agreement within ±0.003 on every metric.

## Baseline (real pipeline, 990 dev tasks)

any@10: all 0.553 · answerable 0.593 · cat-5 0.404 · bm25 cat-5 0.577. Per-cat: multi_hop 0.492, open_domain 0.340, single_hop 0.611, temporal 0.691. Verdict: correct_refusal 0.216 (46/213: 45 trigger-d, 1 trigger-b), FPR 0.179 (139/777, ALL trigger-d). Trigger-(c) is unfitted/disabled; negative_evidence and abstain_likely never fire on this corpus — every d-fire is `premise_mismatch` alone. speaker_match engaged on only 5.8% of the scored pool (lane-signal gating confirmed); cat-5 gold premise units score sm=0.5 neutral (gold-sm mean 0.479).

## Arm 1 — u_speaker batched-lookup fix

Patch: `FeatureProviders.unit_speaker` populated from a single `units` table lookup for every scored candidate (exactly the fix used by verdict_v2's `_unit_speakers`, moved upstream into scoring).

- speaker_match engagement: 5.8% → **99.1%** of scored pool. q_speaker resolves on 99.1% of cat-5, 86.4% of answerable.
- Gold premise sm (cat-5): **78% land at 0.0** (penalized — the premise is the other speaker's turn), 21% at 1.0, ~1% neutral. Answerable gold: 82.7% at 1.0, 14.5% neutral, 2.8% penalized.
- Displacer engagement (cat-5 pack-10): 2821/2849 engaged, 120 penalized.
- Both-holds (displacers engaged AND ≥1 penalized): 27.2% vs the 0.50 bar — **fails** (baseline 2.8%).
- Measured effect at sm=0.4 (real pass): cat-5 any@10 **0.404 → 0.286**, answerable 0.593 → 0.600, verdict unchanged (empirically stable under weight-only re-scoring — the top-100 pool composition barely moves).

sm weight sweep (offline, validated): sm=0 → cat-5 0.413; sm=0.2 → 0.385; sm=0.4 → 0.291. Answerable rises with weight: 0.598 / 0.605 / 0.609. **speaker_match at any positive weight costs cat-5 premise delivery** because the corpus's cat-5 gold is ~72–78% other-speaker turns; at sm=0 the feature is dead weight (costs nothing, gains nothing).

## Arm 2 — wrong-speaker penalty

Feature `speaker_penalty` = 1.0 when unit speaker resolves and ≠ q_speaker (unclamped), sweep w ∈ {−0.1, −0.2} over sm=0 baseline:

- cat-5 any@10: 0.404 (−0.1) / 0.385 (−0.2) — **never better than no-penalty 0.413**; per-task flips vs sm=0: −0.1 loses 2 wins 0; −0.2 loses 8 wins 2.
- Answerable any@10: +0.007 (0.598→0.605); penalty FPR (delivery flips on answerable): −0.1 → 2 lost vs 7 won; −0.2 → 3 lost vs 8 won — mechanically safe (~0.3–0.4% loss rate).
- Asymmetric bonus variant (match → +0.5·w, mismatch neutral): sm0+bonus0.2 gives all 0.562 / ans 0.605 / cat-5 0.404 — equivalent to pen−0.1.

Verdict unchanged across all scoring arms measured on the real pipeline (still 0.216/0.179).

## Arm 3 — abstention floor

Per-trigger hit rates (baseline): unanswerable a=0 b=1 c=0 d=45 → 21.6% refused; answerable a=0 b=0 d=139 → 17.9% FPR. Weak-pool decomposition: cat-5 weak pools (no SUPPORTED group) 47 — mm present on 45 (96%); answerable weak pools 160 — mm present on 139 (87%, all FPR). The operative mechanism ≈ "weak pool AND some majority-other-speaker group exists."

Premise coverage floor variants (INSUFFICIENT iff asked-speaker authored zero top-100 units with `_measure` coverage AND pool weak):
- floor replaces d (strict): CR 0.000 / FPR 0.000
- floor OR abstain+negative: CR 0.000 / FPR 0.001
- floor AND premise_mm: CR 0.000 / FPR 0.001

**Collapse**: asked-speaker units carrying ≥1 matched query term/canon exist in essentially every scored pool (asked speakers talk about the topic even when the specific premise is false), so the zero-support conjunct never holds. mm-tightening probes: require mm on top-scoring group → cat-5 refusals fall to 10/213; require mm on ≥50% of covered groups → 8/213. Pure weak-pool abstention (drop speaker check): CR 0.221 / FPR 0.206 — strictly worse than current. The current any-mm formulation sits at the measured frontier.

## Arm 4 — cat-5 premise delivery residual (baseline, 127 non-delivered)

Classes: outscored-in-pool 102 (48% of cat-5) · never-surfaced 13 (6%) · wrong-speaker-dominated displacers 12 (6%) · delivered 86 (40%). Gold speaker on non-delivered: 91 other-only / 23 asked / 13 unscored. **pack-10 displacers are 78% asked-speaker** (1383 asked vs 385 other) — the premise turn loses to the asked speaker's own topically-loose turns, not to wrong-speaker units. Median gold rank 49.5. Feature gap (mean displacer − gold): ent_idf **+0.268**, rrf_norm +0.170, lane_agree +0.044, t_prox +0.022, cov_idf **−0.044** (gold actually higher coverage). Weighted, rrf_norm (+0.170) + ent_idf (+0.161) ≈ the entire +0.332 score gap — entity-IDF surfaces the named speaker's incidental turns, out-ranking the premise turn. All 46 bm25-only cat-5 hits were surfaced in verbatim's fused pool — the residual is reranking, not recall.

ent_idf weight sweep (offline→real-validated): w=0.6→0.0 gives cat-5 0.404→**0.493**, all 0.556→**0.617**, every cat up (multi_hop +0.040, open_domain +0.064, single_hop +0.070, temporal +0.040); correct_refusal 0.211 / FPR 0.179 unchanged. ent0+phrase0.3 → cat-5 0.507 (offline). ent0 + sm=0 (offline): cat-5 **0.521**, ans 0.655, all 0.626.

## Verdicts

- **Which mechanism closes the cat-5 premise gap: `ent_idf→0`**, not any speaker arm. Measured real: cat-5 any@10 0.404→0.493 (−0.084 remaining to bm25 0.577), all 0.556→0.617, verdict untouched. Combined with `speaker_match→0`: cat-5 ~0.521 (offline). The premise gap is an entity-IDF displacement problem — the asked speaker's incidental turns out-score their own false-premise turns — consistent with the a5 LOLO result (−ent +0.077 MRR; ent_idf worst weight).
- **Speaker arm value**: the u_speaker visibility fix works mechanically (engagement 5.8%→99.1%) but the feature it powers is a liability on cat-5 (0.404→0.286 @0.4) and a small gain on answerable (+0.007–0.016 with weight). Net value ≈ slightly positive only if sm is killed (0) or made asymmetric (match-bonus/penalty −0.1 ≈ +0.007 answerable); as symmetric match/mismatch it is net-negative. Recommend: ship the provider fix only alongside sm=0 or the asymmetric variant; do NOT weight-boost it.
- **Abstention floor verdict**: the spec'd premise-coverage floor is dead on arrival — fires 0/990 (CR 0.000, FPR ~0) because loose topical support by the asked speaker is ubiquitous. The existing premise_mm trigger is the ONLY working refusal mechanism on this corpus (45/46 cat-5 refusals, all 139 FPR); it sits at the measured frontier — every tightening variant loses refusals (top-group-mm 10/213, majority-covered-mm 8/213), and pure weak-pool abstention is strictly worse (0.221/0.206 vs 0.216/0.179). The 17.9% FPR is weak-pool withholding on answerable tasks (139/160 = 87% of answerable weak pools) — a recall/pool-strength problem, not a speaker-attribution problem; cutting it needs better premise-turn delivery (arm-4), not verdict surgery.
- **Measured FPR of each**: wrong-speaker penalty delivery-FPR ≈ 0.3–0.4% on answerable (2–3 flips lost of 777) — safe but useless for cat-5; all scorer/verdict arms leave verdict FPR at 0.179 unchanged (verdict is score-order-insensitive); floor variants FPR ~0 at the cost of CR 0.000 — worst trade on the frontier.
