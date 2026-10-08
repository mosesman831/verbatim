# Verification audit — Verbatim v7 post \"Beat-it retrieval plumbing fixes\"

**Objective.** Measure refusal quality + lane participation on the REAL LoCoMo dev split post-fix, and quantify the reviewer's flagged regression (live graph lane → weak-FTS-seeded spreading activation → partial-labeled junk delivered as READY → correct_refusal collapse).

## Method & provenance

- Store rebuilt read-only via `VERBATIM_EVAL_LOCOMO=1 python3 -m eval.v7.track_r --dataset locomo --split dev --arms verbatim --workdir /tmp/v7w` on HEAD. Dev split: **990 tasks / 2877 items**, categories single_hop 429, adversarial 213, temporal 175, multi_hop 126, open_domain 47 (cat-5 = adversarial = unanswerable).
- Authoritative task-level numbers from `/tmp/v7w_report.json` (real `Memory.search`, k_list=[10,20], timeout_ms=2000).
- Deep-dive probes replay `facade._search_v7` byte-for-byte (`/tmp/probe_v7.py`, store opened read-only) at `limit=_MAX_LIMIT=64` — the same pool the arm searches (facade always runs the pipeline at _MAX_LIMIT then truncates to caller's k). Probes match the report within deadline-jitter: cat-5 refusals 45 vs probe-46; answerable refusals 136 identical. ±1–2 jitter comes from deadline-bounded lanes (graph always hits its slice deadline).
- Per-item `group_label` values are the **pipeline's own labels** (propagated from S7 groups onto explain items). A replayed `classify_groups` on returned `scored` is NOT bit-identical — `_materialize_details` mutates `detail` between S7 classify and S8 pack (adds `session_id`), changing `_group_key` precedence (session→src→unit) and inflating grouping. Do not trust re-classification of materialized items as a verdict audit path; use `explain.items[].group_label`.

## (a) correct_refusal + status distribution — post-fix

Authoritative (track_r report, verbatim arm):

| task set | n | ready | insufficient | refused |
|---|---|---|---|---|
| unanswerable (adversarial/cat-5) | 213 | 168 | 45 | **45/213 = 0.211** |
| answerable (cats 1–4) | 777 | 641 | 136 | — |

Per-category status (probe data, same verdicts): single_hop 369/60 · temporal 148/27 · multi_hop 91/35 · open_domain 33/14 · adversarial 167–168/45–46.

- All 45 cat-5 refusals fired **trigger=d** (`trigger=d; calibration=unfitted`). Trigger (c) never engaged (unfitted); (a)/(b) never needed.
- Delivered refs are non-empty even on `insufficient` (pack ships ~20 distinct corpus refs regardless of verdict; `abstained` rides on status only). The metric counts these as refused via `withheld`.
- Overall arm metrics: **any@10 0.535, all@10 0.456, prop@10 0.489, ndcg 0.318, abstain_rate 0.183**, session any@10 0.846. Attribution: delivered 524, lane_miss 104, rank_shift 88, packed_out 0, abstain 61, unsupported 213.
- **vs pre-fix baseline** (lane_miss_diagnosis.md): any@10 0.195 → 0.535, delivered 222 → 524, lane_miss 444 → 104, packed_out 11 → 0. Retrieval far healthier; refusal did not collapse to the reviewer-probe 0.0 but sits at ~0.21.

## (b) premise-check FPR on answerable cats 1–4

- **136/777 answerable tasks (17.5%) end INSUFFICIENT — every one fired trigger=d**, i.e. the new premise-mismatch path (no `abstain_likely` class exists on this split; `negative_evidence` observed on 0 groups — premise_mismatch carries the trigger alone).
- FPR per category: multi_hop 27.8% (35/126), open_domain 29.8% (14/47), single_hop 14.0% (60/429), temporal 15.4% (27/175).
- Of the 136: **62 had gold evidence inside the top-10 pool** (true false abstention — deliverable evidence withheld); 74 had no gold in the pool (retrieval miss anyway; refusal defensible-but-mislabeled).
- Trigger-d precondition audit (answerable): single-speaker resolution on 671/777 (86%); premise-mismatch groups present on 659 of those; the trigger fires whenever NO group reaches SUPPORTED. Conditional rate: 136/659 = 20.6% of pmm-present resolved tasks — the check refuses ~1 in 5 of BOTH populations (cat-5: 46/211 = 21.8%), near-zero discriminative power as constructed.

## (c) lane participation — post-fix

Store after full drain (all jobs succeeded — source_project/embed/harvest 2877 each, admit 7122):

| table | post-fix | pre-fix |
|---|---|---|
| units total | **4143** (turn 2877 + sentence_window 1266) | 7020 (twins) |
| graph_edges | **145,042** (co_mention 56,580, same_session 40,764, temporal_near 28,960, adjacent_turn 17,879, causal 859) | 0 |
| observations_v7 | 38 | 0 |
| entity_canon | **853** | 1513 |
| entity_mentions | 13,683 (speaker 2877 + mention 10,806) | — |
| events_v7 / state_facts / preferences / profiles_v7 | 1248 / 156 / 219 / 244 | — |
| unit_fts / unit_vectors | 4143 / 2877 | — |
| t2_facts / facts_fts / entities / fts_rows | **0 / 0 / 0 / 0** | 0 |
| claims / reviews | 7122 / 14,812 — all reviews state='open' | — |

- `require_review=True` (AdmissionConfig default) confirmed parking every claim: nothing reaches ACTIVE → `t2_facts`/`facts_fts`/`entities`/`claim_entities` all 0 on default config. Expected — but the claim/entity plane contributes nothing to retrieval in this config.
- Per-lane candidate emission over all 990 tasks (probe explain):

| lane | emits >0 | status profile | skip reason |
|---|---|---|---|
| lex | 989/990 | ok 989 | — |
| dense | 990/990 | ok 990 | — |
| ent | 990/990 | ok 980, partial 10 | — |
| graph | **990/990** | **partial 990 — always `deadline`** (still emits 200 cands after truncated expansion) | emitted 0 pre-fix |
| obs | 613/990 | ok 614, skipped 370 `intent_temporal`, deadline 6 | correct skip |
| typed | 371/990 | ok 366, partial 11, deadline 6 | `intent_not_typed` — correct skip |
| time | 367/990 | ok 207, partial 196 | `no_window` 587 — correct skip |
| fuzzy | 42/990 | ok 181 (0-cand), partial 42 | `no_oov_terms` 767 mostly; `no_vocab_table` on 6 |

- Graph lane emits on every query but burns its full slice deadline every time — live but expensive; median search 557ms, p90 1332ms.

## (d) weak-evidence regression — measured

**The delivered \"junk\" is the planted near-miss premise itself.** Cat-5 tasks carry `evidence_ids` naming the adversarial premise turns. Of 168 ready cat-5 tasks, **91 (54.2%) deliver ≥1 of their premise refs** — retrieval lands exactly where the question was designed to look. (22/45 refused tasks also surfaced the premise ref but withheld on coverage.) Report `any@10` on adversarial = 0.385 reflects these premise hits.

Delivered-label mix on cat-5 (top-10, authoritative group_label):

- ready tasks (168): **partial 1146 / supported 524 / weak 0** — 14 of 168 ship a partial-only top-10 (the reviewer's exact scenario); 154 ship ≥1 supported-labeled near-miss.
- insufficient tasks (45): all-partial pools, withheld by verdict.
- `weak` labels almost never materialize anywhere (30 of ~77k answerable items, **0 on cat-5**) — `graph_ppr`/`ent`/`lex` are `_REAL` signals, so every surfaced item clears the partial tier via `real and (cov>0 or ecov>0)`.

Separation — per-item term coverage (`matched_terms \u2229 qterms / |qterms|`) of best delivered item:

| population | min | q1 | median | q3 |
|---|---|---|---|---|
| cat-5 ready | 0.20 | 0.385 | 0.455 | 0.538 |
| answerable gold-hit | 0.231 | 0.417 | 0.500 | 0.571 |
| best GOLD item (answerable) | 0.111 | 0.364 | 0.450 | 0.538 |

Floor sweeps (refuse when best top-10 term-cov < t):

| floor | cat-5 refused (+of 167 ready) | answerable cost (of 777) | gold-hit tasks lost |
|---|---|---|---|
| 0.45 | +57 | 350 | 163 |
| 0.50 | +85 | 424 | 205 |
| 0.55 | +120 | 586 | 315 |
| 0.60 | +136 | 648 | 357 |

Variant floors measured:

- \"no supported item in top-10 → refuse\": 60/213 cat-5 (28.2%) at 197/777 FPR (25.4%) — strictly worse than current.
- \"pmm item in top-10 AND no supported → refuse\": 52/213 (24.4%) at 121/777 (15.6%).
- \"pmm present AND no supported item authored by q_speaker → refuse\" (speaker-aware guard): 97/213 cat-5 (45.5%) at 190/777 (24.5%) — the only variant that moves TPR much, at unacceptable FPR. (Top-10 measured: q_speaker-authored supported item on 114/211 resolved cat-5 vs 481/671 resolved answerable.)
- pmm-fraction of top-10 also fails to separate (cat-5 med 0.2 vs answerable med 0.1).

**Finding: no retrieval-level signal separates the populations.** Cat-5 questions are near-misses by construction — they name the asked speaker and real topics, so the speaker's own topically-adjacent turns produce high-coverage groups that honestly meet the coverage labeler. Coverage/score/label-mix/pmm-fraction all overlap. The defect isn't junk labeled weak — it's near-miss evidence labeled *supported*.

## (e) junk canon check

- `entity_canon` = **853 rows** (pre-fix 1513, \u221244%). Space-containing canons: **233/853 (27.3%)** — but **zero two-speaker fusions**: the `_cap_spans` ':' boundary completely eliminated the \"jon gina\" class (every space-canon checked against the 10 speaker canons: 0 contain ≥2).
- Residual space-canons are a different class: contraction fragments (`can t` 64, `you re` 40, `don t` 25, ~21 distinct), vocative+addressee pairs (`thanks nate` 49, `hey john` 33, `congrats evan`…, ~78), and legitimate multi-word names (`ac valhalla`, `animal crossing`, `icefields parkway`, `iron man`, `eisenhower matrix`).
- Verdict: the colon fix worked for its target; a residual vocative/contraction class remains (harmless noise but still junk for canon hygiene).

## Verdicts

- **correct_refusal post-fix: 0.211 (45/213)** — not the reviewer-probe 0.0; the dev split refuses ~1 in 5 cat-5 via trigger=d. Of the 168 leaked, 54% deliver the very premise turn the question was built around — the regression is real, and it is a *supported-labeled near-miss* problem, not a weak-junk problem. Every refusal fires premise_mismatch; speaker binding resolves on 211/213 (99%) — the r7 coverage estimate held; the **no-SUPPORTED guard** is the bottleneck, not speaker resolution.
- **Premise-check FPR: 0.175 (136/777)**, all via trigger=d — same trigger that saves cat-5. 62 of 136 had gold in the pool. Trigger-d fires at 21.8% on resolved cat-5 and 20.6% on resolved answerable — near-zero discriminative power; it tracks thin coverage, not unanswerability.
- **Weak-evidence floor: NOT justified in the proposed forms.** Delivered junk is 67% partial / 31% supported at top-10, 0% weak. No coverage floor separates (every point trades ≥1 answerable per cat-5 refused). The only variant that raises TPR (speaker-aware SUPPORTED guard) reaches 45.5% cat-5 refusal at 24.5% FPR — still a bad trade. Recommendation: do not add a coverage/score floor to trigger (d); cat-5 needs a presupposition-vs-fact signal the group labeler can't express (e.g. verify the asked relation against structured state, or mark item-level speaker-mismatch *within* the top group rather than across groups).
- **Lane status table (post-fix)**: lex ok 989/990 · dense ok 990/990 · ent ok 980 (+10 partial) · graph emits 990/990 (always deadline-partial) · obs 613 emit / 370 intent-skip · typed 371 / 607 intent-skip · time 367 / 587 no-window · fuzzy 42 emit / 767 no-OOV-skip. Units 4143 (twins suppressed), graph_edges 145,042, obs 38, entity_canon 853 — all lane plumbing live; claims parked under require_review default (facts/entities/fts_rows 0 by config, not by bug).
