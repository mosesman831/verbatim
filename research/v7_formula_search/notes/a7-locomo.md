# a7-locomo — LoCoMo benchmark structure working note

Scope: what the LoCoMo QA benchmark actually measures at the retrieval layer, extracted from the ACL 2024 paper (2024.acl-long.747, the version matching the released data), arXiv v1 (2402.17753), the released repo (snap-research/locomo), `locomo10.json` itself (cloned and inspected), and repo issues #6/#27/#29 plus downstream reproductions. Everything below is verified against the released artifacts, not just the paper prose.

---

## 1. The two versions of the benchmark — compare against the right one

| | arXiv v1 (2402.17753, Feb 2024) | ACL 2024 official (2024.acl-long.747) |
|---|---|---|
| Conversations | 50 | 10 (the longest 10, released as `locomo10.json`) |
| Turns | ~300 avg/conv (~9K tokens) | 5,882 total; 588.2 avg/conv; 27.2 sessions/conv; 21.6 turns/session; ~16.6K tokens/conv |
| QA questions | 7,512 (SH 2,705 / MH 1,104 / Temp 1,547 / OD 285 / Adv 1,871) | 1,986 (SH 841 / MH 282 / Temp 321 / OD 96 / Adv 446) |
| Retriever table | Table 3, gpt-3.5-turbo-16k reader | Table 3, gpt-3.5-turbo reader; different numbers |

**The released data is the ACL version.** Every downstream eval (mem0, Zep, MemMachine, Verbatim's harness) runs on `locomo10.json` = 10 conversations / 1,986 QAs. The arXiv-v1 retrieval table was computed on the pre-release 50-conversation set and must not be quoted as the locomo10 baseline. (One trap: arXiv v1 Table 3 contains a literal typo — multi-hop Dialog R@10 is printed as "247.4"; the true value is ~47.4, monotonic between 34.4@5 and 64.1@25.)

## 2. Category definitions — precise semantics (sub-question 1)

Paper §4.1 (identical text in both versions) defines five reasoning categories **in this prose order**: (1) single-hop, (2) multi-hop, (3) temporal, (4) open-domain/commonsense, (5) adversarial. The released data does **not** number them in that order — see §3.

Definitions, enriched with what the released data actually contains:

- **Single-hop** — answer lives in a single session. In data: `category=4`, n=841, mean evidence 1.07; 795/841 have exactly one evidence turn. Example: "What did the charity race raise awareness for?" → "mental health", evidence `D2:2`.
- **Multi-hop** — answer requires synthesizing information from multiple sessions. In data: `category=1`, n=282, mean evidence **3.13** (max 19). Evidence is genuinely dispersed: 89% of multi-evidence items sit in *different* sessions (distinct-session distribution: 2→165, 3→56, 4→28, 5→11, … 15→1); only ~2% of evidence pairs are same-session adjacent. Scored with a comma-split multi-sub-answer F1 in the official eval.
- **Temporal** — answer needs time-related cues / temporal reasoning. In data: `category=2`, n=321, mean evidence 1.17; 77% start with "When", but the long tail is "How long did it take…", "What year…". Evidence = the turn containing a relative-time phrase; answering requires joining it to `session_N_date_time` (e.g., "yesterday" in D1:3 + session date "8 May 2023" → answer "7 May 2023").
- **Open-domain** — answer integrates speaker-provided information with external commonsense/world knowledge. In data: `category=3`, n=96, mean evidence 2.17, includes counterfactuals ("Would Caroline still want to pursue counseling if she hadn't received support…"). Answers can carry a `;` suffix ("answer; justification") — eval takes `answer.split(';')[0]` (11/96 affected). Four cat-3 questions have **no evidence** at all → 92 scoreable.
- **Adversarial (unanswerable)** — "designed to trick the agent into providing wrong answers, with the expectation that the agent will correctly identify them as unanswerable." In data: `category=5`, n=446, field is `adversarial_answer` (the trap answer), *no* `answer` field (except 2 noisy rows). **Mechanism, verified in data: minimal-pair corruption** — e.g. cat-4 "How did *Melanie's* son handle the accident?" (D18:6–7) vs cat-5 "How did *Caroline's* son handle the accident?" (same evidence refs). Other patterns: pronoun/speaker swaps ("…her store" for Jon), nonexistent events. Every cat-5 question still has ~1.03 evidence items — the *premise-related* turns a retriever should find so the answerer can refuse.

## 3. The category-id swap (sub-question 2) — confirmed, with the fix

The integer `category` in `locomo10.json` does **not** follow the paper's §4.1 numbering. The true mapping, verified three ways:

| data `category` | type | n | evidence-bearing | paper Table 5 label (paper order) |
|---|---|---|---|---|
| 1 | multi-hop | 282 | 282 | "multi-hop retrieval 282 (14.2%)" |
| 2 | temporal | 321 | 321 | "temporal reasoning 321 (16.1%)" |
| 3 | open-domain | 96 | 92 | "open domain knowledge 96 (4.8%)" |
| 4 | single-hop | 841 | 841 | "single-hop retrieval 841 (42.3%)" |
| 5 | adversarial | 446 | 446 | "adversarial 446 (22.4%)" |

Evidence: (a) official `task_eval/evaluation.py` `eval_question_answering` — cat `1` gets the comma-split multi-sub-answer `f1()`, cats `2,3,4` get plain `f1_score`, cat `3` gets the `;`-split, cat `5` gets the refusal keyword check; (b) counts above match the ACL Table 5 percentages 1:1; (c) content checks — cat 2 is 77% "When…", cat 1 has multi-session evidence, cat 4 is 95% single-evidence.

Independent documentation: repo issue #6 (May 2025, first report), issue #29 (documentation request), MemMachine's Appendix A (Sep 2025), Goal-Directed Search paper arXiv:2511.21726 Appendix E (Nov 2025). mem0's standardized `memory-benchmarks` repo hardcodes `{1: multi-hop, 2: temporal, 3: open-domain, 4: single-hop, 5: adversarial}` with `CATEGORIES_TO_EVALUATE = [1,2,3,4]`.

Caution: mem0's *old* issue reply (#2609) guessed `{1:SH, 2:T, 3:MH, 4:OD}` — superseded by their own later benchmark code. And kitfunso/hippo-memory's README still propagates the paper-order labels verbatim — a live example of the swap corrupting downstream per-category tables.

## 4. License (sub-question 3)

**CC BY-NC 4.0 confirmed in three places**: repo `LICENSE.txt` (full "Attribution-NonCommercial 4.0 International" text), paper §B.2 ("released under the CC BY-NC 4.0 DEED license"), and the author's HF dataset card `adymaharana/locomo` (`license:cc-by-nc-4.0`). Non-commercial use only — fine for an internal eval harness; do not ship the data in a product.

## 5. Evidence items & retrieval scoring (sub-question 4)

**An evidence item is a dialog turn**, addressed by `dia_id = "D<session>:<turn>"` (1-based; `D1:3` = session 1, turn 3). Conversations store `session_N` lists of `{speaker, dia_id, text}` plus `session_N_date_time`. Observations (the paper's second retrieval unit) are per-session, per-speaker `[assertion_text, "D_s:t"]` pairs — each pinned to one source turn. Session summaries are the third unit (~130 tokens/session).

**Paper's official retrieval metric** (`evaluation.py` lines 228–235): after a RAG run stores its retrieved ids in `q[model_key + '_context']`, per-question score is

```
if context[0].startswith('S'):                       # summary-mode: ids like "S3"
    recall_acc = |{ e.split(':')[0][1:] for e in evidence } ∩ sessions| ... 
    # precisely: mean over evidence e of (session_id(e) ∈ retrieved_sessions)
else:                                                # dialog/observation mode: ids like "D3:7"
    recall_acc = |evidence ∩ retrieved_ids| / |evidence|
# questions with empty evidence or missing _context contribute 1.0
```

i.e., **proportional evidence recall** (fraction of the question's gold turns found), macro-averaged over questions — *not* a hit rate. Formally, for question q with gold set E_q and retrieved set R_k: `recall@k(q) = |E_q ∩ R_k| / |E_q|`; score = mean_q recall@k(q).

**Verbatim's "any@k" is a different quantity**: `any@k(q) = 1[∃ e ∈ E_q : e ∈ R_k]`. Pointwise any@k ≥ recall@k, with equality iff |E_q| = 1. Since 73% of scored questions (1127/1540; 1127 single-evidence of 1,536 evidence-bearing) have exactly one evidence item, the two metrics coincide on most of the set and diverge mainly on **cat 1 (mean 3.13 evidence)** and **cat 3 (2.17)**. A third variant, agora's `full_recall@k`, requires ALL gold turns in top-k.

Worked example (3 questions, k=10):
- Q1 (cat 4): E={D2:2}; retrieved {D2:8, D2:5, **D2:2**, …} → any=1, recall=1.0, full=1.
- Q2 (cat 1): E={D1:3, D5:2, D9:4}; retrieved contains only D5:2 → any=1, recall=0.333, full=0.
- Q3 (cat 2): E={D11:3}; not retrieved → all 0.
- Scores: any@10 = 2/3 = 0.667; recall@10 = (1+0.333+0)/3 = 0.444; full@10 = 1/3 = 0.333.

Rough conversion sanity-check: if a retriever hits each evidence item independently with probability p, E[any] ≈ 1−(1−p)^{|E|}; for p=0.5 and |E|=3.13 that's 0.885 any vs 0.50 recall — a huge stated-metric gap on cat-1-heavy comparisons. **Never compare Verbatim's any@k against the paper's recall@k as if they were the same metric.**

**Evidence-field noise (verified in data)**: 6 malformed evidence entries — `'D8:6; D9:17'` (semicolon-joined), `'D'` alone, `'D:11:26'` (missing session id), and 3 space-joined multi-id strings (`'D22:1 D22:2 D9:10 D9:11'`). Robust handling: extract all `D\d+:\d+` tokens with a regex per evidence string (recovers the 4 joined entries = 11 extra turns), drop `'D'` and `'D:11:26'` (or interpret the latter as `D11:26` — ambiguous). ~0.2% of evidence entries; invisible in aggregate but they break exact-match parsers.

**Indexing granularity used for the published numbers** (`rag_utils.py::get_context_embeddings`): one embedding per dialog turn; the embedded string is `(session_date_time) speaker said, "text"` (+ `[shares <blip_caption>]` for image turns); retriever = DRAGON-plus (CLS token, L2-normalized, dot-product). Note speaker name and session timestamp are baked into the indexed text — Verbatim's known bugs (possessive/case-sensitive entity match, no speaker field, unresolved relative-time phrases) are exactly the fields the paper's baseline exploits.

## 6. Published retrieval numbers (sub-question 5)

**First-party (ACL 2024 Table 3, DRAGON retriever on locomo10, proportional evidence recall):**

| Unit | k | SH | MH | Temp | OD | Adv | Overall |
|---|---|---|---|---|---|---|---|
| Dialog | 5 | 68.0 | 35.4 | 70.4 | 33.1 | 43.9 | 56.7 |
| Dialog | 10 | 77.7 | 46.6 | 77.3 | 40.8 | 54.7 | 66.2 |
| Dialog | 25 | 87.1 | 62.5 | 83.5 | 52.6 | 66.3 | 76.7 |
| Dialog | 50 | 91.1 | 73.0 | 89.6 | 61.7 | 72.5 | 82.7 |
| Observation | 5 | 67.1 | 41.4 | 73.1 | 35.1 | 37.7 | 56.2 |
| Observation | 10 | 70.5 | 50.9 | 76.4 | 37.6 | 45.1 | 61.3 |
| Observation | 25 | 74.7 | 60.6 | 80.3 | 49.0 | 53.5 | 67.5 |
| Observation | 50 | 76.3 | 67.8 | 82.0 | 55.9 | 59.5 | 71.2 |
| Summary | 2 | 63.0 | 33.0 | 57.7 | 31.5 | 65.9 | 65.9 |
| Summary | 5 | 77.0 | 54.3 | 72.8 | 47.6 | 79.1 | 72.1 |
| Summary | 10 | 88.7 | 72.7 | 84.5 | 67.3 | 88.8 | 84.7 |

(arXiv v1's version of this table — Dialog R@10 = 67.5 etc. — was computed on the pre-release 50-conv set; do not mix.) Key reads: **open-domain is the weakest retrieval category** (40.8 R@10 dialog) and **multi-hop is second-weakest** (46.6) — the same two categories Verbatim is told to win. Session-summary units reach 84.7 overall at k=10 — high coverage, but the paper shows QA *drops* on summaries (information loss); for a byte-pin engine, summaries are a routing/coarse filter, not deliverable evidence.

**Independent reproductions (retrieval-only):** DanceNitra/agora `mnemo` probe (retrieval map, embedder-only, no LLM): 10 convs / 5,882 turns / 1,531 answerable questions; metrics recall@{5,10,20} + full_recall@k by category with cluster-aware stats (paired Wilcoxon, per-conversation win rate, conversation-level bootstrap — *because 1,536 questions are nested in only 10 conversations*). Their headline: reproduces the BEIR "BM25 is a strong zero-shot baseline" pattern — LoCoMo has high question↔evidence lexical overlap favoring lexical methods, and gold-turn recall under-credits semantically-equivalent unannotated turns. kitfunso/hippo-memory ships a deterministic `evidence` scoring mode (dia_id recall@K); no published numbers found.

**Downstream systems report QA/judge accuracy, not retrieval** — listed only for context: mem0 J-score ~66–77% (judge leniency criticized in memory-benchmarks issue #10), SUMER 66.79 judge accuracy (arXiv:2511.21726), MemMachine 91.69% (gpt-4.1-mini judge, their own prompts), EverMemOS claimed 92.32% (independent reproduction reportedly ~38%). A dial481 audit finds **6.4% of the answer key is wrong** (99 score-corrupting errors across 1,540 questions — hallucinated gold answers, temporal errors, speaker-attribution flips) and a generic LLM judge accepts ~62.8% of intentionally-wrong topical answers. **Implication: evidence-id retrieval scoring is substantially more trustworthy than QA-judge scoring on this benchmark** — Verbatim's evidence-based harness choice is validated, modulo the ~0.2% malformed evidence strings and scattered INACCURATE_EVIDENCE_REF labels (issue #27).

## 7. Candidate formulas / decisions — equations, cost, placement, verdicts

Since this question is benchmark-structure, the "candidates" are the metric + harness + granularity formulas the synthesis engineer must pin.

### C1 — Evidence-metric set (eval harness)
- `any@k(q) = 1[max_{e ∈ E_q} 1(e ∈ R_k)]`, score = mean over evidence-bearing q.
- `recall@k(q) = |E_q ∩ R_k| / |E_q|` (paper's official metric — needed for comparability to Table 3).
- `full@k(q) = 1[E_q ⊆ R_k]` (strict coverage; discriminating on cat 1 where mean |E| = 3.13).
- Params: k ∈ {5, 10, 20, 50} matching paper k values; q-set = 1,536 (cats 1–4 with ≥1 parseable evidence turn).
- Time per query: k hash lookups ≤ 50×19 ≈ O(k·|E|) ≈ <1 µs; whole eval ≈ 1536 × ~1 µs ≈ **<0.01 ms — identical at 10k and 100k memories** (post-retrieval, corpus-independent). Resident size: evidence strings ~2,800 × ~6 B ≈ 17 KB total.
- **Verdict: ship all three** — any@k headline (matches current harness), recall@k for paper comparability, full@k as cat-1 diagnostic. Confidence: high. What would change it: if the harness switches to turn-set dedup, recompute baselines.

### C2 — Category mapping {1:MH, 2:T, 3:OD, 4:SH, 5:Adv}
- Formula: relabel table above; evaluate cats ∈ {1,2,3,4} (evidence-bearing n = 1,536). 
- Cost: zero. **Verdict: ship.** Confidence: maximal — five independent confirmations + eval-code semantics. Reject any paper-order labeling; if an external table disagrees on category *names*, check which mapping they used before comparing numbers.

### C3 — Evidence granularity = dialog turn (dia_id)
- Engine-side implication: every indexed memory must carry its source `dia_id`; the byte-pin rule is satisfiable since evidence targets whole turns (~124 chars avg).
- Index size if Verbatim indexes at turn granularity (per memory): text ~124 B + FTS index ≈ 0.4–0.6× text (~50–75 B) + hashed-vector payload (e.g. 4 B × ~32 buckets ≈ 128 B, or nothing on default path) → **~200–330 B/memory → 2–3 MB at 10k, 20–33 MB at 100k** — trivially resident. Session-level alternative: ~272 units/conv ≈ 4.6% of the rows — coarser, cheaper, but loses byte-level evidence addressing.
- **Verdict: ship turn-level as canonical evidence unit; keep session_id + speaker + session_date_time as indexed metadata fields.** (These are exactly the fields behind three of the filed bugs.)

### C4 — Robust evidence parser
- `parse_evidence(s) = re.findall(r'D(\d+):(\d+)', s)` per string, dedup; recovers 11 gold turns from the 4 joined malformed entries; drops `'D'`, `'D:11:26'`.
- **Verdict: ship** ��� one-line fix, eliminates a silent eval bug. Confidence: high.

### C5 — Metric variants to NOT adopt as primary
- **Session-level summary retrieval as the scored unit**: paper gets 84.7 R@10 but the unit is a lossy generated artifact — cannot satisfy byte-pin; QA degrades on it. **Reject as evidence unit; optional as a coarse candidate-generation lane** (session prefilter narrows the eligible set cheaply — approximate *candidate generation*, allowed by the eligibility rule).
- **any@k as the sole reported number**: overstates multi-hop competence (hits easiest of ~3.1 items). **Ship with recall@k alongside**; treat any-only reporting as reject.
- **Comparing any@k to paper recall@k**: different formulas — **reject**; recompute paper-metric on Verbatim runs before claiming parity/superiority to DRAGON (proportional R@10 = 66.2 dialog is the bar; on the same question set Verbatim's BM25 any@10 = 59.6 would convert to a *lower* recall@10 — the gap is real but smaller than headline numbers suggest only if measured correctly).

### C6 — Statistical rigor on categories
- OD n=96 (92 scored): per audit math, adjacent systems need a ≥15-pt gap for significance; MH n=282; evidence items nested in only 10 conversations → report per-conversation means + Wilson 95% CIs (or the agora-style conv-level bootstrap), never bare point estimates.
- **Verdict: ship Wilson CIs + per-category breakdown; treat <10-pt per-category deltas as noise.** Confidence: high — arithmetic is straightforward and both independent audits reached the same conclusion.

### C7 — Adversarial (cat 5) handling
- Evidence exists (premise turns, mean 1.03). Paper's official scorer is a 2-phrase keyword match ('no information available' / 'not mentioned') — primitive; audits note its alternative MCQ formatter is broken on 444/446 rows. For *retrieval*, cat-5 evidence is findable like any other (DRAGON dialog R@10 Adv = 54.7); the refusal is an answering-layer concern.
- **Verdict: optional — include cat-5 in retrieval scoring only if the harness wants premise-recall coverage (adds 446 questions); exclude from headline any@k per the cats 1–4 convention.** Confidence: medium — depends whether Verbatim wants a separate abstention eval later.

## 8. Calibration for Verbatim's priorities (so-what summary)

Open-domain (cat 3) and multi-hop (cat 1) are the stated priorities — and they are objectively the weakest published retrieval cells (DRAGON R@10: OD 40.8, MH 46.6 vs SH 77.7, Temp 77.3). The reason differs per category: OD questions deliberately avoid lexical answers (needs commonsense inference over retrieved speaker facts — evidence turns rarely share surface terms with the answer); MH needs *coverage* across 2–15 sessions (fusion/lane-diversity problem — MMR-style dispersion helps more than raw score). Temporal needs the session-date join + relative-phrase resolution (77% "When…"). The benchmark's high question↔evidence lexical overlap explains why a 40-line BM25 (any@10 = 59.6) currently beats Verbatim (54.6): lexical lanes dominate when eligibility/post-fusion machinery degrades BM25's natural output — matching the agora reproduction's "BM25 is a strong zero-shot baseline" finding on this exact data.