# c5-consolidation — Consolidation and derived views without generation

## 0. Framing constraint

Verbatim's byte-pin rule ("every delivered sentence must byte-match retained source bytes; generated text with no pin is not a memory") means an observation row **cannot emit a synthesized claim** like Hindsight's "Alice is a Python-focused developer." The deterministic observation is therefore an **aggregate record**, not a sentence: `(entity, predicate) -> {head_evidence, proof_count, pins[], validity interval, supersession link}`. Its deliverable text is the head evidence's stored payload — already byte-pinned — and the observation contributes *routing, grouping, proof and freshness metadata* to ranking, never novel prose. This makes the design easier, not harder: no LLM needed on the default path.

## 1. Candidate: slot-grouped observation rows (recommendation)

**Formula.** For every typed fact evidence row `f = (entity_id e, predicate p, value v, ts t, payload bytes)`:

```
key(f)      = (e, p)                        # the slot
obs(e,p)    = { head: argmax_{f in slot, not superseded} f.ts,
                proof_count: |{f in slot, not superseded}|,
                pins: [f.id for f in slot, not superseded],
                as_of: max f.ts over pins,
                value_hist: [(v_i, t_valid_i, t_invalid_i)] }
```

- `head` is the newest non-superseded evidence; its bytes are the observation's text.
- `proof_count` counts distinct supporting evidence rows (Hindsight uses "proof count" terminology — docs.vectorize.io).
- Parameters: none beyond the slot schema. Slot extraction is already Verbatim's typed-fact job.

**Retrieval semantics.** Observation rows are a peer lane (`observations` already exists in the lane list). A hit emits the head's pinned bytes with `proof_count` attached; post-fusion boost `+alpha_p * log(1+proof_count)/log(1+P_max)` inside the existing bounded-boost budget (alpha_p <= 0.10 consistent with boost alphas 0.2/0.2/0.1).

**Worked example.** Evidence: `(alice, works_at, "Google", t1)`, `(alice, works_at, "Google", t5)`, `(alice, works_at, "Meta", t9)`. Slot `(alice, works_at)` yields obs: head = "Alice works at Meta" span @t9, proof_count = 1 (current epoch) or 3 (lifetime count — recommend two fields: `proof_count` current + `proof_total` lifetime), pins = [t9-fact], value_hist = [Google:[t1,t9), Meta:[t9,inf)].

**Per-query cost.** One extra FTS lane ~= cost of the existing lexical lane: FTS5 MATCH on ~15k obs rows (at 100k memories, ~15% extraction ratio) is sub-ms to ~1 ms; observation pool ~10 candidates -> freshness flags 10 x 0.002 ms = 0.02 ms. Total added: **~1-2 ms at 10k and ~2-4 ms at 100k** — inside the "tens of ms" budget.

**Bytes.** Row ~= 194 B fixed (id 16 + entity 8 + pred 2 + claim ptr 8 + proof 2 + pins 4x8 + as_of 8 + supersede ptr 16 + validity 16); measured 100k memories -> ~15k obs -> **~2.9 MB**. With an optional MinHash signature (128x4 B) per obs: ~10.6 MB/100k. Negligible.

**Stage:** write-path (consolidation job) + lane (observations) + fusion boost input.

**Verdict: ship (default).** It is the only consolidation shape that satisfies byte-pin by construction. Confidence: high — every vendor converged on this unit (Hindsight observations w/ proof count + quotes; Zep bi-temporal edges; Mem0 facts). What would change it: evidence that obs lanes hurt single-hop retrieval — then demote to boost-only.

## 2. Candidate: near-duplicate merge policy

**The measured problem.** I ran Jaccard on word-1gram shingles over claim pairs (pure Python, 128-perm MinHash verified against exact J):

| pair type | J(w1) | J(w3) | char-5 | should merge? |
|---|---|---|---|---|
| paraphrase ("prefers python for data pipelines" / "likes python for data pipeline work") | 0.44 | 0.12 | 0.49 | yes |
| same claim | 1.00 | 1.00 | 1.00 | yes |
| contradiction ("works at google" / "works at meta") | 0.60 | 0.33 | 0.52 | **no** |
| value-change ("favorite drink coffee" / "tea") | 0.67 | 0.50 | 0.69 | **no** |
| different entity, same value ("alice@google" / "bob@google") | 0.60 | 0.33 | 0.60 | **no** |
| num-diff ("deadline march 15" / "march 18") | 0.67 | 0.50 | 0.92 | **no** |
| near-repeat ("loves react for frontend work" / "...development work") | 0.86 | 0.50 | 0.61 | yes |

**No Jaccard threshold separates paraphrases from contradictions on short claim text** (paraphrase 0.44 < contradiction 0.60). Any threshold >=0.5 that catches paraphrases also merges "works at Google" with "works at Meta." This is the single most important measured result of this note.

**Consequence — the merge gate must be structural, not similarity-based:**

```
merge(a, b)  :=  a.slot == b.slot AND norm_value(a) == norm_value(b)
                 OR  jaccard_w1(a.text, b.text) >= 0.8
supersede    :=  a.slot == b.slot AND norm_value(a) != norm_value(b)
```

- `J >= 0.8` is the conservative free-text gate: catches near-verbatim repeats (0.86) while every contradiction-shaped pair measured <=0.67. Missed paraphrases are *harmless* — they dedupe at slot level anyway (both facts pin into the same (e,p) observation).
- LSH for candidate generation only: MinHash-128, bands x rows b=16, r=8 -> P(candidate|J=0.8)=0.95, P(J=0.6)=0.24 (rechecked exactly). S-curve: `P = 1 - (1 - J^r)^b`, threshold ~= (1/b)^(1/r) = 0.71. Source: mmd ch.3 / datasketch defaults (num_perm=128, example threshold 0.5 — for long documents; short claims need the higher gate shown).
- **Comparison — vendor analogs:** Hindsight reconciles near-dup observations by embedding cosine >= **0.97** (`HINDSIGHT_API_CONSOLIDATION_DEDUP_THRESHOLD`, default on, 1.0 disables) then an LLM "focused check" before folding evidence sets. Mem0 lets the LLM pick ADD/UPDATE/DELETE/NOOP over top-10 similar memories (repo: `vector_store.search(top_k=10)` in `mem0/memory/main.py`). Zep/Graphiti dedupes edges via LLM restricted to same entity-pair edges. All three use generation where Verbatim must be deterministic — the (e,p)-slot + J>=0.8 gate is the no-LLM equivalent.
- Cosine-on-embeddings is **rejected for the default profile**: Verbatim's blake2b subword hashing encoder is explicitly non-semantic — cosine between hash-vectors does not measure paraphrase. Keep it for the optional neural tier only.

**Cost.** MinHash-128 signature: measured **0.75 ms naive / 0.22 ms optimized** per claim (pure Python, 4-core). LSH insert = 16 band-hash appends. Total write-path <=0.5 ms/claim, background only. Query-time: zero (index built at write).

**Verdict: ship (default, conservative gate).** Slot-keyed merge is free and exact; J>=0.8 catches near-verbatim repeats. Confidence: high for slot-keying; medium for J>=0.8 (my pair set is small — validate against Verbatim's actual claim distribution before locking the constant; the *finding that no pure threshold works* is robust).

## 3. Candidate: supersession chains

**Formula (Graphiti bi-temporal, made deterministic).** Evidence and observations carry `(t_valid, t_invalid)` event-time plus `(t_created, t_expired)` system-time. New evidence f'=(e,p,v',t'):

```
if f'.slot == obs.slot and f'.v != obs.head.v:
    obs_old.t_invalid := f'.t_valid          # close old head's validity
    obs_old.superseded_by := obs_new.id      # forward chain pointer
    obs_new.t_valid := f'.t_valid; head := f'
    # chain stays traversable for "what was true at time T" queries
```

Identical to Graphiti: "invalidates the affected edges by setting their t_invalid to the t_valid of the invalidating edge... prioritizes new information" (Zep paper section 2.2.3). Verbatim replaces Graphiti's LLM contradiction check with slot equality + value difference — strictly weaker detection (misses paraphrased contradictions) but zero-cost, deterministic, and never wrong on the pairs it does catch. Raw facts are never deleted (append-only), consistent with "an LLM may add a fact... never update/delete/overwrite prior facts" — supersession marks *derived* rows, not evidence.

**Worked example.** "deadline is march 15" (t2) -> "deadline moved to march 18" (t9): slot (project, deadline), values differ -> obs[deadline=march 15].t_invalid=t9, superseded_by->obs[march 18]; query "what is the deadline" returns head only; query "deadline in week of t5" walks history.

**Cost:** write-path O(1) per same-slot fact; query-time chain walk <= hop count (measured slot histories are short; cap traversal at depth 5). Storage: +32 B/obs (validity pair + pointer). Latency: negligible, folded into the ~1-2 ms lane cost above.

**Verdict: ship (default).** Confidence: high — bi-temporal is the consensus design (Zep paper, Graphiti `EntityEdge.valid_at/invalid_at` fields in `graphiti_core/edges.py`).

## 4. Freshness semantics (sub-question 2)

Consolidation is a background job, so there is always a window of *unconsolidated* evidence. Options:

| policy | semantics | cost | risk |
|---|---|---|---|
| A. obs watermark flag | `stale(o) := max_ts(evidence[e,p]) > o.as_of` | 0.002 ms/obs (measured, indexed SQLite @100k) | none — flag only |
| B. always-union | emit raw facts AND observations; obs adds proof/freshness metadata | free | raw evidence never hidden |
| C. obs replaces facts | pack prefers obs, drops pinned facts | cheap | **dangerous**: hides evidence, violates auditability |
| D. derived_only flag | obs row marked `derived: true`, must carry pins to be deliverable | free | none |

Hindsight's production answer (docs): "Fresh observations: used directly; **stale observations: agent verifies against current facts before relying on them**" — i.e., A+B. Mem0's update path merges facts into one string (destructive — conflicts with byte-pin).

**Recommendation: A + B + D.** Observation lane always emits; freshness is a per-hit flag `fresh|stale` computed by one indexed `max(ts)` lookup per candidate; the pack includes both the obs row and its pinned evidence (the evidence IS the deliverable). A stale observation should not suppress the newer raw evidence — the flag tells downstream readers the derived view lags. No observation may render as text without a pin (derived_only enforced at pack time).

## 5. Evidence: consolidated/typed retrieval vs raw turns (sub-question 3)

**Honest caveat first:** almost all published numbers are **end-to-end LLM-judged QA accuracy**, not retrieval any@k. Verbatim's metric (any@10 on evidence) has almost no published analog; mnemo's reproduction is the only retrieval-level public number found (recall@1 24.4%, R@5 46.7%, MRR 0.358 on a LongMemEval single-hop slice — far below any vendor claim). Treat vendor numbers as mechanism evidence, not Verbatim predictions.

| system | benchmark | vendor claim | corrected / independent | notes |
|---|---|---|---|---|
| Zep | LoCoMo | 84% -> **75.14% +- 0.17** | Mem0-CTO audit showed cat-5 scoring bug; Zep acknowledged (zep-papers#5) | edge/fact graph |
| Zep | DMR | 94.8% (4-turbo), 98.2% (4o-mini) | full-context baselines 94.4/98.0 — parity | paper itself calls DMR saturated (60 msgs fit in ctx) |
| Zep | LongMemEval_s | +15.2% (4o-mini), +18.5% (4o) vs baseline | none | per-type gains: preference, multi-session, temporal, knowledge-update |
| Mem0 | LoCoMo J | 66.88 overall (SH 67.13 / MH 51.15 / OD 72.93 / T 55.51) | repro issues #3944 (~20% observed), #2800, #4003 open; extraction vs RAG-chunks 67 vs ~61 J (vendor) | ADD/UPDATE/DELETE via LLM — not portable to Verbatim |
| Hindsight | LongMemEval | 83.6 (20B) / 89.0 / 91.4 vs Zep-4o 71.2 | vendor; VT+WaPo are co-authors — "reproduced by collaborators" is not independent | biggest gains: multi-session 21.1->79.7, temporal 31.6->79.7 (20B vs own full-context) |
| Hindsight | LoCoMo | 83.18 / 85.67 / 89.61 vs Memobase 75.78, Zep 75.14 | baselines "as claimed on Backboard leaderboard, not independently reproduced" | largest deltas in open-domain (90.96-95.12 vs ~67-77) — consistent with obs network helping entity/aggregate queries |
| Generative Agents | believability ablation | full 29.89 vs no-reflection 26.88 TrueSkill mu | paper | reflection/consolidation contributes ~3 mu points; qualitative |

**Read for Verbatim:** consolidation wins concentrate exactly where raw turns are weakest — multi-session aggregation, temporal/knowledge-update tracking, entity-centric/open-domain questions. Single-hop verbatim retrieval gains least (Mem0's SH edge over Zep is from dense memories, not consolidation). This matches Verbatim's measured gap: open-domain any@10 0.196 vs BM25 0.370 — the worst category is precisely where obs-style consolidation helps most.

## 6. When consolidation pays (sub-question 4)

| query class | obs benefit | mechanism |
|---|---|---|
| entity-attribute ("what does Alice do") | high | head gives current value in one row |
| evolving attribute ("does user still like X") | high | supersession chain answers directly |
| multi-session aggregation | high | proof_count groups scattered evidence; LongMemEval multi-session 21->79.7 in Hindsight |
| open-domain / "about X" | high | obs is the canonical entry row; Hindsight OD 90+ vs ~72 |
| counting ("how often did X mention Y") | medium | proof_count |
| single-hop "what did X say at T" | low | raw FTS sufficient; obs adds lane noise — keep lane weight <= mid tier |
| verbatim quote | none | byte-pin path unchanged |

**Costs:** ~2.9-10.6 MB/100k index bytes; <=0.5 ms/claim background write-path; +1 lane ~= +1-4 ms query at 100k. Query-time risk is lane noise — mitigate by scoring obs under the existing bounded-boost framework, not as a top-weight lane.

## 7. Provisional constants — verdicts on the listed items under review

- **RRF k=60**: confirm as default — canonical value; Hindsight uses k=60 in paper (but Graphiti *code* uses `rank_const=1` in `graphiti_core/search/search_utils.py:rrf()` — the production system diverged from literature; flag for a c-fusion re-check, not decided here).
- **Boost alphas 0.2/0.2/0.1**: compatible — add `alpha_p` (proof boost) <=0.10 inside same envelope.
- **Cross-encoder pool <=32**: consistent with Hindsight (reranks RRF output with ms-marco-MiniLM-L-6-v2); obs lane adds <=10 candidates to that pool — fine.
- **Entity alias rules / BM25F field weights / PPR hops**: out of scope for consolidation note.

## Summary verdicts

1. Slot-grouped observation rows (e,p)+proof pins -> **ship** — only byte-pin-safe consolidation unit.
2. Slot-keyed merge + J>=0.8 free-text gate (MinHash-128, LSH b=16 r=8 candidate gen) -> **ship** — measured: no pure similarity threshold separates paraphrase from contradiction; structural key is mandatory.
3. Supersession chains via (t_valid, t_invalid) on same-slot value change -> **ship** — deterministic Graphiti bi-temporal.
4. Freshness = watermark flag + always-union + derived_only enforcement -> **ship** — 0.002 ms/check measured.
5. Embedding-cosine merge >=0.97 -> **optional** (neural tier only) — hashing encoder can't measure paraphrase.
6. LLM-synthesized observation text -> **reject** (default) — violates byte-pin; only an optional quality-profile step where every sentence re-pins to source bytes.
7. Obs replacing raw facts in pack -> **reject** — destroys auditability; union instead.
8. proof_count fusion boost (alpha_p*log(1+pc), <=+10%) -> **ship** — inside existing boost budget.
