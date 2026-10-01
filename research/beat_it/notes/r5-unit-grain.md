# V7 unit granularity — session/turn duplication analysis

Sub-question: \"Unit granularity design\" — (a) duplication mechanism, (b) measured cost, (c) literature, (d) recommended unit schema.

## Method / provenance

- Store: `/tmp/v7meas/mem.db` — rebuilt on this VM via the real eval ingest path (`VerbatimArm.ingest` → `Memory.add` → `add_with_retry` → `Ingester.drain_report`) over `eval/v7` `locomo` dev split (`VERBATIM_EVAL_LOCOMO=1`, `corpora.load_corpus(\"locomo\", split=\"dev\")`). Result reproduces the parent's counts exactly: **2877 sources, units = 2877 session + 2877 turn + 1266 sentence_window = 7020, 134 distinct session labels** [measured]. Original path `/tmp/v7-trackr-ezeh8hhn/mem.db` and clone `/workspace/verbatim-new` did not exist on this VM (parent VM \u2260 child VM); I cloned `github.com/mosesman831/verbatim` and rebuilt.
- **Deviation:** `eval/v7/lane_miss_diagnosis.md` is not in the fresh clone (uncommitted on the parent's machine). Analysis relies on the task-briefing summary of its findings.
- Measurements: direct `lane_lexical` invocation on 120 real dev questions (all evidence-bearing tasks: 157 gold turn-units), `LaneSlice(deadline_ms, cap)` varied; `eligible` fenced to turn unit_ids for the turn-only counterfactual — this exercises nomination, scoring, cap, and deadline end-to-end against the unchanged index.

## (a) Duplication mechanism — confirmed in code

1. **Per-item ingest → 1-message source.** `eval/v7/arms.py:579-788` calls `Memory.add(item_document(item), metadata=item_metadata(item))` per LoCoMo dialog item; `item_metadata` carries `session_id=\"{sid}/session_{n}\"` (`eval/v7/arms.py:175-233`). Facade merges metadata into add_args (`verbatim/memory/facade.py` `merged = {**meta, **turn}`).
2. **Explicit session_id mints a session even for a single turn.** `verbatim/projections/units_v7.py:1004`: `sid = t[\"session_hint\"] or args_session` → a `by_sid[sid]` session is created for ANY labeled message regardless of count. The lone-message carve-out at `units_v7.py:1044-1048` (`if not multi: ... sessions = []`) fires only when `args_session is None AND the message has no session_hint`. So all 2877 labeled single-message sources mint a 1-member session. Measured: 2877/2877 session units have exactly 1 child turn via `parent_unit_id` [measured].
3. **Single-member aggregation = byte-identical twin.** Session emission at `units_v7.py:1209-1230` uses `_agg_span` (`min start, max end` over members → the member's own span), `_agg_speaker`, `_agg_persp`, `_covering_occurred` over `idxs=[the one turn]`. Measured on store: **all 2877 session units are byte-identical to their child turn in all five FTS fields** (text, speaker, entities, session, when) [measured].
4. **Every kind lands in the FTS plane.** `verbatim/jobs/units_jobs.py:~227-300` writes `unit_fts_rows`/`unit_fts_content`/`lex_df`/`lex_stats` for all units; only `entity_mentions` and pass-2 events/prefs/state are gated by `if u[\"kind\"] != \"turn\": continue` (`units_jobs.py:299`). Store confirms: `unit_fts_rows` = 7020 (2877 turn / 2877 session / 1266 window); `entity_mentions`=11609, `events_v7`=1248, `state_facts`=156, `preferences`=219 — all turn-only [measured].
5. **No kind fence anywhere downstream.** `_universe` (`verbatim/retrieval/v7/lexical.py:498-530`) selects `rowid, unit_id, source_id, revision FROM units WHERE scope_id=?` — no `kind` predicate. Fusion keys candidates by `unit_id` (`verbatim/retrieval/v7/fusion.py:330-489`) so twins are distinct entries competing for pool slots.
6. **Dedup happens only at pack.** `verbatim/retrieval/v7/pack.py:370-395` `_collapse` folds entries with equal `(_signature(item), lifecycle)`; session/turn twins collapse there — after already consuming nomination, scoring, cap, and fused-limit slots. **Windows never collapse**: `_signature` is normalized full text (`pack.py:346-368`); measured 500/500 windows are strict substrings of their parent turn's text [measured] → distinct signature → delivered as separate near-duplicate items.
7. The authors anticipated this failure mode for episodes — docstring (`units_v7.py:~40`): \"a no-valley session emits no episode units (a 1:1 duplicate of the session unit is noise)\" — but no equivalent rule exists for 1-member sessions.

## (b) Measured cost on the rebuilt dev store

All counts [measured] on `/tmp/v7meas/mem.db`; lane runs over 120 evidence-bearing dev questions (157 gold turn-units).

| Measurement | Value |
|---|---|
| units rows / fts rows | 7020 (2877 turn + 2877 session + 1266 window) |
| kind mix of lane-admitted candidates (cap 2000, t8000) | turn 41.2% / session 41.2% / window 19.2% — ~59% of scored+admitted docs are non-turn |
| session-vs-turn rank within an admitted pair (n=53,477) | identical score in ~100% (median \\|\u0394score\\| = 0.000); ordering is a coin flip on unit_id hash — session ahead exactly 50.0% |
| gold in lex top-200, mixed vs turn-only universe, t8000 | **116/157 → 134/157 (+18, +15.5%)** |
| gold in lex top-200, mixed vs turn-only, t90ms (deadline binds) | **102/157 → 126/157 (+24, +23.5%)** |
| gold in top-800, t90ms | 130/157 → **149/157 (+19)** |
| lane wall-time, cap200 t8000 | mixed: median 68ms / p90 102ms → turn-only: **56ms / 74ms** (-18% median, -27% p90) |
| gold never admitted even at cap2000/t8000 | 4/157 (nomination-miss / true lexical paraphrase gap) |
| avg nominated docs per query | 1232 (~60% non-turn → ~740 wasted doc-scorings \u2248 ~40ms/query at ~54\u00b5s/doc) [calculated] |
| df inflation (text field) | 'on' df=6524 (all) vs ~2877 turn docs; 'painting' 92 vs ~40; '2023' 3303 vs ~1482 — per-term df \u2248×2.2-2.4 while N ×2.44 → idf roughly preserved (\u0394 \u2248 +0.07 for rare terms, \u22480 for common) [calculated] — df inflation is NOT the main harm |
| `when` field degeneracy | '2026-09-23','2026-09','2026','september','wednesday' all df=7020 — `occurred_*` never parsed from the `when` string, so `_when_tokens` (`units_jobs.py:701-754`) falls back to recorded_at_us = ingest wall-clock date on every unit. Any dated query term matches 7020 when-rows with idf\u22480 → pure nomination/postings noise [measured] |
| `[when]` header inside text | corpus date fragments land in the TEXT field: '2023' df=3303, 'pm' df=4670, 'on' df=6524 [measured] |

Interpretation:
- **Cap-slot waste is the dominant mechanism, not score distortion.** Twins score identically (byte-identical fields → identical BM25F); each admitted pair consumes 2 slots for 1 distinct text → effective pool depth \u2248 cap×0.41 distinct turns at cap=200. Turn-only universe recovers +18 gold (+15.5%) at cap200/t8000 and +24 (+23.5%) when the 90ms slice makes scoring truncation also bind.
- **Scoring-time waste \u2248 60% of `_score_candidates` work** (nominated mix measured above): duplicates burn the lane's deadline share before real turns are reached — under a shared 500ms budget (lex slice \u224885-110ms) that converts directly to truncated scoring.
- **Fusion/pack:** both twins survive to fusion (distinct unit_ids → 2 fused slots per text), then `_collapse` removes one at pack — net: waste is upstream-only for sessions, but windows deliver as substring duplicates that `_collapse` cannot see (signature equality only).
- **df is a correctness wart, not the recall driver:** doubling N and df together leaves idf within ~2%; the harm is via `when`-field junk tokens at df=7020 (degenerate — fix separately) and header date-junk in text.

## (c) Literature — retrieval-unit granularity for dialogue memory

Classic IR passage retrieval (the general principle — retrieve at evidence granularity, deliver with context):
- Callan, \"Passage-Level Evidence in Document Retrieval\", SIGIR 1994 — passages of long documents as index units beat whole-doc ranking. https://doi.org/10.5555/188490.188589
- Kaszkiel & Zobel, \"Passage Retrieval Revisited\", SIGIR 1997 — fixed-length arbitrary passages beat whole-document ranking +8% (TREC d2/4) and +18-37% (Federal Register); best uses score the document by its best passage or return passage+context. https://dl.acm.org/doi/10.1145/278459.258561
- Zobel et al., \"Efficient retrieval of partial documents\", IP&M 31(3) 1995 — passage units of fixed size as the retrievable unit.
- Window-passage follow-up (window score vs whole-doc, TREC-5/6): fixed 50-word windows +24% over whole-doc tf-idf; combining window+doc scores did NOT help — the small unit alone carried the signal. https://doi.org/10.1177/0165551014233572

Dialogue-memory systems (unit design in practice):
- LoCoMo benchmark itself (Maharana et al., ACL 2024): gold evidence annotated at **turn level**; its RAG baselines index dialog turns and *generated* session observations/summaries — the coarse unit is a summary, never a concatenation of raw turns. https://aclanthology.org/2024.acl-long.747/ , https://github.com/snap-research/locomo
- SECOM, \"On Memory Construction and Retrieval for Personalized Conversational Agents\", ICLR 2025 (arXiv 2502.05589): directly compares turn-level, session-level, summarization units on **LoCoMo** — \"the granularity of memory unit matters\"; proposes topic-coherent **segment-level** units + compression denoising, beating both turn and session granularity. https://doi.org/10.48550/arxiv.2502.05589
- EMem / \"A Simple Yet Strong Baseline for Long-Term Conversational Memory\" (arXiv 2511.17208): \"retrieval at the granularity of entire sessions or whole rounds often fails to recall fine-grained details while retrieval of turns fails to recover the larger context\" — resolves via **EDU-level index units + entity/concept anchors + Personalized PageRank expansion** for context (i.e., small index unit + graph join, not bigger index unit).
- Generative Agents (Park et al. 2023, arXiv 2304.03442): memory stream = atomic **observations**; higher-level **reflections** synthesized as *separate* memory items — the \"atomic index + derived reflective layer\" pattern.
- MemoryBank (Zhong et al., AAAI 2024, arXiv 2305.10250): per-turn memory items with timestamps; summaries/user portraits live in a derived layer updated asynchronously.
- Mem0 (arXiv 2504.19413): extracted **fact items** (\u03c9_i) from message pairs + async session summary as context — atomic facts, not raw turn duplicates; state of the art on LoCoMo QA.
- COMEDY (COLING 2025, arXiv 2402.11975): compressive memory — session-specific summaries/event recaps/user portraits distilled by LLM; abandons retrieval over raw units entirely.
- Dialogue-RAG / IUR (ACL 2025, aclanthology.org/2025.acl-long.1191): utterance-level units but **rewritten** (ellipsis/coreference completion) before indexing — raw turn text is a noisy index unit; canonicalizing it helps retrieval.
- RAPTOR (ICLR 2024, arXiv 2401.18059): hierarchical tree where parent nodes are *abstractive summaries* of children — supports multi-level retrieval, but the coarse nodes contain different text, never byte-identical concatenations.

Consensus pattern: **index at the smallest evidence-bearing unit (turn/EDU/fact); build context by expansion or summary, never by re-indexing a container's concatenated text.** A concatenated-duplicate unit appears in no published design — it is a defect, not a granularity choice.

## (d) Recommended unit model for Verbatim/LoCoMo

Gold granularity is the **turn** (`dia_id`-level evidence). Multi-turn context must be *joinable*, not *indexed twice*.

| kind | index (FTS/df) | deliver | role |
|---|---|---|---|
| turn | yes | yes | primary evidence unit; matches gold granularity |
| sentence_window | yes | resolve to parent turn at delivery | sub-evidence precision for >3-sentence turns (1266 windows under long turns); classic passage-doc relation — window carries rank signal, parent carries quote/context |
| session | **no (concatenated)** | group header / neighbors context | context container: `_build_groups` headers (`pack.py:522-597`) and the unwired `neighbors_fn` (`pack.py:770-783`) — join, don't index; if a session-summary layer is wanted it must be generated text (LoCoMo's own baselines: observations/summaries), not concatenation |
| episode | same as session | context | dormant today (needs \u22654 turns; all eval sessions have 1 member) |

Concrete changes (cheapest first):
1. **Kind-fence the lexical universe** — `kind IN ('turn','sentence_window')` in `_universe` (`lexical.py:524-530`) or a `retrievable` flag. No reindex needed. [measured: +18 gold @cap200/t8000, +24 @t90ms, -18% median latency]
2. **Don't emit session units for 1-member sessions** — mirror the existing episode no-duplicate rule at `units_v7.py:1044-1048`. Prevents the twin for per-item ingest; needs reindex.
3. **Windows → index-only, deliver-as-parent** (or substring-aware `_collapse`): prevents near-duplicate delivery slots; windows already carry `parent_unit_id` so mapping is free.
4. **Recompute `lex_df`/`lex_stats` over the indexable kinds only** — keeps idf honest; effect small [calculated ~±2%] but free with the reindex in (2).
5. Separately: fix `_when_tokens` degeneracy (parse `occurred_*`/the `when` string at ingest, or drop ingest-date fallback tokens) — removes 5 df=7020 junk terms from every dated query's nomination set [measured].

## Verdicts

1. **Confirmed defect, not design:** per-item ingest + explicit `session_id` mints a byte-identical session twin of every turn (`units_v7.py:1004`, `1209-1230`; lone-message carve-out `1044-1048` only for unlabeled messages). 2877/2877 sessions have exactly 1 child turn and identical FTS fields [measured].
2. **Fence lexical `_universe` to `kind='turn'` (+ optional windows) — highest value, one-line change, no reindex.** Expected: +15-24% gold-in-pool [measured: 116→134 @cap200/t8000; 102→126 @cap200/t90ms; 130→149 @cap800/t90ms], -18% median / -27% p90 lane latency [measured]. ~59% of scored docs are non-turn duplicates [measured].
3. **Emit session units only for len(members)>1** (the episode dedup rule already in the docstring). Expected: eliminates the twin at source for per-item ingest; cost: `derive_units` guard + reindex.
4. **Make windows index-only, delivering their parent turn.** Expected: removes substring near-duplicates from pack slots (windows can't `_collapse`: `pack.py:370-395`); keeps long-turn precision. Cost: delivery-side mapping via existing `parent_unit_id`.
5. **Keep sessions/episodes as structural context, not retrieval units** — wire `neighbors_fn` (`pack.py:770-783`) for same-session neighbor expansion; literature (SECOM, EMem, LoCoMo baselines) supports segment/summary-level coarse units, never concatenations.
6. **Fix `when`-field ingest-date fallback** (`units_jobs.py:701-754`): 5 tokens at df=7020 pollute every dated query's postings [measured]. Expected: modest nomination-budget relief; independent of unit model.
7. Residual gap after unit fix: 4/157 gold are nomination-misses at cap2000/t8000 [measured] — the true paraphrase/lexical-recall problem the diagnosis attributes ~38% of misses to; unit granularity does not touch it (needs dense/fuzzy lanes or rewritten units per Dialogue-RAG).

Expected combined effect on any@10 (calculated estimate): cap-slot waste + deadline-truncation jointly drove ~62% of lane_miss; the unit fix removes the dominant share of slot waste and ~60% of wasted scoring work — plausibly +8-15 points of evidence recall at current caps/deadlines, to be confirmed by a track_r rerun.

ATTACHMENT:{\"url\":\"https://app.devin.ai/attachments/76e5069a-e8c4-4820-bc04-959e2ce2b493/unit_granularity_notes.md\",\"fileSize\":15467}