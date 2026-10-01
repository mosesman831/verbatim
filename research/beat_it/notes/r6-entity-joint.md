# Entity-joint retrieval for multi-entity questions — research note (subagent r-note)

Scope: (a) what posting structures exist for speaker×content / entity-pair queries; (b) literature on entity-aware fielded scoring vs joint postings; (c) how Zep/Graphiti/Hindsight answer \"both X and Y\"; (d) a concrete, implementable recipe with measured costs. All code refs are to `mosesman831/verbatim` @ 892f2b8 (`/workspace/verbatim-new`). All [measured] numbers are from a synthetic LoCoMo-shaped store I built through the real write path (`project_units_v7`): 2,900 sources → 5,810 units (2,900 turn + 2,900 session + 10 sentence_window — confirms the \"session+turn duplicate per add\" finding), 4,387 `entity_mentions` rows, 36 canons, warm sqlite3 page cache. Extrapolations are marked [calculated].

## (a) Available posting structures — schema inventory

1. **`entity_mentions`** — the per-canon postings table. PK `(scope_id, canon, unit_id, byte_start, generation)` (`schema_v7.py:344-352`); `role` CHECK includes `subject/object/speaker/mention` (`schema_v7.py:350`). **Write side gates on turn units only**: `if u[\"kind\"] != \"turn\": continue` before the INSERT loop (`units_jobs.py:299-318`). So the ent lane never emits session/window units — correct for turn-granularity gold, but joint-evidence *session* units can't come via this lane at all.
2. **`units.speaker_canon` + `idx_units_speaker (scope_id, generation, speaker_canon, unit_id)`** (`schema_v7.py:235-236`) — a true speaker→units posting index. Measured: 145 turns/speaker, ~1.1ms cold COUNT [measured]. This is the speaker×content primitive.
3. **`unit_fts` fielded FTS5** — columns `(text, speaker, entities, session, \"when\")` (`schema_v7.py:623-625`). The `entities` field is written for **every** unit kind, not just turns (`units_jobs.py:263-281`: `ent_field` is set before the turn-only gate). Token-level, so it covers entity tokens *inside* merged multi-word canons (bug below). Fielded MATCH works: `entities : \"gina\"`, `speaker : \"jon\"`, `text : \"destress\"` all resolve [measured].
4. **`graph_edges (scope_id, src_unit, type, dst_unit)`** — exists in DDL but **empty and producers are orphan code** (parent's r4 finding; confirmed 0 rows [measured]). Not usable today.
5. **`entity_aliases_v7` co-mention pairs** — `derive_aliases` computes co-mention sets (`entities_v2.py:551-558`) but only to support alias rules A1–A6, not retrieval. `expand_query` is **bidirectional over ACTIVE pairs** (`entities_v2.py:689-709`) — the existing mitigation for the merge bug below.
6. **No dedicated entity-pair posting table exists.** Closest mechanisms are the em self-join intersection and the FTS entities-field AND — both measured below.

**Speaker-role pin nuance:** `extract_mentions` adds `(canon(speaker), 0, 0, SPEAKER)` (`entities_v2.py:298-308`); `seen_pair` dedups on `(canon, byte_start)` (`units_jobs.py:304-313`). In eval render `[when] Speaker: text`, byte 0 is `[`, so the pin survives: `role='speaker'` rows DO exist [verified: `('jon','speaker',0,0)`]. So `entity_mentions` can distinguish \"X said\" from \"X mentioned\" via role — but `units.speaker_canon`/`fts.speaker` is the more direct index.

## Measured costs (synthetic store, warm cache)

| primitive | SQL shape | result | time |
|---|---|---|---|
| em union `canon IN (jon,gina)` (current lane) | `SELECT unit_id,canon ... WHERE canon IN (?,?)` | 380 rows / 290 units | 0.136-0.143ms |
| em **pair intersection** | `... GROUP BY unit_id HAVING COUNT(DISTINCT canon)=2` | 78 units | 0.150ms |
| intersection + units join (eligibility) | join `units u` + gen fence | 78 | 0.236ms |
| FTS `entities:jon AND entities:gina` | `unit_fts MATCH 'entities : \"jon\" AND entities : \"gina\"'` | 157 (78 turn + 78 session-dupe + 1 window) | 0.072ms |
| FTS `speaker:jon AND text:destress` | fielded AND | 24 | 0.035ms |
| FTS `(speaker:jean OR speaker:john) AND entities:paris` | \"both visited Paris\" restrictor | 50 | 0.062ms |
| `idx_units_speaker` count | `speaker_canon=? AND kind='turn'` | 145 turns | ~1.1ms cold |
| unfielded junk match | `MATCH 'i'` / `'weekends'` | 5,805 rows | ~1.2ms each |

[all measured]. Pair reduction union→intersection: 290→78 ≈ **3.7×** at df~185/canon. At real scale (speaker df ≈ 300–600, union ≈ 600–1,200 rows) intersection stays sub-ms — index-bounded, ~\u03a3df rows read [calculated]. Split evidence for \"Which city have both Jean and John visited?\": `speaker(jean)+entity(paris)`=14 turns, `speaker(john)+entity(paris)`=11 turns, ~0.1ms each [measured] — gold is *split* evidence joinable on the city entity, not speaker co-mentions.

## The actual failure mechanism (sharper than \"broad flood\")

Three compounding, all in current code:

1. **Canon-sorted scan + deadline = co-mention blindness.** The scan is `ORDER BY m.canon, m.unit_id, ...` (`entity.py:498-511`). A deadline/row-limit cut mid-scan has seen all of the alphabetically-first canon's postings and little of the second's — joint units then score as single-canon because `matched`/`qcov` are computed from *rows seen so far* (`entity.py:617-624`). Measured: at a 100/380-row cut, **0/78** joint units keep both canons; at 200/380, **3/78** [measured]. Under deadline pressure the `multi_canon_boost` is precisely the first casualty — the worst possible degradation order.
2. **Facet starvation.** `pipeline.py:877-912`: facets rerun the lane with `cap=slice_.cap`, results are **appended after** whole-query candidates then truncated to `slice_.cap` (`merged[:capv]` at 904-906). Whenever the whole-query lane emits ≥cap candidates — every 2-entity question over speaker canons — *zero* facet candidates survive, `signals[\"facet\"]` never reaches fusion, and `facet_bonus/v1` (`fusion.py:42-56`, 104-112, 379, 422-432) can never fire. Per-entity coverage is structurally dead exactly when needed.
3. **Cap-run merge across ':' steals postings and mints junk canons.** `_cap_spans` merges capitalized runs not crossing `_SENT_BOUNDARY = {\".!?\
\"}` — `:` is **not** a boundary (`entities_v2.py:181`, 204-253). Render prefix `Jon: Gina went...` produces canon `jon gina`; the turn lands on neither `jon`'s text postings nor `gina`'s at all — mentions = `{jon gina:mention, jon:speaker}`, **no `gina` row** [measured]. A1 aliases (`('jon gina','jon','A1','active')` auto-emitted [measured]) + bidirectional `expand_query` patch the read side only when the alias job ran. This fragments `df_units` bookkeeping and inflates `entity_canon` (part of the 1,513-entry vocab) with `jon gina`/`gina my`-class merges — measurable on the real store: `SELECT COUNT(*) FROM entity_canon WHERE canon LIKE '% %'`.
4. (Context) `_when_tokens` emits ISO-day+month+year+month-name+weekday per unit (`units_jobs.py:701-754`) → `november`,`2023`,`tuesday` at df=5,810 = every unit [measured] — the `df=7,020 'wednesday'` finding reproduced.

## (b) Literature — three design points on the axis

- **BM25F fielded scoring** (entities as a weightable field): Robertson, Zaragoza & Taylor, CIKM'04 + Zaragoza et al. TREC-13 variant C (per-field `b_f`); covered in `notes/b1-bm25-bm25f.md`. Verbatim's bm25f/v1 (`entities 0.8 / speaker 0.3`) is this design — it **soft-boosts** entity-bearing units, never restricts to joint coverage.
- **Joint entity\u2229term postings**: textbook conjunctive retrieval (Manning/Raghavan/Sch\u00fctze IIR ch.2, postings merge join). In FTS5 this is `entities:a AND entities:b` or the `entity_mentions` self-join — no new index needed [measured above].
- **Entity-pair posting lists (materialized)**: the E-R-retrieval line — Saleiro, Mili\u0107-Frayling, Rodrigues, Soares, *RELink* (arXiv 1706.03960): entity index + **entity-pair relationship index** over ClueWeb-09-B (476M entity / 418M pair extractions, sentence-level pairs); *Early Fusion for E-R Retrieval* (arXiv 1707.09075): per-entity-pair meta-documents as first-class retrieval targets (4.1M entities, 71.7M pairs). EQFE (Dalton/Dietz/Allan SIGIR'14, cited in c4) is the related entity-anchored-expansion evidence. At Verbatim's scale a pair table is unnecessary — self-join is already ~0.15ms.
- **Lit takeaway:** hard intersection is the right primitive for *relational* questions (\"what did Jon tell Gina\"), but \"both X and Y\" *comparison* questions need split-evidence joins — E-R retrieval joins on the *answer* entity (the city), not co-mentions of the subjects. A pure co-mention restrictor is the wrong cut for the parent's second example.

## (c) Zep / Graphiti / Hindsight on \"both X and Y\"

- **Zep = Graphiti** (hosted product over the same library). `graphiti.search()` = hybrid (BM25 + semantic) over edges/nodes/communities, RRF-fused (default), optional `node_distance` rerank anchored to a **single** `focal_node_uuid`, or `cross_encoder`/`mmr` (help.getzep.com/graphiti/working-with-data/searching.mdx, help.getzep.com/sdk-reference/graph/search.mdx) [vendor]. `connected_node_uuids` filter is **source-OR-target union**, not intersection [vendor]. **No entity-pair intersection mechanism** — \"both X and Y\" is answered by union breadth + reranker: exactly Verbatim's current shape.
- **Hindsight (TEMPR)**: 4 arms + unweighted RRF k=60 (d1). Graph arm scores `tanh(0.5 · COUNT(DISTINCT shared entity_id))` via `unit_entities` — a **co-mention-count boost**, the closest published mechanism to what's needed, but a boost, not a restriction; `entity_cooccurrences` exists for entity resolution, not retrieval (d1).
- **Net:** neither does conjunctive entity postings. Verbatim's `entity_mentions` intersection is a capability they lack; the failure is Verbatim's ordering/capacity, not the concept.

## (d) Recipe EJ — implementable entity-joint retrieval

Trigger: `len(query entity canons) >= 2` after `expand_query` — already computed as `all_canons`/`canon_to_q` in `lane_entity` (`entity.py:478-493`); both example queries extract {2 canons}, classify comparison/multi_hop, and decompose into 2 per-entity facets [measured].

**EJ-1 — Intersection-first ordering in `lane_entity` (the core fix).** Before the union scan, one bounded pass:
```sql
SELECT m.unit_id FROM entity_mentions m
 WHERE m.scope_id=? AND m.generation<=? AND m.canon IN (<all_canons>)
 GROUP BY m.unit_id HAVING COUNT(DISTINCT m.canon)>=2
```
then keep only units whose matched set maps to ≥2 distinct *query* canons via `canon_to_q` (same semantics as `multi_canon_boost` at `entity.py:621-623`, just ordered first). Emit them tagged `signals[\"joint\"]=qcov`, then run the existing union scan for remaining slice budget. Cost: +0.15–0.5ms [measured, ~\u03a3df]. Under deadline, joint units are guaranteed complete before any singles are scanned — inverts the worst-first degradation order.

**EJ-2 — FTS-entities variant (zero new schema, broader coverage).** `unit_fts MATCH 'entities : \"c1\" AND entities : \"c2\"'` — token-level, covers units where an entity survives only inside a merged canon (`entities:gina` matched the `jon gina` unit that `entity_mentions` misses [measured]); emits turn+session+window rows (dedup/prefer turn). 0.072ms [measured]. Ship alongside or instead of EJ-1.

**EJ-3 — Facet slot reservation (required for comparison questions).** In `pipeline.py:894-906`, replace append-then-truncate with reserved per-facet slots (e.g. `floor(cap/(1+n_facets))` each, interleave before capping) so `facet_bonus/v1` can actually fire. Zero new index. This is where the 'Which city have both Jean and John visited?' split evidence lives — measured 14/11 turns per speaker-facet at ~0.1ms.

**EJ-4 — Speaker-pair restriction for speaker-resolved canons.** When `resolve_query_speaker_canons` (`entity.py:391-422`, already feeding rerank `speaker_match` w=0.4, `rerank_features.py:91,380,564-580`) resolves query canons to `speaker_canon`s, probe `speaker IN (q_speakers)` \u2229 content terms — \"how does X destress\" evidence is authored *by* X. `speaker:jon AND text:destress` = 0.035ms [measured].

**EJ-5 — Write-path fix (separate bug).** Add `:`/`;` to `_SENT_BOUNDARY` (`entities_v2.py:181`) or strip the `[when] Speaker:` prefix pre-extraction — kills `jon gina`-class merges, restores per-canon postings, shrinks the junk share of the 1,513-entry vocab. One-line-ish; re-ingest to compare.

**Not recommended now:** materialized `entity_pairs` table (RELink design) — self-join is already sub-ms; revisit only at ≥10× df growth or for pair-level byte-proximity scoring.

## Cost/lift model

- EJ-1+EJ-2 add ~0.1–0.5ms to an ~85ms slice (LANE_COSTS_V1 ENT=2.0 nominal `policy.py:153-161`; real ~200ms per parent) — noise; the union scan can even stop early once `cap` joint units are emitted.
- Lift on cat1/cat4 (reasoned — needs a real-mem.db run): ent emission collapses from ~600–1,200-row flood to ~tens of joint units; joint-evidence survival under deadline goes from ~0–4% (measured 3/78 at half-scan) to ~100% for co-mention gold. For the measured 62% cap/deadline share of lane_miss this is the direct entity-lane fix; for the 38% paraphrase-gap share it is neutral. **Honest bound:** \"both\"-style gold *split* across two single-entity turns is covered only via EJ-3, not EJ-1 — ship both.
- Files touched (proposal only — no edits made): `entity.py` (scan order + joint pass), `pipeline.py:894-906` (facet merge), `entities_v2.py:181` (boundary, optional).

## Verdicts

- **V1 (ship, primary fix):** Intersection-first in `lane_entity` — emit `qcov>=2` units before the union scan; covers relational multi-entity questions; kills canon-sorted deadline fragility. ~+0.15ms measured; joint-evidence survival → ~100% under deadline (vs 0–4% measured mid-scan). Cat1 relational subset.
- **V2 (ship, cheap complement):** FTS `entities:c1 AND entities:c2` probe — broader recall than `entity_mentions` (merged canons, all unit kinds); 0.072ms measured.
- **V3 (ship, required for comparison):** Reserve per-facet slots in the facet merge (`pipeline.py:894-906`); current append+truncate starves facet candidates whenever whole-query output ≥ cap — zero index cost, expected lift on cat4 \"both X and Y\" split-evidence items.
- **V4 (ship, small):** Speaker-pair probe when canons resolve to speakers — 0.035ms measured; cat4 precision.
- **V5 (ship as own fix):** `_cap_spans` merge across `:` mints `jon gina`-class junk canons and steals per-canon postings — add `:` to `_SENT_BOUNDARY`; shrinks entity_canon vocab, restores df integrity.
- **V6 (defer):** Materialized `entity_pairs` posting table — unneeded at this scale.
- **Vendor context:** Graphiti/Zep and Hindsight have no conjunctive entity-pair retrieval — V1 gives Verbatim a capability neither baseline has.

## Appendix — implementation details for the reviewer

- **EJ-1 SQL nuance:** `HAVING COUNT(DISTINCT m.canon)>=2` counts *expanded* canons; two aliases of one query canon \u2260 joint coverage. Keep the SQL as a superset filter, then map matched canons→query canons via `canon_to_q` (`entity.py:289-308`) in the accumulation loop — same `qcov` semantics as the existing boost.
- **Wire-in:** (i) same lane output with joint units first + `signals[\"joint\"]` (lowest risk); (ii) separate `ent-joint` pseudo-lane for independent RRF mass (needs policy row + LANE_COSTS entry).
- **Facet granularity verified on the real analyzer:** 'How do Jon and Gina both like to destress?' → `{jon,destress}` / `{gina,both,like,destress}`; 'Which city have both Jean and John visited?' → `{city,have,both,jean}` / `{john,visited}` [measured] — facet ent calls get correct single-canon postings; they just die in `merged[:capv]` today.
- **Cheap checks for the real `mem.db`:** (1) `SELECT COUNT(*) FROM entity_canon WHERE canon LIKE '% %'` (junk merge share); (2) em pair-intersection size on top speaker pairs; (3) `SELECT COUNT(*) FROM entity_mentions WHERE role='speaker'` (pin survival); (4) `SELECT canon,COUNT(*) FROM entity_mentions WHERE canon LIKE '% %' ... LIMIT 20` (worst merges); (5) one gold multi-entity question with `t_lane`+`cap_truncated`/`deadline`/`facets` stats to see whether V1 ordering or V3 starvation dominates.
- **Provenance:** all ms/row counts [measured] on `/tmp/entq/mem.db` (2,900 sources via `project_units_v7`, generation 1) unless tagged [code] (file:line @892f2b8), [paper] (arXiv links), [vendor] (help.getzep.com), [calculated] (extrapolation to parent's df/cap regime). No edits under `verbatim/`; scratch only in `/tmp/entq/`.