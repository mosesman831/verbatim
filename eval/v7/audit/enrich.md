# Audit: enrichment / query analysis / coreference / sessions (w-aud-enrich, agent f7053595)

## Capability map (existing vs V7)
- identifiers.py: exact-identifier regexes + byte offsets REUSABLE; entity extraction = capitalized runs + dict only (isupper gate :397) — mention extractor, not V7 entity pipeline. No canon fold, no lowercase/vocab lookup, no alias rules.
- normalize.py (norm/v1): fold pipeline + utf8_offsets REUSABLE; no clitic split, no identifier channel, no stem/trigram channels, no strict UTF-8 decode.
- temporal.py (temporal/v1): deterministic anchored skeleton REUSABLE; missing V7 rule families (seasons, approximates, embedded relative, early/mid/late, durations), anchor chain message_at∨session_started_at∨recorded_at, occurred_source.
- typing.py: marker classifier, no span pins/subject/object — seed lexicons only.
- polarity.py: quoted>hypothetical>hedge>negated>affirmative — reusable exclusion signal; no clause scoping (supersession._hedged has clause windows).
- grounding.py: pin verification reusable for T2; T2 extraction explicitly not implemented.
- dedup_sig.py: 3-token shingles reusable for duplicate links.
- evidence/entities.py: ambiguity-preserving scoped resolution + pronoun vocab REUSABLE; needs caller-supplied context — no session-window derivation, no agreement rules.
- querying/analyze.py: query_analysis/v1 classes (NO_ANSWER_LIKELY/TEMPORAL/IDENTIFIER/PREFERENCE/PROCEDURAL/ENTITY/FACTUAL); no IntentClass taxonomy, no canon vocab lookup, relative time marked not resolved.
- retrieval/query.py: ambiguous_time → drops filter (D7-18 confirmed mechanism).
- typed_lane.py: entity_postings byte-exact never folded (D7-41); LIKE token prefilter (D7-08).
- lanes.py: exact_id lane, lane_graph (off by default), probe/compose primitives on claim plane.
- Sessions: facade _session_id is transport-level; working.py session working sets; episodes group by host task; identity.py conversation_id is AUTH partition (D7-42 — don't conflate with retrieval grouping). No units/adjacency/neighbor expansion.

## Integration seams
- Write: jobs/source_jobs.py::_derive (~1342-1356) is where V7 T0 slots in; today one record-level bundle per (source_id, revision) — needs per-unit granularity.
- Read: querying/analyze.py + typed_lane/entity_postings.

## New defect candidates D7-25..D7-42 (full detail in agent report; summary)
- D7-25 norm/v1 not norm/v2 (no clitic/identifier-channel/stem/trigram)
- D7-26 entity extraction capitalization-dependent, no canon fold
- D7-27 legacy entity_aliases keyed by entity_id — incompatible with (scope,canon,alias_canon)
- D7-28 no IntentClass taxonomy / canon vocab lookup in query analysis
- D7-29 temporal/v1 missing rule families + anchor policy + occurred_source
- D7-30 no structured event extraction (markers only)
- D7-31 no state-key extraction (typing markers only)
- D7-32 no structured preference extraction
- D7-33 no coreference sieve (entities.py needs caller context; harvest only local-links)
- D7-34 no conversation units/session grouping/neighbor retrieval (extends D7-21)
- D7-35 T2 extraction unimplemented (grounding.py contract)
- D7-36 grounding can't represent V7 fact semantics (subject/predicate/state_key)
- D7-37 update detection synchronous inside write commit
- D7-38 projection is record-level bundle — wrong granularity for per-unit rows
- D7-39 analyze.py relative time marks RELATIVE but stores no interval
- D7-40 speaker on envelopes never canonically indexed (extends D7-19)
- D7-41 entity_postings exact-value match, no canon bridge
- D7-42 conversation_id is auth scope partition — must not be conflated with retrieval grouping
