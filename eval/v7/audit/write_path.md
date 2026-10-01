# V7 Wave Audit — `Memory.add` Write Path (w-aud-write, archived)

`Memory.add` (`verbatim/memory/facade.py:898-908`) signature:
`(content, *, scope, session_id=None, metadata=None, idempotency_key=None, infer=False, replaces=None, source_id=None, change=None)`. V7 §21.02 turn fields (`speaker`, `occurred_at`, message-list form) absent — D7-26.

## Ack tx (one `store.tx()`, ~35+ statements, all atomic)

1. `_require_live`; payload→UTF-8 (≤1 MiB); metadata (≤8 KiB); idem/`replaces` validation.
2. Retention → `protected`; `_bound_ref`; `external_id = hmac(ns‖payload)` (hash #1).
3. In-tx: `governance.authorize(INGEST)`; semantic digest (hash #2) → idem replay/conflict; dedup pre-check; envelope build (SYSTEM_EVENT, speaker_id=owner, per-process session_id); `ingest_envelope` (validate, capture-permission, trust clamp, payload store hash #3, sources+source_revisions INSERT, span INSERT + UPDATE operation_key w/ re-SELECT+re-HMAC hash #4, perspective, source_envelopes receipt, in-tx `screen_content` rules_v1 → label+quarantine, `_link_label` second write, `_enqueue_harvest`, journal event, `mint_receipt`); replay dup handling; protected→2×`open_quarantine`; `record_obligations`; `enqueue_source_jobs` (**unfiltered `SELECT DISTINCT … FROM source_envelopes`** = D7-25); infer=False→`_cancel_harvest`; `ensure_state` bundle (~10 stmts objects/object_revisions/derivations/producer = D7-27); `replaces` CAS; `detect_update_candidates` scan in-tx (D7-13); `_acceptance`; AddResult + idem meta + session receipt; COMMIT.
4. Post-commit: `_compact_session` (>512, chunked pending query), `worker.wake()`, `_receipt_caps` second snapshot (D7-34).
5. Async: managed drain — lane order control>privacy>maint>bg>ordinary; HARVEST→ADMIT per span→SOURCE_PROJECT (derive+prescan+_write_lexical+_write_postings+_write_enrichment)→SOURCE_EMBED. Each job = own lease/commit/settle txs; no coalescing (D7-31).

## T0 enrichment vs V7 §04.1

Fielded FTS: single-column `text` unicode61 only. Canonical entities/aliases: only value-keyed `entity_postings` (BINARY, D7-02). Temporal: one `parse_temporal`, no dual-axis rows. Events: none (events table is ops journal). Preferences/state keys: none. Unit/session segmentation: none (D7-30). Dense: `source_vectors` blob, no matrix; consumer lane skipped (D7-17).

Insertion seams: `_write_lexical`/`_write_fts_pair` (source_jobs.py:342-405) for units+fielded FTS; `_write_postings` (418-454) for canonical entities/aliases; `_derive`/`_anchor_for` (1322-1353) for dual-axis temporal; `_write_enrichment` (457-490) for events/prefs; EPISODE_BUILD-style background job for session segmentation; `handle_source_embed` for matrix.

## Latency (V6 envelope @2048, infer=False)

commit_tx p50 34.8 / p95 66.5. Nested: update_prescan 11.3/23.9, source_state 5.1/11.1, source_jobs 5.4/8.3, ingest 4.8/7.9, session_receipt 1.2/2.5. Scaling ×4 corpus: commit_tx p95 ×2.9, source_state ×10.8, source_jobs ×4.1, update_prescan ×2.3. B1 ≤30 ms path: remove in-tx update scan, O(1) enqueue, slim ensure_state, dedup record_obligations, screening pre-tx.

## require_review chain (D7-20 + new D7-29)

Default `require_review=True`: every harvested proposal → PENDING + review (`policy.py:838-843`); explicit-remember/whitelist branches unreachable. Claims never ACTIVE → invisible to typed lanes; source-side `source_lexical_ready` still settles honestly. New D7-29: admit follow-ons (relate/supersession/unstructured scans over pending+active+disputed) run for PENDING claims — write-path amplifier scaling with pending backlog.

## New defects (D7-25+)

- D7-25 O(envelopes) `_receipt_target` unfiltered DISTINCT scan per add (HIGH)
- D7-26 no turn-shaped add (speaker/occurred_at/message-list/bulk) (HIGH)
- D7-27 ack tx carries work outside whitelist: screening regex in-tx, _link_label re-read, spans operation_key update, enqueue-then-cancel, ensure_state doc/edge writes (HIGH)
- D7-28 4× SHA-256 + 2× decode + regex on ≤1 MiB payload mostly in-lock (MED)
- D7-29 admit follow-on relation passes for PENDING claims (MED)
- D7-30 no durable units/sessions written on write plane (MED)
- D7-31 job-per-tx drain, no coalescing, bulk import unreachable (MED)
- D7-32 `writers_waiting` marks process-local — cross-process drainer sees nothing (LOW-MED)
- D7-33 two post-delivery write txs per search (exposure + causal mint) (LOW-MED)
- D7-34 post-commit `_receipt_caps` snapshot per add (LOW)
