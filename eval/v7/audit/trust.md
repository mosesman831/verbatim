# Audit: verdict/pack/security/mutation (w-aud-trust, agent 47ff23dd)

## Verdict
- 50% literal floor confirmed verdict.py:63-67 + ~309-385 — AND duplicated in governed abstain.py:246-270 (D7-31: removing from one path only → cross-API disagreement).
- No `weak` label / `strict` / `missing` field on results (types_v3 GroupSupport = SUPPORTED|PARTIAL|INSUFFICIENT only).
- Identifier match byte-exact; MissingDescriptor declared but unproduced (verdict detail has term COUNTS not which terms).
- Calibration keyed encoder+class not (profile,tier,CE) matrix; no calibration=unfitted surface.

## Pack
- Token accounting = serialized_wire_bytes//4 including JSON metadata (pack.py:2193-2203, package.py:111-117) — switching to tok/v1 shifts admission/goldens (D7-34).
- No never-empty/skip-long-continue; collides with atomic-group invariant — needs explicit escape (D7-35).
- Atomic conflict groups strong today (pack.py:1468-1559, 2021-2073); no conflict_unresolved surface / reader CONFLICTS section.
- No ±N neighbor expansion, no session headers, no reader view.

## Security
- rules_v1: content_form×attack_risk axes, rules_v1:2025-01 pinned, deterministic.
- D7-27: v2 _gate discards attach_label id → orphan labels (ingest.py:955-969) vs envelope path links (envelopes.py:473-495).
- D7-28: envelope vs source_envelope kind drift between purge pairs (purge.py:1320-1321) and retrieval holds (union.py:601).
- D7-36: no retrieval-time rescreen vs rules_revision; no rescan JobKind — sleeper-rule upgrades leave stale labels live (structural).
- D7-37: no assembly-level sliding-window screening.
- D7-38: TrustClass 7-value vocab vs V7 six labels; consumer Hit carries no trust.
- D7-43 (verify): direct ('claim',id,rev) quarantine holds not consulted by union cascade.

## Closure/purge — FLAGSHIP FINDINGS
- **D7-25 confirmed**: source_fts/source_fts_rows/source_fts_idx deleted ONLY by consumer forget sweep (controls.py:1821-1831) — purge.py/closure.py have ZERO source_fts refs → purged source text stays searchable. Realized instance of the class V7-19.09 prevents.
- **D7-26 spec↔design conflict**: source_exposure has no redacted column (schema_v5.py:373-379 deliberately countable post-purge); influence rows UPDATE redacted=1 never deleted; V7-19.09 lists exposure rows in closure scope — needs explicit V7 decision.
- D7-29: journal failure policy split — claim lane hard-fails INTEGRITY, source lane warns.
- D7-30: minted `search:{new_id()}` receipts never journaled → unjoinable receipt_ids.
- D7-41: post-read journal writes must be whitelisted in cache-settle watchlists (cache.py:~99) or every search busts cache.
- New derived tables need THREE registries: closure traversal + purge scrub tuples + consumer _sweep_derived/_V5_SOURCE_TABLES + verify + receipts + mutation anchor.

## Mutation conventions
- yaml: id/check/file/anchor(exact string, occurs exactly once)/replacement/killed_by(pytest ids)/requirements; static gate ≥18 mutations, anchor!=replacement; restore+sha256 check; worktree byte-identical.
- MUT-10 lesson generalized: anchors follow refactors in same commit; killer test must exercise anchored path.
