# Audit: schema/storage gap map (w-aud-schema, agent 5df2fb30)

## CRITICAL — table name collisions (D7-25a)
- `events` EXISTS as V1 audit spine (schema.py:205-215, event_seq AUTOINCREMENT, recall.py:896 reads MAX(event_seq)). CREATE IF NOT EXISTS silently no-ops → V1 shape survives. RENAME to `unit_events`.
- `entity_aliases` EXISTS keyed (entity_id, normalized_alias) (schema.py:98-104; purge.py:893, export.py:522/1015 pin V1 shape). RENAME to `entity_aliases_v7`.

## Lazy-additive recipe (clone exactly)
- DDL_V5_ADDITIVE + ensure_additive_tables + _ensure_v6_additive pattern (schema_v5.py:380-450, migrations.py:820-879). Per-statement executes in caller tx, NEVER executescript in-tx. Deferred phase at migrations.py:518 → add `_ensure_v7_additive` recorded as "v7-additive-ensure". SCHEMA_VERSION stays 5. Store.create untouched.
- FTS trio discipline: carrier (INTEGER rowid + scope/generation fence) + content + VT + ai/ad/au triggers; DELETE-then-INSERT never REPLACE (source_jobs.py:342-378); carrier+content written unconditionally even when FTS5 absent, lane reports unavailable; probe trigger existence not just table; trigram needs SQLite≥3.34 separate probe (provider.py:235).
- units can't be unit_fts content table (TEXT pk, holds byte offsets) — needs its own carrier+content.

## Generation fence is `<=` not `=`
- source_lane.py:168-187 — lifecycle bumps meta.projection_generation without rewriting rows. lex_stats/lex_df reads take MAX(generation)≤snapshot per key; update_stats writes at fence generation; purge must decrement/rebuild (stale df = aggregate leak).

## Closure/quarantine allowlists (new tables invisible until registered — integration wave)
- derivations.OBJECT_KINDS (:78) needs 'unit' kind; deletion._RESOLVABLE_KINDS/_SCOPE_SIMPLE/_DATED_TABLES; closure._SIDE_WALKERS/_MEMBER_TABLES; purge.OBJECT_KINDS+erasers; candidates._quarantine_cascade; consumer _sweep_derived/_V5_SOURCE_TABLES (controls.py:1832-1849); check_integrity _COUNT_TABLES; export.
- V7 units need covering cascade: ('source',sid,rev)/('source_envelope',env,rev) hold withholds unit+children; new ('unit',uid,rev) granular kind.
- screening_log/t2_facts/run_manifests lack scope_id — add column or derivation path.
- Missing generation column on several §30 tables per spec — add to ALL (V7-30.02).
- run_manifests: eval-only, consider profiles/schema.py module-owned precedent instead of DDL_V7.
- V7-30.04 migration cursor: reuse backfill_cursor (schema_v5.py:227, job_key='v7_migrate:<scope>').
- Store flag or per-probe v7_tables_present for FTS absence.
