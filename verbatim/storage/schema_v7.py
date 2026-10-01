"""V7 schema (SPEC_V7 §30, docs/v7_contracts.md — wave A, worker w-schema).

Additive, digest-safe, lazy: every table here rides NEITHER the migration
statements nor any ``_creation_ddl_*`` string — those byte strings are hashed
into recorded ``migration_history`` digests and ``_verify_history`` refuses
mutation on drift (V4-41.03). Instead the V6 additive pattern applies:

* ``ensure_v7_additive(conn)`` — the lazy twin for ``Store.create``-fresh
  stores (which never run ``apply()``): the first V7 writer invokes it inside
  its own write transaction so the tables and the rows that need them commit
  atomically. Statements execute one by one — NEVER ``executescript``, which
  would implicitly COMMIT the caller's transaction.
* ``v7_statements()`` — the same DDL as an ordered tuple for the deferred
  ``apply()`` ensure phase (same family as ``_ensure_v6_additive``; the
  migrations-side phase itself is main-session integration, not this file).

Generation fencing (V7-30.02): every table carries ``generation`` stamped
with the committed ``meta.projection_generation`` at write time; readers
filter ``generation = current``. A projection rebuild therefore writes the
new generation BESIDE the live one and flips the fence atomically — which is
why natural-keyed derived tables append ``generation`` to their primary key
(same source re-derived at a new generation reuses its natural key; the
fence requires coexistence until the flip). The §30 normative keys remain
leading prefixes of every PK, so ``WHERE <§30 key prefix>`` lookups are
unchanged. Minted-id tables (events_v7, observations_v7, standing_rules,
standing_queries, t2_facts, screening_log, run_manifests) keep the §30
single-column key — their ids mint per generation — with ``generation`` as a
plain column.

Name collisions (§30 names vs. existing tables): ``events`` already exists
(v1 audit journal — ``event_seq`` PK, incompatible shape) and
``entity_aliases`` already exists (v1 span-pinned approval table). An
``IF NOT EXISTS`` create under those names would silently adopt the wrong
shape and break every downstream worker. Following the spec's own collision
convention (``observations`` exists in schema_v3 → §30 ships
``observations_v7``), the V7 event calendar is ``events_v7`` and the V7
alias table is ``entity_aliases_v7``. ``V7_TABLE_RENAMES`` records the
§30-name → physical-name mapping.

Fielded FTS5 (V7-06.01/02/04): three virtual tables share ONE external
content table, mirroring the v5 three-piece shape
(``source_fts_rows``/``source_fts``/``source_fts_idx``):

* ``unit_fts_rows`` — rowid carrier; eligibility metadata (unit_id,
  scope_id, generation) lives here so MATCH results join once.
  ``UNIQUE(unit_id, generation)`` keeps one indexed row per unit per
  generation (``source_fts_rows`` convention).
* ``unit_fts_content`` — the external content table; the five fielded
  columns (text, speaker, entities, session, ``"when"``) ride here.
  ``REFERENCES unit_fts_rows(row_id)`` keeps the pair together.
* ``unit_fts`` / ``unit_fts_stem`` / ``unit_fts_tri`` — the §30-named FTS5
  virtual tables (MATCH targets): unicode61+diacritics-folded fielded index,
  Porter-stemmed shadow, trigram fuzzy/substring index. All three read
  ``content='unit_fts_content'``; nine triggers (ai/ad/au per index) mirror
  the content table into each index. Per-index trigger sets keep a missing
  tokenizer from breaking the other indexes — an absent trigram index
  degrades the fuzzy lane to ``unavailable`` instead of failing inserts.

Reprojection never UPDATEs the carrier rowid (v5 rule carried): stale rows
are deleted content-first then re-inserted, so every index mutation passes
through the triggers (REPLACE on the carrier would mint a new rowid without
firing the delete trigger, stranding index terms).

Capability gating: each virtual table is created only when a live probe of
its tokenizer succeeds (``fts5_tokenizer_available``) — a build without the
``trigram``/``porter`` tokenizers still gains every plain table and reports
the affected lane unavailable, never a failed open. Probes create+drop a
``temp`` FTS5 table; results cache per process (a build property).

Covering indexes (V7-30.03, asserted by EXPLAIN QUERY PLAN tests — see
tests/storage/test_schema_v7.py):

* ``idx_units_scope_gen (scope_id, generation, unit_id)`` — the eligibility
  fence: every lane filters ``scope_id=? AND generation=?``; covering for
  unit-id listing.
* ``idx_units_session (scope_id, generation, session_id, seq, unit_id)`` —
  neighbor-window expansion (±N turns inside a session, V7-12.05) and
  chronological session packing (V7-12.06).
* ``idx_units_occurred (scope_id, generation, occurred_start_us,
  occurred_end_us, unit_id)`` — temporal-lane interval overlap
  (V7-09.06).
* ``idx_units_recorded (scope_id, generation, recorded_at_us, unit_id)`` —
  the recorded axis, secondary temporal retrieval (V7-09.06).
* ``idx_units_speaker (scope_id, generation, speaker_canon, unit_id)`` —
  speaker-scoped questions ("what did X say", V7-08.05).
* ``idx_units_source (source_id, revision)`` / ``idx_units_parent
  (parent_unit_id, generation)`` — per-source re-derivation (V7-30.01),
  deletion closure, and child→parent expansion (sentence_window → turn).
* ``entity_mentions`` PK ``(scope_id, canon, unit_id, generation)`` IS the
  covering postings index (L-ent: ``scope+canon+gen`` → unit_ids, ≤ 2 ms).
  ``idx_entity_mentions_unit (unit_id)`` — closure by unit.
* ``entity_aliases_v7`` PK serves canon→alias expansion;
  ``idx_entity_aliases_v7_alias (scope_id, alias_canon, generation, state,
  canon)`` covers the reverse (alias→canon) used at query analysis.
* ``idx_graph_edges_src (scope_id, src_unit, generation, type, dst_unit,
  weight)`` — bounded-PPR expansion, fully covering (V7-08.10 mandates
  ``(scope, src, type)`` covering; generation placed before the payload
  columns since the fence equality is always present);
  ``idx_graph_edges_dst`` — inbound edges + incident-edge closure.
* ``idx_events_v7_subject (scope_id, subject_canon, predicate_lemma,
  generation, occurred_start_us, occurred_end_us, event_id)`` — the
  event-index-first temporal lane ("when did X …", V7-09.09);
  ``idx_events_v7_window (scope_id, generation, occurred_start_us,
  occurred_end_us, event_id)`` — pure window overlap;
  ``idx_events_v7_unit (unit_id)`` — closure.
* ``idx_state_facts_key (scope_id, state_key, status, generation,
  unit_id)`` — current_value reads ("what is true now", V7-16.03/04).
* ``idx_standing_rules_scope (scope_id, status, generation)`` — prefetch
  scans active rules per scope (V7-20.02); ``idx_standing_rules_unit``
  — closure.
* ``idx_obs_v7_slot (scope_id, slot, generation, stale)`` — L-obs slot
  reads + stale-freshness (V7-14.03); ``idx_obs_v7_scope`` — scope scans.
* ``idx_profiles_v7_scope (scope_id, generation)`` — whole-scope profile
  materialization reads (``profile(subject=None)``, V7-14.06); the PK
  ``(scope_id, subject_canon, slot, generation)`` covers per-subject reads.
* ``idx_standing_queries_scope (scope_id, dirty, generation)`` —
  dirty-scan rebuilds on scoped change (V7-14.07) + scope closure.
* ``idx_t2_facts_state (scope_id, state_key, generation)`` /
  ``idx_t2_facts_subject (scope_id, subject_canon, generation)`` — T2
  state-key chains and subject lookups.

* ``idx_screening_log_unit (unit_id)``, ``idx_screening_log_source
  (source_id)``, ``idx_screening_log_scope (scope_id, at_us)`` — audit
  joins + closure (V7-19.12).
* ``idx_uvb_scope (scope_id, generation)`` — vector-block closure per
  scope (the PK leads ``encoder_id``, so it cannot serve scope-only
  traversal).
* ``idx_unit_fts_rows_scope (scope_id, generation)`` — MATCH-rowid →
  eligibility join (implicitly covering: secondary indexes carry rowid).
* ``idx_entity_canon_scope ON entity_canon(scope_id, generation, canon)``
  — per-generation canon enumeration for the entity vocabulary
  (V7-08.02); the PK ``(scope_id, canon, generation)`` covers point
  lookups and stays covering for canon-only selects.

Deletion closure (V7-06.11/V7-19.09): no V7 table holds a foreign key into
the evidence plane — derived rows are removed by scoped DELETEs on these
indexes in whatever order closure chooses, exactly like the v5 projection
tables.

Schema v8 (SPEC_V8 §19, V8-19.01) rides the same lazy-additive path:
``DDL_V8`` adds ``unit_time_mentions`` (write-time resolved temporal
mentions, V8-09.04), ``unit_doclen`` (maintained BM25F field lengths,
V8-07.04), two further ``units`` indexes (event-time ordering +
neighbor lookup, V8-09.06/V8-06.07), and the additive
``events_v7.subject_source`` column (subject provenance for backfilled
events, V8-09.07 — applied through the PRAGMA-probed idempotent-ALTER
path, SQLite having no ``ADD COLUMN IF NOT EXISTS``).
``ensure_v7_additive`` chains into ``ensure_v8_additive`` so every V7
writer leaves the V8 objects beside the V7 plane in the same
transaction, and ``meta.schema_v8`` records the provisioned version.
"""

from __future__ import annotations

import sqlite3

from ..core.types import json_dumps

SCHEMA_V7_TAG = "schema_v7/v1"
SCHEMA_V8_TAG = "schema_v8/v1"

#: Every §30 table (physical names — see module docstring + V7_TABLE_RENAMES
#: for the two collision renames). The FTS5 virtual tables are included:
#: they are §30-named objects even though they are capability-gated.
V7_TABLES: tuple[str, ...] = (
    "units",
    "unit_fts",
    "unit_fts_stem",
    "unit_fts_tri",
    "lex_stats",
    "lex_df",
    "unit_vectors_block",
    "entity_canon",
    "entity_mentions",
    "entity_aliases_v7",
    "graph_edges",
    "events_v7",
    "state_facts",
    "preferences",
    "standing_rules",
    "observations_v7",
    "profiles_v7",
    "standing_queries",
    "t2_facts",
    "screening_log",
    "run_manifests",
)

#: §30 normative name → physical name where a collision forced a rename
#: (the spec's own ``observations`` → ``observations_v7`` precedent).
V7_TABLE_RENAMES: dict[str, str] = {
    "events": "events_v7",
    "entity_aliases": "entity_aliases_v7",
}

#: Wiring tables behind the §30 FTS5 indexes — carrier + external content.
#: Not §30-named artifacts themselves, but ``v7_tables_present`` counts them:
#: the indexes are unusable without them.
V7_INTERNAL_TABLES: tuple[str, ...] = (
    "unit_fts_rows",
    "unit_fts_content",
)

#: The §30 FTS5 virtual tables (MATCH targets).
V7_FTS_TABLES: tuple[str, ...] = (
    "unit_fts",
    "unit_fts_stem",
    "unit_fts_tri",
)


DDL_V7 = """
-- ==== Retrievable evidence units (§30, V7-30.01) ==========================
--
-- The default retrieval substrate for conversations/documents (V7-13.02):
-- deterministic projections of sources + source_revisions + add-time
-- conversational arguments. No foreign keys — derived plane, closure members.
-- PK extends §30's (unit_id) with generation so a rebuild writes the new
-- generation beside the live one before the atomic fence flip (V7-30.02).
CREATE TABLE IF NOT EXISTS units (
    unit_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    scope_id TEXT NOT NULL,
    kind TEXT NOT NULL
        CHECK (kind IN ('turn', 'sentence_window', 'session', 'episode')),
    parent_unit_id TEXT,
    session_id TEXT,
    seq INTEGER,
    speaker_canon TEXT,
    perspective TEXT,
    recorded_at_us INTEGER,
    occurred_start_us INTEGER,
    occurred_end_us INTEGER,
    occurred_precision TEXT,
    occurred_source TEXT,
    byte_start INTEGER,
    byte_end INTEGER,
    generation INTEGER NOT NULL,
    PRIMARY KEY (unit_id, generation)
);
CREATE INDEX IF NOT EXISTS idx_units_scope_gen
    ON units(scope_id, generation, unit_id);
CREATE INDEX IF NOT EXISTS idx_units_session
    ON units(scope_id, generation, session_id, seq, unit_id);
CREATE INDEX IF NOT EXISTS idx_units_occurred
    ON units(scope_id, generation, occurred_start_us, occurred_end_us,
             unit_id);
CREATE INDEX IF NOT EXISTS idx_units_recorded
    ON units(scope_id, generation, recorded_at_us, unit_id);
CREATE INDEX IF NOT EXISTS idx_units_speaker
    ON units(scope_id, generation, speaker_canon, unit_id);
CREATE INDEX IF NOT EXISTS idx_units_source
    ON units(source_id, revision);
CREATE INDEX IF NOT EXISTS idx_units_parent
    ON units(parent_unit_id, generation);

-- ==== Fielded lexical shadow (§30, V7-06.01/02/04) ========================
--
-- Rowid carrier + external content table; the three §30 FTS5 virtual tables
-- and their mirror triggers are in the capability-gated section below.
-- Carrier/content are plain tables so the pair stays writable on a build
-- without FTS5 (the index lanes then report unavailable — v5 convention).
CREATE TABLE IF NOT EXISTS unit_fts_rows (
    row_id INTEGER PRIMARY KEY,
    unit_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    UNIQUE (unit_id, generation)
);
CREATE INDEX IF NOT EXISTS idx_unit_fts_rows_scope
    ON unit_fts_rows(scope_id, generation);

-- External content for all three unit indexes. ``when`` is a hard keyword
-- in SQLite DDL (CASE…WHEN) and MUST stay double-quoted — triggers below
-- quote it identically.
CREATE TABLE IF NOT EXISTS unit_fts_content (
    fts_row_id INTEGER PRIMARY KEY REFERENCES unit_fts_rows(row_id),
    text TEXT,
    speaker TEXT,
    entities TEXT,
    session TEXT,
    "when" TEXT
);

-- ==== Incremental corpus statistics (§30, V7-06.06) =======================
--
-- Keyed by (scope, generation, stats_version) — never a corpus fingerprint
-- (D7-10): a committed add adjusts df/length sums in O(terms of the add)
-- inside the same generation-fenced transaction as the posting write.
-- stats_version rides the key so a re-scored stats version coexists with
-- the one it replaces until the fence flips.
CREATE TABLE IF NOT EXISTS lex_stats (
    scope_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    field TEXT NOT NULL,
    stats_version TEXT NOT NULL,
    n_units INTEGER NOT NULL DEFAULT 0,
    total_len INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scope_id, generation, field, stats_version)
);
CREATE TABLE IF NOT EXISTS lex_df (
    scope_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    field TEXT NOT NULL,
    term TEXT NOT NULL,
    stats_version TEXT NOT NULL,
    df INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scope_id, generation, field, term, stats_version)
);

-- ==== Dense matrix blocks (§30, V7-07.04) =================================
--
-- Contiguous per-(encoder, scope, generation) vector matrix in bounded
-- blocks; ``rowmap_blob`` maps row index → unit ordering, ``data_blob`` the
-- packed rows, ``scale_blob`` the quantization scales (int8/bit modes).
CREATE TABLE IF NOT EXISTS unit_vectors_block (
    encoder_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    block_no INTEGER NOT NULL,
    n_rows INTEGER NOT NULL,
    dims INTEGER NOT NULL,
    quant TEXT NOT NULL CHECK (quant IN ('f32', 'int8', 'bit')),
    scale_blob BLOB,
    data_blob BLOB NOT NULL,
    rowmap_blob BLOB NOT NULL,
    PRIMARY KEY (encoder_id, scope_id, generation, block_no)
);
CREATE INDEX IF NOT EXISTS idx_uvb_scope
    ON unit_vectors_block(scope_id, generation);

-- ==== Entity plane (§30, V7-08.01–06) =====================================
--
-- Canonical entity vocabulary with unit-document frequency for IDF-weighted
-- entity signals (V7-08.06). generation appended to the PK for rebuild
-- coexistence (V7-30.02); §30 key (scope_id, canon) is the leading prefix.
CREATE TABLE IF NOT EXISTS entity_canon (
    scope_id TEXT NOT NULL,
    canon TEXT NOT NULL,
    generation INTEGER NOT NULL,
    display TEXT,
    df_units INTEGER NOT NULL DEFAULT 0,
    kind TEXT,
    first_seen_us INTEGER,
    last_seen_us INTEGER,
    PRIMARY KEY (scope_id, canon, generation)
);
CREATE INDEX IF NOT EXISTS idx_entity_canon_scope
    ON entity_canon(scope_id, generation, canon);

-- Canonical postings: the L-ent lane's index — ``scope+canon(+gen)`` →
-- pinned (unit_id, byte range, role). PK leads with the §30 key and is
-- itself the covering postings index; ``byte_start`` joins the key so a
-- unit mentioning the same canon at several spans keeps every pinned
-- occurrence instead of silently collapsing (§30 keys are minimums).
CREATE TABLE IF NOT EXISTS entity_mentions (
    scope_id TEXT NOT NULL,
    canon TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    surface TEXT,
    byte_start INTEGER,
    byte_end INTEGER,
    role TEXT CHECK (role IS NULL
        OR role IN ('subject', 'object', 'speaker', 'mention')),
    PRIMARY KEY (scope_id, canon, unit_id, byte_start, generation)
);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_unit
    ON entity_mentions(unit_id);

-- Alias table (V7-08.03/04, alias/v1 rules A1–A6). Named entity_aliases_v7:
-- ``entity_aliases`` already exists as the v1 span-pinned approval table —
-- ``IF NOT EXISTS`` under that name would silently keep the wrong shape.
CREATE TABLE IF NOT EXISTS entity_aliases_v7 (
    scope_id TEXT NOT NULL,
    canon TEXT NOT NULL,
    alias_canon TEXT NOT NULL,
    generation INTEGER NOT NULL,
    rule_id TEXT,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    method TEXT CHECK (method IS NULL
        OR method IN ('rule', 'caller', 'review')),
    state TEXT CHECK (state IS NULL
        OR state IN ('active', 'candidate', 'rejected')),
    PRIMARY KEY (scope_id, canon, alias_canon, generation)
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_v7_alias
    ON entity_aliases_v7(scope_id, alias_canon, generation, state, canon);

-- ==== Typed weighted edges (§30, V7-08.07/10) =============================
--
-- Bounded-PPR substrate. PK leads (scope, src, type) per the §30 key;
-- generation appended for rebuild coexistence. ``idx_graph_edges_src``
-- additionally covers dst_unit+weight so the PPR frontier expansion reads
-- the index alone; ``idx_graph_edges_dst`` serves inbound traversal and
-- incident-edge closure on a purged/quarantined unit (V7-08.11).
CREATE TABLE IF NOT EXISTS graph_edges (
    scope_id TEXT NOT NULL,
    src_unit TEXT NOT NULL,
    type TEXT NOT NULL,
    dst_unit TEXT NOT NULL,
    generation INTEGER NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    evidence_ref TEXT,
    PRIMARY KEY (scope_id, src_unit, type, dst_unit, generation)
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_src
    ON graph_edges(scope_id, src_unit, generation, type, dst_unit, weight);
CREATE INDEX IF NOT EXISTS idx_graph_edges_dst
    ON graph_edges(scope_id, dst_unit, generation, type, src_unit);

-- ==== Deterministic event calendar (§30, V7-09.08/09) =====================
--
-- Named events_v7: ``events`` already exists as the v1 audit journal
-- (event_seq AUTOINCREMENT) — reusing the name would silently bind the v1
-- shape. The temporal lane hits the event index first for temporal intents:
-- subject/predicate equality into occurred interval, all covering.
CREATE TABLE IF NOT EXISTS events_v7 (
    event_id TEXT PRIMARY KEY,
    unit_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    subject_canon TEXT,
    predicate_lemma TEXT,
    object_text TEXT,
    polarity TEXT,
    occurred_start_us INTEGER,
    occurred_end_us INTEGER,
    precision TEXT,
    pins_json TEXT,
    rule_id TEXT,
    generation INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_v7_subject
    ON events_v7(scope_id, subject_canon, predicate_lemma, generation,
                 occurred_start_us, occurred_end_us, event_id);
CREATE INDEX IF NOT EXISTS idx_events_v7_window
    ON events_v7(scope_id, generation, occurred_start_us, occurred_end_us,
                 event_id);
CREATE INDEX IF NOT EXISTS idx_events_v7_unit
    ON events_v7(unit_id);

-- ==== State keys / "what is true now" (§30, V7-16.03/04) ==================
--
-- (scope, state_key) → per-unit values with validity interval and
-- current/historical/disputed lifecycle label. PK extends §30's key with
-- ``value_norm`` + generation: one unit can state the same key twice with
-- different values (the disputed case, V7-16.05) — both must coexist, and
-- identical values dedupe naturally (SQLite treats NULL key parts as
-- distinct, so NULL ``value_norm`` rows always insert).
CREATE TABLE IF NOT EXISTS state_facts (
    scope_id TEXT NOT NULL,
    state_key TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    value_text TEXT,
    value_norm TEXT,
    valid_from_us INTEGER,
    valid_to_us INTEGER,
    status TEXT CHECK (status IS NULL
        OR status IN ('current', 'historical', 'disputed')),
    producer TEXT,
    pins_json TEXT,
    PRIMARY KEY (scope_id, state_key, unit_id, value_norm, generation)
);
CREATE INDEX IF NOT EXISTS idx_state_facts_key
    ON state_facts(scope_id, state_key, status, generation, unit_id);

-- ==== Preference facts (§30, V7-13.08) ====================================
--
-- ``object_text`` joins the PK so one unit can hold several preferences
-- from the same subject ("I love sushi and hate olives") without
-- collapsing — §30's (scope, subject, unit) key stays the leading prefix.
CREATE TABLE IF NOT EXISTS preferences (
    scope_id TEXT NOT NULL,
    subject_canon TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    object_text TEXT,
    polarity TEXT,
    strength TEXT,
    occurred_start_us INTEGER,
    pins_json TEXT,
    PRIMARY KEY (scope_id, subject_canon, unit_id, object_text, generation)
);

-- ==== Standing rules (§30, V7-20.02) ======================================
--
-- Implicit-recall rules detected by T0 imperative/normative patterns;
-- prefetch scans the active set per scope.
CREATE TABLE IF NOT EXISTS standing_rules (
    rule_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    unit_id TEXT,
    text_pin TEXT,
    trigger_entities TEXT,
    trigger_topics TEXT,
    valid_until_expr TEXT,
    status TEXT,
    generation INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_standing_rules_scope
    ON standing_rules(scope_id, status, generation);
CREATE INDEX IF NOT EXISTS idx_standing_rules_unit
    ON standing_rules(unit_id);

-- ==== Consolidated artifacts (§30, V7-14) =================================
--
-- Grounded observations: consolidated beliefs over ≥ 2 supporting units of
-- distinct evidence families about one slot. Closure members — erase /
-- quarantine / supersession of a support retires or withholds the row
-- (V7-14.08); ``stale`` implements the freshness contract (V7-14.03).
CREATE TABLE IF NOT EXISTS observations_v7 (
    obs_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    slot TEXT,
    text TEXT,
    producer TEXT,
    proof_count INTEGER NOT NULL DEFAULT 0,
    support_refs_json TEXT,
    contradict_refs_json TEXT,
    first_us INTEGER,
    last_us INTEGER,
    stale INTEGER NOT NULL DEFAULT 0,
    generation INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_v7_slot
    ON observations_v7(scope_id, slot, generation, stale);
CREATE INDEX IF NOT EXISTS idx_obs_v7_scope
    ON observations_v7(scope_id, generation);

-- Per-subject profile materialization (V7-14.06): current + historical slot
-- values each pointing at pinned support.
CREATE TABLE IF NOT EXISTS profiles_v7 (
    scope_id TEXT NOT NULL,
    subject_canon TEXT NOT NULL,
    slot TEXT NOT NULL,
    generation INTEGER NOT NULL,
    value TEXT,
    status TEXT,
    support_refs_json TEXT,
    updated_us INTEGER,
    PRIMARY KEY (scope_id, subject_canon, slot, generation)
);
CREATE INDEX IF NOT EXISTS idx_profiles_v7_scope
    ON profiles_v7(scope_id, generation);

-- Standing queries (V7-14.07): stored answer pack rebuilt in the background
-- when the scope's filtered units change — ``dirty`` marks pending rebuilds.
CREATE TABLE IF NOT EXISTS standing_queries (
    sq_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    principal TEXT,
    query TEXT,
    filters_json TEXT,
    pack_blob BLOB,
    pack_digest TEXT,
    built_generation INTEGER,
    dirty INTEGER NOT NULL DEFAULT 0,
    built_us INTEGER,
    generation INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_standing_queries_scope
    ON standing_queries(scope_id, dirty, generation);

-- ==== T2 grounded LLM facts (§30, V7-13.12/13) ============================
--
-- Quote-verified extracted facts; ``verified`` records the byte-match
-- outcome, ``unit_ids_json`` the pinned support set (any pinned unit held /
-- superseded / erased retires the fact — V7-13.13). ``scope_id`` rides the
-- §30 "scope-keyed" preamble even though the column list omits it — the
-- state-key/subject indexes and closure traversal are scoped.
CREATE TABLE IF NOT EXISTS t2_facts (
    fact_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    unit_ids_json TEXT,
    statement TEXT,
    quotes_json TEXT,
    subject_canon TEXT,
    predicate TEXT,
    object TEXT,
    occurred_start_us INTEGER,
    occurred_end_us INTEGER,
    state_key TEXT,
    model_id TEXT,
    prompt_digest TEXT,
    verified INTEGER NOT NULL DEFAULT 0,
    generation INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_t2_facts_state
    ON t2_facts(scope_id, state_key, generation);
CREATE INDEX IF NOT EXISTS idx_t2_facts_subject
    ON t2_facts(scope_id, subject_canon, generation);

-- ==== Screening audit (§30, V7-19.02/12) ==================================
--
-- Poisoning/redaction decision journal. ``unit_id``/``source_id`` are plain
-- columns (no FK): write-channel screening can precede unit projection, and
-- the audit row must survive evidence closure like the v1 events journal.
CREATE TABLE IF NOT EXISTS screening_log (
    decision_id TEXT PRIMARY KEY,
    unit_id TEXT,
    source_id TEXT,
    scope_id TEXT,
    rules_version TEXT,
    outcome TEXT CHECK (outcome IS NULL
        OR outcome IN ('allow', 'label', 'redact', 'quarantine', 'block')),
    rule_ids TEXT,
    at_us INTEGER,
    generation INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_screening_log_unit
    ON screening_log(unit_id);
CREATE INDEX IF NOT EXISTS idx_screening_log_source
    ON screening_log(source_id);
CREATE INDEX IF NOT EXISTS idx_screening_log_scope
    ON screening_log(scope_id, at_us);

-- ==== Evaluation provenance (§30, eval only) ==============================
CREATE TABLE IF NOT EXISTS run_manifests (
    digest TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    created_us INTEGER,
    generation INTEGER NOT NULL DEFAULT 0
);
"""

# The FTS5 virtual tables + mirror triggers are capability-gated, exactly
# like DDL_FTS5 / DDL_V5_FTS5: each virtual table is created only when a
# live probe of its tokenizer succeeds, and each index's trigger set is
# created only when that index exists. A missing tokenizer degrades that
# index's lane to unavailable — it never fails the ensure.
#
# (name, tokenizer_spec, create-sql) — order is creation order.
V7_FTS_DDL: tuple[tuple[str, str, str], ...] = (
    (
        "unit_fts",
        "unicode61 remove_diacritics 2",
        """CREATE VIRTUAL TABLE IF NOT EXISTS unit_fts USING fts5(
    text, speaker, entities, session, "when",
    content='unit_fts_content',
    content_rowid='fts_row_id',
    tokenize='unicode61 remove_diacritics 2'
)""",
    ),
    (
        "unit_fts_stem",
        "porter unicode61",
        """CREATE VIRTUAL TABLE IF NOT EXISTS unit_fts_stem USING fts5(
    text,
    content='unit_fts_content',
    content_rowid='fts_row_id',
    tokenize='porter unicode61'
)""",
    ),
    (
        "unit_fts_tri",
        "trigram",
        """CREATE VIRTUAL TABLE IF NOT EXISTS unit_fts_tri USING fts5(
    text,
    content='unit_fts_content',
    content_rowid='fts_row_id',
    tokenize='trigram'
)""",
    ),
)

#: All three CREATE VIRTUAL TABLE statements as one DDL string — the
#: reference/digest form (same convention as ``DDL_V5_FTS5``). Per-table
#: gating uses ``V7_FTS_DDL`` above.
DDL_V7_FTS5 = "\n".join(sql for _, _, sql in V7_FTS_DDL) + "\n"

# Mirror triggers: insert/delete/update on the content table mirror into
# each index — identical shape to FTS_TRIGGERS/V5_FTS_TRIGGERS. Per-index
# sets so an absent index never turns content writes into trigger errors.
# (index_name, create-trigger-sql)
V7_TRIGGER_DDL: tuple[tuple[str, str], ...] = (
    (
        "unit_fts",
        """CREATE TRIGGER IF NOT EXISTS unit_fts_ai AFTER INSERT ON unit_fts_content BEGIN
    INSERT INTO unit_fts(rowid, text, speaker, entities, session, "when")
        VALUES (new.fts_row_id, new.text, new.speaker, new.entities,
                new.session, new."when");
END""",
    ),
    (
        "unit_fts",
        """CREATE TRIGGER IF NOT EXISTS unit_fts_ad AFTER DELETE ON unit_fts_content BEGIN
    INSERT INTO unit_fts(unit_fts, rowid, text, speaker, entities,
                         session, "when")
        VALUES ('delete', old.fts_row_id, old.text, old.speaker,
                old.entities, old.session, old."when");
END""",
    ),
    (
        "unit_fts",
        """CREATE TRIGGER IF NOT EXISTS unit_fts_au AFTER UPDATE ON unit_fts_content BEGIN
    INSERT INTO unit_fts(unit_fts, rowid, text, speaker, entities,
                         session, "when")
        VALUES ('delete', old.fts_row_id, old.text, old.speaker,
                old.entities, old.session, old."when");
    INSERT INTO unit_fts(rowid, text, speaker, entities, session, "when")
        VALUES (new.fts_row_id, new.text, new.speaker, new.entities,
                new.session, new."when");
END""",
    ),
    (
        "unit_fts_stem",
        """CREATE TRIGGER IF NOT EXISTS unit_fts_stem_ai AFTER INSERT ON unit_fts_content BEGIN
    INSERT INTO unit_fts_stem(rowid, text)
        VALUES (new.fts_row_id, new.text);
END""",
    ),
    (
        "unit_fts_stem",
        """CREATE TRIGGER IF NOT EXISTS unit_fts_stem_ad AFTER DELETE ON unit_fts_content BEGIN
    INSERT INTO unit_fts_stem(unit_fts_stem, rowid, text)
        VALUES ('delete', old.fts_row_id, old.text);
END""",
    ),
    (
        "unit_fts_stem",
        """CREATE TRIGGER IF NOT EXISTS unit_fts_stem_au AFTER UPDATE ON unit_fts_content BEGIN
    INSERT INTO unit_fts_stem(unit_fts_stem, rowid, text)
        VALUES ('delete', old.fts_row_id, old.text);
    INSERT INTO unit_fts_stem(rowid, text)
        VALUES (new.fts_row_id, new.text);
END""",
    ),
    (
        "unit_fts_tri",
        """CREATE TRIGGER IF NOT EXISTS unit_fts_tri_ai AFTER INSERT ON unit_fts_content BEGIN
    INSERT INTO unit_fts_tri(rowid, text)
        VALUES (new.fts_row_id, new.text);
END""",
    ),
    (
        "unit_fts_tri",
        """CREATE TRIGGER IF NOT EXISTS unit_fts_tri_ad AFTER DELETE ON unit_fts_content BEGIN
    INSERT INTO unit_fts_tri(unit_fts_tri, rowid, text)
        VALUES ('delete', old.fts_row_id, old.text);
END""",
    ),
    (
        "unit_fts_tri",
        """CREATE TRIGGER IF NOT EXISTS unit_fts_tri_au AFTER UPDATE ON unit_fts_content BEGIN
    INSERT INTO unit_fts_tri(unit_fts_tri, rowid, text)
        VALUES ('delete', old.fts_row_id, old.text);
    INSERT INTO unit_fts_tri(rowid, text)
        VALUES (new.fts_row_id, new.text);
END""",
    ),
)

#: All nine mirror triggers as one DDL string (reference/digest form).
V7_FTS_TRIGGERS = "\n".join(sql for _, sql in V7_TRIGGER_DDL) + "\n"

#: Index names created by ``DDL_V7`` — the migrations-side ensure phase can
#: probe these by name alongside the tables (ADDITIVE_INDEXES precedent) so
#: a store that gained tables before an index landed still self-heals.
V7_INDEXES: tuple[str, ...] = (
    "idx_units_scope_gen",
    "idx_units_session",
    "idx_units_occurred",
    "idx_units_recorded",
    "idx_units_speaker",
    "idx_units_source",
    "idx_units_parent",
    "idx_unit_fts_rows_scope",
    "idx_uvb_scope",
    "idx_entity_canon_scope",
    "idx_entity_mentions_unit",
    "idx_entity_aliases_v7_alias",
    "idx_graph_edges_src",
    "idx_graph_edges_dst",
    "idx_events_v7_subject",
    "idx_events_v7_window",
    "idx_events_v7_unit",
    "idx_state_facts_key",
    "idx_standing_rules_scope",
    "idx_standing_rules_unit",
    "idx_obs_v7_slot",
    "idx_obs_v7_scope",
    "idx_profiles_v7_scope",
    "idx_standing_queries_scope",
    "idx_t2_facts_state",
    "idx_t2_facts_subject",
    "idx_screening_log_unit",
    "idx_screening_log_source",
    "idx_screening_log_scope",
)

#: Trigger names per index — probed by ``v7_fts_present``.
V7_TRIGGERS: tuple[str, ...] = (
    "unit_fts_ai", "unit_fts_ad", "unit_fts_au",
    "unit_fts_stem_ai", "unit_fts_stem_ad", "unit_fts_stem_au",
    "unit_fts_tri_ai", "unit_fts_tri_ad", "unit_fts_tri_au",
)


# ---------------------------------------------------------------------------
# Schema v8 additions (SPEC_V8 §19 — additive, digest-safe)
# ---------------------------------------------------------------------------

#: The §19 V8 tables — unit-keyed derived rows like the rest of the
#: plane: recomputable projections, generation-fenced (V8-19.04), swept
#: by purge closure (V8-19.03).
V8_TABLES: tuple[str, ...] = (
    "unit_time_mentions",
    "unit_doclen",
)

#: V8 index names — probed alongside the tables (V7_INDEXES convention).
V8_INDEXES: tuple[str, ...] = (
    "idx_utm_scope_time",
    "idx_units_scope_occurred",
    "idx_units_session_seq",
)

#: V8 additive columns — (table, column, ALTER ddl). SQLite has no
#: ``ADD COLUMN IF NOT EXISTS``; ``ensure_v8_additive`` probes
#: ``PRAGMA table_info`` and applies the ALTER only when the column is
#: absent (the codebase's idempotent-alter convention for additive
#: columns — the ``PRAGMA table_info`` probes in migrations.py).
V8_ALTER_COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "events_v7",
        "subject_source",
        "ALTER TABLE events_v7 ADD COLUMN subject_source TEXT",
    ),
)


# §19's normative DDL, verbatim — tables and indexes only. The block's
# ``ALTER TABLE events_v7 ADD COLUMN subject_source TEXT``
# (NULL | 'extracted' | 'speaker_backfill', V8-09.07) is not idempotent,
# so it executes through ``V8_ALTER_COLUMNS`` rather than this string.
DDL_V8 = """
-- Write-time resolved temporal mentions (V8-09.04). Half-open [start_us, end_us).
CREATE TABLE IF NOT EXISTS unit_time_mentions (
    unit_id          TEXT    NOT NULL,
    generation       INTEGER NOT NULL,
    scope_id         TEXT    NOT NULL,
    ord              INTEGER NOT NULL,          -- mention order within the unit
    start_us         INTEGER NOT NULL,
    end_us           INTEGER NOT NULL CHECK (end_us > start_us),
    precision        TEXT    NOT NULL
        CHECK (precision IN ('instant','day','week','month','season','year')),
    span_start       INTEGER NOT NULL,          -- byte offsets into the unit's pinned bytes
    span_end         INTEGER NOT NULL,
    anchor_us        INTEGER NOT NULL,          -- the unit's occurred instant used as anchor
    resolver_version TEXT    NOT NULL,
    PRIMARY KEY (unit_id, generation, ord)
);
CREATE INDEX IF NOT EXISTS idx_utm_scope_time
    ON unit_time_mentions(scope_id, generation, start_us, end_us);

-- Maintained BM25F field lengths (V8-07.04).
CREATE TABLE IF NOT EXISTS unit_doclen (
    unit_id    TEXT    NOT NULL,
    generation INTEGER NOT NULL,
    scope_id   TEXT    NOT NULL,
    field      TEXT    NOT NULL,                -- text | entities | when | speaker | session | ctx
    len        INTEGER NOT NULL CHECK (len >= 0),
    PRIMARY KEY (unit_id, generation, field)
);

-- Event-time ordering and neighbor lookup (V8-09.06, V8-06.07).
CREATE INDEX IF NOT EXISTS idx_units_scope_occurred
    ON units(scope_id, generation, occurred_start_us);
CREATE INDEX IF NOT EXISTS idx_units_session_seq
    ON units(scope_id, generation, session_id, seq);
"""


def _split(script: str) -> list[str]:
    """Split a DDL script on statement-terminating semicolons (comments and
    blank lines stripped) — same convention as schema_v4/v5._split."""
    stmts: list[str] = []
    buf: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        buf.append(line)
        if stripped.endswith(";"):
            stmts.append("\n".join(buf).rstrip().rstrip(";"))
            buf = []
    return stmts


def v7_statements() -> tuple[str, ...]:
    """``DDL_V7`` as an ordered statement tuple — plain tables and indexes
    only. The FTS5 virtual tables and mirror triggers are NOT here: they are
    capability-gated per tokenizer (``V7_FTS_DDL``/``V7_TRIGGER_DDL``)."""
    return tuple(_split(DDL_V7))


def v8_statements() -> tuple[str, ...]:
    """``DDL_V8`` as an ordered statement tuple — the §19 tables and
    indexes only. The ``events_v7.subject_source`` ALTER is not here:
    SQLite lacks ``ADD COLUMN IF NOT EXISTS``, so it executes through
    ``ensure_v8_additive``'s PRAGMA-probed ``V8_ALTER_COLUMNS`` path."""
    return tuple(_split(DDL_V8))


_TOKENIZER_PROBE_CACHE: dict[str, bool] = {}

#: Tokenizers built into the FTS5 extension itself (fts5_tokenizer.c) —
#: present whenever FTS5 is compiled in. ``trigram`` additionally requires
#: SQLite ≥ 3.34.0. Non-builtin names can only exist if a custom tokenizer
#: was registered — nothing in this build does, so they probe False.
_FTS5_BUILTIN_TOKENIZERS = frozenset({"unicode61", "porter", "ascii", "trigram"})
_TRIGRAM_MIN_VERSION = (3, 34, 0)


def _fts5_static(conn: sqlite3.Connection, spec: str) -> bool:
    """Name-level capability answer for connections that cannot run DDL
    (``PRAGMA query_only`` readers): builtin tokenizer + FTS5 compiled in +
    the trigram version floor. Authoritative for the builtin names — the
    writer-path live probe decides everything else."""
    try:
        opts = {
            str(r[0]).upper()
            for r in conn.execute("PRAGMA compile_options")
        }
    except sqlite3.Error:
        return False
    if "ENABLE_FTS5" not in opts:
        return False
    base = spec.split(None, 1)[0] if spec and spec.split() else ""
    if base not in _FTS5_BUILTIN_TOKENIZERS:
        return False
    if base == "trigram" and sqlite3.sqlite_version_info < _TRIGRAM_MIN_VERSION:
        return False
    return True


def fts5_tokenizer_available(conn: sqlite3.Connection, spec: str) -> bool:
    """Probe one FTS5 tokenizer spec on this build.

    On a writable connection the probe is live: create and drop a throwaway
    ``temp`` virtual table — a build without the tokenizer raises
    ``sqlite3.Error`` → False. On a ``query_only`` snapshot connection the
    live probe cannot run, so the answer falls back to the static
    compile-options check (builtins only). Results cache per process —
    tokenizer availability is a compile-time build property. Safe inside a
    caller's write transaction (temp objects, single statements, no
    implicit commit).
    """
    cached = _TOKENIZER_PROBE_CACHE.get(spec)
    if cached is not None:
        return cached
    try:
        query_only = bool(
            conn.execute("PRAGMA query_only").fetchone()[0]
        )
    except sqlite3.Error:
        query_only = False
    if query_only:
        ok = _fts5_static(conn, spec)
    else:
        ok = False
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE temp._v7_tok_probe"
                f" USING fts5(x, tokenize='{spec}')"
            )
            conn.execute("DROP TABLE temp._v7_tok_probe")
            ok = True
        except sqlite3.Error:
            ok = False
    _TOKENIZER_PROBE_CACHE[spec] = ok
    return ok


def _object_exists(conn: sqlite3.Connection, name: str, kind: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type=? AND name=?",
        (kind, name),
    ).fetchone()
    return row is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Declared-column probe (``PRAGMA table_info``); False on error —
    the ``store._verify_content_digests._has_col`` convention."""
    try:
        return any(
            r[1] == column
            for r in conn.execute(f"PRAGMA table_info({table})")
        )
    except sqlite3.Error:
        return False


def ensure_v7_additive(conn: sqlite3.Connection) -> None:
    """Idempotently create the V7 additive schema inside the caller's
    transaction (V6 lazy-additive semantics carried).

    Safe on any store and inside any write tx: every statement is
    ``CREATE ... IF NOT EXISTS`` executed via ``conn.execute`` — never
    ``executescript``, which would implicitly COMMIT the surrounding
    transaction. Plain tables/indexes are created unconditionally; each
    FTS5 virtual table is created only when its tokenizer probes available,
    and each index's trigger set only when that index exists — a build
    without FTS5 (or without the trigram tokenizer) still gains the plain
    schema and reports the lane unavailable rather than failing open.

    The V8 additive set (SPEC_V8 §19) rides this same path: the ensure
    chains into ``ensure_v8_additive`` so every V7 writer leaves the V8
    objects beside the V7 plane in the same transaction.
    """
    for stmt in v7_statements():
        conn.execute(stmt)
    for name, tokenizer, sql in V7_FTS_DDL:
        if fts5_tokenizer_available(conn, tokenizer):
            conn.execute(sql)
    for index_name, sql in V7_TRIGGER_DDL:
        if _object_exists(conn, index_name, "table"):
            conn.execute(sql)
    ensure_v8_additive(conn)


def ensure_v8_additive(conn: sqlite3.Connection) -> None:
    """Idempotently create the V8 additive schema inside the caller's
    transaction (SPEC_V8 §19, V8-19.01 — the ``ensure_v7_additive``
    lazy-additive pattern carried).

    Same discipline: statements execute one by one via ``conn.execute``
    — never ``executescript``. The ``events_v7.subject_source`` column
    applies through a ``PRAGMA table_info`` probe (SQLite lacks
    ``ADD COLUMN IF NOT EXISTS``), and the ``meta.schema_v8`` marker
    records the provisioned schema version — its later bumps enqueue the
    §19 reprojection jobs (the deriver workers' side). The V8 objects
    stand on the V7 plane (indexes over ``units``, a column on
    ``events_v7``): when the V7 base is absent this ensures it first, so
    the call stays safe on any store.
    """
    if not _object_exists(conn, "events_v7", "table"):
        ensure_v7_additive(conn)
        return
    for stmt in v8_statements():
        conn.execute(stmt)
    for table, column, ddl in V8_ALTER_COLUMNS:
        if not _object_exists(conn, table, "table"):
            continue
        if not _has_column(conn, table, column):
            conn.execute(ddl)
    if not _object_exists(conn, "meta", "table"):
        return
    marker = conn.execute(
        "SELECT value_json FROM meta WHERE key = 'schema_v8'"
    ).fetchone()
    if marker is None or marker[0] != json_dumps(SCHEMA_V8_TAG):
        conn.execute(
            "INSERT INTO meta(key, value_json) VALUES ('schema_v8', ?)"
            " ON CONFLICT(key) DO UPDATE"
            " SET value_json = excluded.value_json",
            (json_dumps(SCHEMA_V8_TAG),),
        )


def v7_tables_present(conn: sqlite3.Connection) -> bool:
    """True when every §30 table (plus the FTS wiring) exists.

    FTS5 virtual tables count toward presence: a build without the required
    tokenizers leaves them absent and this returns False — the V7 schema is
    honestly not fully provisioned. Use ``v7_fts_present`` for the FTS-only
    half.
    """
    names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    return all(t in names for t in V7_TABLES + V7_INTERNAL_TABLES)


def v7_fts_present(conn: sqlite3.Connection) -> dict[str, bool]:
    """Per-index presence map for the three §30 FTS5 virtual tables —
    including their mirror triggers, since an index without triggers would
    silently under-nominate (``_source_fts_ready`` precedent)."""
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    triggers = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
    }
    out: dict[str, bool] = {}
    for name in V7_FTS_TABLES:
        trig = {
            f"{name}_ai",
            f"{name}_ad",
            f"{name}_au",
        }
        out[name] = name in tables and trig <= triggers
    return out


def v8_tables_present(conn: sqlite3.Connection) -> bool:
    """True when every §19 V8 object exists — both tables, all three
    indexes, and every ``V8_ALTER_COLUMNS`` additive column."""
    names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if not all(t in names for t in V8_TABLES):
        return False
    indexes = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }
    if not all(i in indexes for i in V8_INDEXES):
        return False
    return all(
        _has_column(conn, table, column)
        for table, column, _ddl in V8_ALTER_COLUMNS
    )
