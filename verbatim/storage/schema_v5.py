"""V5 schema (docs/v5_contracts.md §3, SPEC_V5 §07/§14.3/§30) — SCHEMA_VERSION=5.

The v5 relations are the source-projection plane: every row is a derived
projection of retained source bytes (canonical ``source_revisions.payload``
stays sacred — V5-03 hard rule 2). Deletion closure removes these rows and
rebuilds may regenerate them, so they carry no foreign keys into the
evidence plane — purge order stays unconstrained and reindexing is honest.

``source_lexical_projection`` follows the existing FTS convention: the
searchable token column is shadowed by an external-content FTS5 index via a
rowid carrier (``source_fts_rows``, eligibility metadata) + content table
(``source_fts``) + virtual table (``source_fts_idx``) + ai/ad/au triggers —
the same three-piece shape as ``fts_rows``/``facts_fts``/``facts_fts_idx``.
Reprojection never UPDATEs the carrier's rowid: stale rows are deleted and
fresh ones inserted, which is what keeps the trigger-maintained index exact
(REPLACE on a composite-PK content row would mint a new rowid without
firing the delete trigger, stranding index terms).

The virtual table + triggers live in ``DDL_V5_FTS5``/``V5_FTS_TRIGGERS`` —
capability-gated exactly like ``DDL_FTS5``: a SQLite build without FTS5
still migrates the plain tables and reports the source lexical lane
unavailable rather than failing open.

``jobs.kind`` gains the three v5 pipeline kinds (``source_project``,
``source_embed``, ``source_backfill`` — V5-08.16) through the sanctioned
CHECK-widening parent swap; like the v2/v3 rebuilds it runs outside the
migration transaction under the fenced ``PRAGMA foreign_keys = OFF``
procedure in ``migrations.py``.
"""

SCHEMA_VERSION = 5
# Convention-parity alias (schema_v3/v4 export SCHEMA_VERSION_V3/_V4).
SCHEMA_VERSION_V5 = SCHEMA_VERSION

DDL_V5 = """
PRAGMA foreign_keys = ON;

-- ==== Source-state control artifact (§14.3, contracts §3) ================
--
-- source_state/v1: one row per source recording the approved mutation head
-- and raw-record lifecycle. ``control_version`` is the monotonic CAS target
-- (distinct from the byte revision chain); ``disposition`` carries the
-- §14.3 lifecycle — the frozen contract names active|superseded|corrected|
-- retracted and the spec additionally admits recorded|archived|erased, so
-- the column is deliberately un-CHECKed rather than widening later.
-- ``effective_at`` is RFC3339; a future value schedules the successor —
-- read-time evaluation only, never an early-current marker (V5-14.12).
CREATE TABLE IF NOT EXISTS source_state (
    source_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    control_version INTEGER NOT NULL,
    mutation_head TEXT NOT NULL,
    disposition TEXT NOT NULL,
    superseded_by TEXT,
    effective_at TEXT,
    known_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    updated_at TEXT NOT NULL,
    producer TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_state_ns
    ON source_state(namespace, disposition);

-- ==== Source lexical projection (contracts §3) ===========================
--
-- Searchable token projection of retained source bytes (V5-07.08): binds
-- source revision, scope, tokenizer/index generation, token text, document
-- length, and a digest of what was projected. Reprojection of the same
-- (source_id, revision) replaces the row — the generation column records
-- which index generation produced it so stale rows are filtered at read.
CREATE TABLE IF NOT EXISTS source_lexical_projection (
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    scope_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    tokens TEXT NOT NULL,
    doc_len INTEGER NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY (source_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_source_lexical_scope
    ON source_lexical_projection(scope_id, generation);

-- FTS5 shadow carrier, mirroring fts_rows: eligibility metadata lives on
-- the rowid carrier so MATCH results join once for scope/generation, and
-- the UNIQUE(source_id, revision, generation) keeps one indexed row per
-- (revision, generation) pair exactly like fts_rows' claim triple.
CREATE TABLE IF NOT EXISTS source_fts_rows (
    row_id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    scope_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    UNIQUE (source_id, revision, generation)
);
CREATE INDEX IF NOT EXISTS idx_source_fts_rows_scope
    ON source_fts_rows(scope_id, generation);

-- Content table the triggers mirror into source_fts_idx (mirrors
-- facts_fts). Only the token text rides here; deleting the carrier row
-- deletes this row first so the shadow index never keeps orphan terms.
CREATE TABLE IF NOT EXISTS source_fts (
    fts_row_id INTEGER PRIMARY KEY REFERENCES source_fts_rows(row_id),
    text TEXT
);

-- ==== Source vectors (contracts §3) ======================================
--
-- Hashing-encoder vectors, contiguous per namespace for the bounded exact
-- scan; one row per (source, revision, encoder) so a pinned encoder swap
-- writes a fresh row rather than rewriting history.
CREATE TABLE IF NOT EXISTS source_vectors (
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    namespace TEXT NOT NULL,
    encoder TEXT NOT NULL,
    generation INTEGER NOT NULL,
    vector BLOB NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY (source_id, revision, encoder)
);
CREATE INDEX IF NOT EXISTS idx_source_vectors_ns
    ON source_vectors(namespace, encoder, generation);

-- ==== Entity postings (§30.4, contracts §3) ==============================
--
-- Deterministic identifier/entity mentions with byte offsets into the
-- pinned source revision (``offsets`` is a producer-serialized list).
-- Candidate/ranking inputs only — never authorization or entity merges
-- (V5-30.15).
CREATE TABLE IF NOT EXISTS entity_postings (
    namespace TEXT NOT NULL,
    entity TEXT NOT NULL,
    entity_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    offsets TEXT NOT NULL,
    generation INTEGER NOT NULL,
    PRIMARY KEY (namespace, entity, source_id, revision)
);
-- Closure/rebuild traversal by source (the PK leads with namespace/entity
-- for the X2 timeline read; deletion closure needs the reverse direction).
CREATE INDEX IF NOT EXISTS idx_entity_postings_source
    ON entity_postings(source_id, revision);

-- ==== Duplicate links (§30.2, contracts §3) ==============================
--
-- Dedup by linking, never deletion: each row binds a source revision to a
-- duplicate group (group_id = earliest live member's source_id per
-- contract §9) with the method and score that linked it. One row per
-- (source, revision, method); the method domain is the frozen contract's
-- three detectors — identical in §30.6 — so it is CHECK-enforced.
CREATE TABLE IF NOT EXISTS duplicate_links (
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    group_id TEXT NOT NULL,
    method TEXT NOT NULL
        CHECK (method IN ('exact_digest', 'normalized', 'minhash')),
    score REAL NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, revision, method)
);
CREATE INDEX IF NOT EXISTS idx_duplicate_links_group
    ON duplicate_links(group_id);

-- ==== Enrichment (§30, contracts §3) =====================================
--
-- T1 deterministic enrichment output per (source, revision, producer);
-- ``producer`` carries the pinned producer label (e.g. enrich/v1).
-- type/polarity/time_* are interpretation metadata — never authority
-- (V5-30.02) — and stay un-CHECKed so producer versions can extend the
-- domains without a schema migration. ``fields_json`` carries the
-- identifiers/entities/time-expression payload.
CREATE TABLE IF NOT EXISTS enrichment (
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    producer TEXT NOT NULL,
    type TEXT,
    polarity TEXT,
    time_precision TEXT,
    time_status TEXT,
    event_at TEXT,
    anchor_at TEXT,
    fields_json TEXT NOT NULL,
    PRIMARY KEY (source_id, revision, producer)
);
CREATE INDEX IF NOT EXISTS idx_enrichment_source
    ON enrichment(source_id, revision);

-- ==== Update candidates (§30.5, contracts §3) ============================
--
-- Advisory possible_updates: a prior live record sharing subject/entity/
-- type with a new record but differing in value, polarity, version, or
-- time. relation/state domains are the frozen contract's exact sets
-- (identical in §30.5), so they are CHECK-enforced. Nothing here mutates
-- lifecycle — adoption is an explicit owner operation (V5-30.19).
CREATE TABLE IF NOT EXISTS update_candidates (
    candidate_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    new_source_id TEXT NOT NULL,
    new_revision INTEGER NOT NULL,
    prior_source_id TEXT NOT NULL,
    prior_revision INTEGER NOT NULL,
    relation TEXT NOT NULL
        CHECK (relation IN ('contradicts', 'newer_value', 'negates', 'refines')),
    score REAL NOT NULL,
    state TEXT NOT NULL
        CHECK (state IN ('open', 'adopted', 'dismissed')),
    created_at TEXT NOT NULL
);
-- Open-candidate scans per namespace (V5-30.21 unresolved-candidate feed).
CREATE INDEX IF NOT EXISTS idx_update_candidates_ns
    ON update_candidates(namespace, state);
-- Both endpoints indexed for deletion closure and prior-record lookups.
CREATE INDEX IF NOT EXISTS idx_update_candidates_new
    ON update_candidates(new_source_id, new_revision);
CREATE INDEX IF NOT EXISTS idx_update_candidates_prior
    ON update_candidates(prior_source_id, prior_revision);

-- ==== Backfill cursor (§07/§08.17, contracts §3) =========================
--
-- Durable source-backfill cursor: per job_key, the last scanned source id,
-- the index generation being backfilled, a done flag, and the RFC3339
-- update stamp. Crash-safe resumability is the point — the cursor commits
-- with the batch it describes.
CREATE TABLE IF NOT EXISTS backfill_cursor (
    job_key TEXT PRIMARY KEY,
    last_source_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    done INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# The FTS5 virtual table + mirror triggers live outside DDL_V5 so the
# capability probe can skip them — same optional-capability contract as
# DDL_FTS5/FTS_TRIGGERS (SPEC §20, §42): no FTS5 build means the source
# lexical lane reports unavailable, never a failed open.
DDL_V5_FTS5 = """
CREATE VIRTUAL TABLE IF NOT EXISTS source_fts_idx USING fts5(
    text,
    content='source_fts',
    content_rowid='fts_row_id',
    tokenize='unicode61'
);
"""

# External-content shadow maintenance — identical trigger shape to
# FTS_TRIGGERS: insert/delete/update on source_fts mirror into the index;
# reprojection uses explicit DELETE+INSERT on the carrier so every index
# mutation passes through these triggers (no REPLACE rowid swap).
V5_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS source_fts_ai AFTER INSERT ON source_fts BEGIN
    INSERT INTO source_fts_idx(rowid, text) VALUES (new.fts_row_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS source_fts_ad AFTER DELETE ON source_fts BEGIN
    INSERT INTO source_fts_idx(source_fts_idx, rowid, text)
        VALUES ('delete', old.fts_row_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS source_fts_au AFTER UPDATE ON source_fts BEGIN
    INSERT INTO source_fts_idx(source_fts_idx, rowid, text)
        VALUES ('delete', old.fts_row_id, old.text);
    INSERT INTO source_fts_idx(rowid, text) VALUES (new.fts_row_id, new.text);
END;
"""

# Fourth rebuild of ``jobs``: widen ``kind`` for the v5 source-pipeline
# kinds (source_project/source_embed/source_backfill, V5-08.16 — the
# JobKind enum already carries them; only the persisted CHECK lags).
# Lanes are unchanged: the v5 kinds dispatch onto existing lanes
# (ordinary for fresh writes, maintenance for bounded backfill/rebuild).
# Same sanctioned parent-swap procedure as the v2/v3 rebuilds — on the
# migration path it runs outside the migration transaction under
# ``PRAGMA foreign_keys = OFF``; on a fresh store the table is empty.
DDL_V5_JOBS_REBUILD = """
CREATE TABLE IF NOT EXISTS jobs_rebuilt (
    job_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN
        ('harvest','admit','compare','embed','reindex','review_apply','purge',
         'episode_index','procedure_validate','replay','screen',
         'sparse_index','late_index','signature_index','episode_build',
         'transition_build','procedure_compile','procedure_refine',
         'consolidate','purge_derived','purge_vault','quarantine_review',
         'revocation_notify','vault_rotate','projection_sync',
         'connector_pull','source_project','source_embed','source_backfill')),
    state TEXT NOT NULL CHECK (state IN
        ('queued','leased','retry_wait','succeeded','failed','cancelled')),
    dedup_key BLOB UNIQUE,
    input_refs_json TEXT NOT NULL DEFAULT '{}',
    policy_epoch INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    not_before_us INTEGER NOT NULL DEFAULT 0,
    deadline_us INTEGER,
    lease_owner TEXT,
    lease_until_us INTEGER,
    generation INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    operation_key TEXT,
    lane TEXT NOT NULL DEFAULT 'ordinary'
        CHECK (lane IN ('ordinary','background','privacy_control',
                        'maintenance','control'))
);
INSERT INTO jobs_rebuilt
    (job_id, scope_id, kind, state, dedup_key, input_refs_json,
     policy_epoch, attempts, not_before_us, deadline_us,
     lease_owner, lease_until_us, generation, error_code,
     operation_key, lane)
SELECT job_id, scope_id, kind, state, dedup_key, input_refs_json,
       policy_epoch, attempts, not_before_us, deadline_us,
       lease_owner, lease_until_us, generation, error_code,
       operation_key, lane
FROM jobs;
DROP TABLE jobs;
ALTER TABLE jobs_rebuilt RENAME TO jobs;
CREATE INDEX IF NOT EXISTS idx_jobs_state_due ON jobs(state, not_before_us);
CREATE INDEX IF NOT EXISTS idx_jobs_scope ON jobs(scope_id, kind, state);
CREATE INDEX IF NOT EXISTS idx_jobs_lane_due ON jobs(lane, state, not_before_us);
"""


def _split(script: str) -> list[str]:
    """Split a DDL script on statement-terminating semicolons (comments and
    blank lines stripped) — same convention as schema_v4._split."""
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


def v5_statements() -> tuple[str, ...]:
    """DDL_V5 as an ordered statement tuple for the 4→5 migration and
    ``Store.create`` — plain tables/indexes only; the FTS5 shadow is
    capability-gated and created separately (``DDL_V5_FTS5`` +
    ``V5_FTS_TRIGGERS``), and the jobs CHECK widening runs as a fenced
    post-commit rebuild, never inside the migration transaction."""
    return tuple(_split(DDL_V5))


# ==== V6 additive tables (SPEC_V6 V6-03.14–16) =============================
#
# These relations extend the v5 schema WITHOUT touching ``DDL_V5`` /
# ``v5_statements()``. That choice is forced, not stylistic: both the
# ``Migration(version=5)`` digest and ``_creation_ddl_v5()`` hash the exact
# statement text, and every recorded ``migration_history`` row is verified
# against those immutable definitions on every ``apply`` (V4-41.03/C63).
# Editing ``DDL_V5`` would therefore invalidate the recorded digest of
# every existing v5 store — the next writable ``Store.open`` would raise
# STORE_CORRUPT instead of migrating. So the additive pattern already used
# for post-version schema bits applies instead:
#
#   * ``migrations._ensure_v6_additive`` — a resumable, schema-shape-probed
#     deferred phase run on EVERY ``apply()`` (same family as
#     ``_ensure_fts`` / ``_ensure_source_fts`` /
#     ``_rebuild_sources_dedup_index``), covering every pre-existing v5
#     store the next time it is opened writable, recorded in the
#     ``schema_operations`` ledger only when it did real work.
#   * ``ensure_additive_tables(conn)`` — the lazy twin for stores created
#     by ``Store.create``, which never runs ``apply()``: the exposure and
#     policy writers invoke it inside their own write transaction so the
#     table and its first row commit atomically. It executes statements
#     one by one — NEVER ``executescript``, which would COMMIT the
#     caller's transaction early.
#
# ``source_exposure`` (V6-03.14, contracts §8): one row per source-lane
# item delivered on the consumer path — the source-side twin of the
# claim-lane ``influence`` rows minted by ``recall_v3``'s write phase.
# PK ``(receipt_id, ord)`` makes a replayed delivery batch idempotent by
# insert-ignore. No foreign keys: the row is a delivery journal, not a
# lifecycle claim — a purged source's exposure facts stay countable,
# exactly like ``influence``.
DDL_V5_ADDITIVE = """
CREATE TABLE IF NOT EXISTS source_exposure (
    receipt_id TEXT NOT NULL,
    ord INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    score_family TEXT NOT NULL,
    delivered_at_us INTEGER NOT NULL,
    namespace TEXT NOT NULL,
    PRIMARY KEY (receipt_id, ord)
);
CREATE INDEX IF NOT EXISTS idx_source_exposure_ns
    ON source_exposure(namespace, delivered_at_us);

-- Validation attestations for policy_artifacts (V6-03.15): the
-- policy_artifacts table itself is frozen v2 DDL, so the paired-run
-- evidence that gates an unvalidated→validated transition is appended
-- here, in the same transaction as the state flip — the artifact row
-- stays byte-exact (digest still pins it) while the attestation carries
-- the gate evidence for audit.
CREATE TABLE IF NOT EXISTS policy_artifact_attestations (
    artifact_id TEXT NOT NULL,
    attested_us INTEGER NOT NULL,
    state TEXT NOT NULL
        CHECK (state IN ('validated','revoked')),
    evidence_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (artifact_id, attested_us)
);

-- Update-detection prior scan (V6-02.16): ``detect_update_candidates``
-- filters ``namespace + disposition IN (...)`` then ``ORDER BY source_id
-- LIMIT 512`` on every add — the two-column index left SQLite sorting
-- every live row per add (measured α≈1.41 superlinear add-ack growth).
-- The third column lets the index itself satisfy WHERE+ORDER+LIMIT, so
-- the scan pays for the limit, not the namespace.
CREATE INDEX IF NOT EXISTS idx_source_state_ns_disp_sid
    ON source_state(namespace, disposition, source_id);
"""

#: Tables created by ``DDL_V5_ADDITIVE`` — probed by the migrations ensure
#: phase and by read APIs that must tolerate a not-yet-ensured store.
ADDITIVE_TABLES = ("source_exposure", "policy_artifact_attestations")

#: Indexes created by ``DDL_V5_ADDITIVE`` — the ensure phase probes these
#: by name alongside the tables so a store that gained the additive
#: tables before an index landed still self-heals on the next ``apply``.
ADDITIVE_INDEXES = (
    "idx_source_exposure_ns",
    "idx_source_state_ns_disp_sid",
)


def v5_additive_statements() -> tuple[str, ...]:
    """``DDL_V5_ADDITIVE`` as an ordered statement tuple — the additive
    tables above, split on the same convention as ``v5_statements``."""
    return tuple(_split(DDL_V5_ADDITIVE))


def ensure_additive_tables(conn) -> None:
    """Idempotently create the V6 additive tables inside the caller's
    transaction.

    Safe on any store at any schema_version ≥ 5 and inside any write tx:
    each statement is ``CREATE ... IF NOT EXISTS`` executed via
    ``conn.execute`` — never ``executescript``, which would implicitly
    COMMIT the surrounding transaction. On a freshly ``Store.create``-ed
    store (which never runs ``apply()``) this is the first-writer-creates
    path; on stores migrated through ``apply()`` it is a no-op.
    """
    for stmt in v5_additive_statements():
        conn.execute(stmt)
