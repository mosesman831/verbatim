"""Schema DDL (SPEC §19-20, SPEC_V2 §37-38). SQLite is the source of truth;
projections are rebuildable. All content tables carry scope identity or a
scoped parent. v2 extends v1 additively: existing IDs and bytes are
preserved, new relations carry revisioned interpretation history.
"""

# v3 tables/alters live in schema_v3.py; v4 kernel tables live in
# schema_v4.py; v5 source-projection tables live in schema_v5.py.
# SCHEMA_VERSION tracks the newest supported version
# (docs/v5_contracts.md §3).
SCHEMA_VERSION = 5

DDL_V1 = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scopes (
    scope_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    principal_id TEXT,
    workspace_id TEXT,
    conversation_id TEXT,
    visibility TEXT NOT NULL CHECK (visibility IN ('conversation','workspace','owner')),
    acl_revision INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scope_grants (
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    principal_id TEXT NOT NULL,
    permission TEXT NOT NULL,
    issuer TEXT NOT NULL,
    granted_event INTEGER NOT NULL,
    revoked_event INTEGER,
    PRIMARY KEY (scope_id, principal_id, permission)
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL,
    external_id TEXT,
    source_kind TEXT NOT NULL CHECK (source_kind IN
        ('user_message','assistant_message','tool_output','import','operator_record')),
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    speaker_id TEXT,
    created_us INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_scope ON sources(scope_id);
-- Dedup is partition-local: the same host external_id may be captured
-- into different scopes independently (V3-12.04 keys the host stream,
-- not the global store). A global unique would couple scopes — scope
-- B's retry would land as a revision on scope A's source, and purging A
-- would then destroy B's evidence.
CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_origin_ext
    ON sources(scope_id, origin, external_id) WHERE external_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS source_revisions (
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    payload BLOB NOT NULL,
    payload_hmac BLOB NOT NULL,
    event_us INTEGER NOT NULL,
    captured_us INTEGER NOT NULL,
    timezone TEXT,
    provenance TEXT NOT NULL CHECK (provenance IN
        ('direct_user','approved_tool','assistant_generated','legacy_import','operator','unknown')),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source_id, revision)
);

CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    start_byte INTEGER NOT NULL CHECK (start_byte >= 0),
    end_byte INTEGER NOT NULL,
    excerpt_hmac BLOB NOT NULL,
    harvester_version TEXT NOT NULL,
    CHECK (end_byte > start_byte),
    FOREIGN KEY (source_id, revision)
        REFERENCES source_revisions(source_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_spans_source ON spans(source_id, revision);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    kind TEXT,
    label TEXT NOT NULL,
    created_event INTEGER NOT NULL,
    row_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_entities_scope_label ON entities(scope_id, label);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    normalized_alias TEXT NOT NULL,
    source_span_id TEXT REFERENCES spans(span_id),
    approval_event INTEGER,
    PRIMARY KEY (entity_id, normalized_alias)
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    subject_id TEXT,
    predicate TEXT,
    created_event INTEGER NOT NULL,
    row_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_claims_scope_pred ON claims(scope_id, predicate);
CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(scope_id, subject_id);

CREATE TABLE IF NOT EXISTS claim_revisions (
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    state TEXT NOT NULL CHECK (state IN
        ('pending','active','disputed','superseded','rejected','archived','erased')),
    object_json TEXT,
    polarity TEXT NOT NULL DEFAULT 'affirmative'
        CHECK (polarity IN ('affirmative','negated')),
    modality TEXT NOT NULL DEFAULT 'asserted'
        CHECK (modality IN ('asserted','hypothetical','habitual','uncertain')),
    condition_json TEXT,
    interpretation_json TEXT,
    recorded_from INTEGER NOT NULL,
    recorded_until INTEGER,
    PRIMARY KEY (claim_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_claim_rev_recorded
    ON claim_revisions(recorded_from, recorded_until);

CREATE TABLE IF NOT EXISTS valid_intervals (
    claim_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    interval_no INTEGER NOT NULL,
    from_us INTEGER,
    until_us INTEGER,
    precision TEXT NOT NULL DEFAULT 'unknown'
        CHECK (precision IN ('instant','day','month','year','unknown')),
    timezone TEXT,
    basis TEXT NOT NULL DEFAULT 'unknown',
    uncertainty_json TEXT,
    PRIMARY KEY (claim_id, revision, interval_no),
    FOREIGN KEY (claim_id, revision)
        REFERENCES claim_revisions(claim_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_valid_intervals_bounds ON valid_intervals(from_us, until_us);

CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    span_id TEXT NOT NULL REFERENCES spans(span_id),
    evidence_role TEXT NOT NULL DEFAULT 'primary'
        CHECK (evidence_role IN ('primary','contextual')),
    family_id TEXT,
    PRIMARY KEY (claim_id, revision, span_id),
    FOREIGN KEY (claim_id, revision)
        REFERENCES claim_revisions(claim_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_span ON claim_evidence(span_id);

CREATE TABLE IF NOT EXISTS claim_entities (
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    entity_id TEXT NOT NULL REFERENCES entities(entity_id),
    role TEXT NOT NULL DEFAULT 'mention',
    span_id TEXT REFERENCES spans(span_id),
    PRIMARY KEY (claim_id, entity_id, role)
);
CREATE INDEX IF NOT EXISTS idx_claim_entities_entity ON claim_entities(entity_id);

CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL CHECK (edge_type IN
        ('supports','conflicts_with','supersedes','corrects','context_of','derived_from')),
    decision_id TEXT,
    created_event INTEGER NOT NULL,
    retired_event INTEGER
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_kind, source_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_kind, target_id, edge_type);

CREATE TABLE IF NOT EXISTS conflict_groups (
    group_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved')),
    dimensions_json TEXT,
    resolution_event INTEGER
);

CREATE TABLE IF NOT EXISTS conflict_members (
    group_id TEXT NOT NULL REFERENCES conflict_groups(group_id),
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    PRIMARY KEY (group_id, claim_id)
);

CREATE TABLE IF NOT EXISTS events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    scope_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    recorded_us INTEGER NOT NULL,
    observed_wall_us INTEGER NOT NULL,
    policy_version TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_scope ON events(scope_id, kind);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    task TEXT NOT NULL,
    request_hmac BLOB NOT NULL,
    backend TEXT NOT NULL,
    model_revision TEXT,
    rubric_version TEXT NOT NULL,
    policy_epoch INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    created_us INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_scope_task ON decisions(scope_id, task);

CREATE TABLE IF NOT EXISTS decision_inputs (
    decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    revision INTEGER,
    content_hmac BLOB,
    PRIMARY KEY (decision_id, object_kind, object_id)
);
CREATE INDEX IF NOT EXISTS idx_decision_inputs_object ON decision_inputs(object_kind, object_id);

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    decision_id TEXT REFERENCES decisions(decision_id),
    proposed_effect_json TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open','approved','rejected','stale')),
    expected_versions_json TEXT NOT NULL DEFAULT '{}',
    resolved_event INTEGER
);
CREATE INDEX IF NOT EXISTS idx_reviews_scope_state ON reviews(scope_id, state);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN
        ('harvest','admit','compare','embed','reindex','review_apply','purge')),
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
    error_code TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_state_due ON jobs(state, not_before_us);
CREATE INDEX IF NOT EXISTS idx_jobs_scope ON jobs(scope_id, kind, state);

CREATE TABLE IF NOT EXISTS job_events (
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    event_seq INTEGER NOT NULL,
    state TEXT NOT NULL,
    error_code TEXT,
    PRIMARY KEY (job_id, event_seq)
);

CREATE TABLE IF NOT EXISTS embeddings (
    span_id TEXT NOT NULL REFERENCES spans(span_id),
    encoder_id TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    dtype TEXT NOT NULL DEFAULT 'float32le',
    vector BLOB NOT NULL,
    source_hmac BLOB NOT NULL,
    PRIMARY KEY (span_id, encoder_id)
);

CREATE TABLE IF NOT EXISTS encoder_manifests (
    encoder_id TEXT PRIMARY KEY,
    artifact_revision TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    normalization TEXT,
    license_id TEXT,
    manifest_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS fts_rows (
    row_id INTEGER PRIMARY KEY,
    claim_id TEXT NOT NULL,
    claim_revision INTEGER NOT NULL,
    scope_id TEXT NOT NULL,
    projection_generation INTEGER NOT NULL,
    UNIQUE (claim_id, claim_revision, projection_generation)
);
CREATE INDEX IF NOT EXISTS idx_fts_rows_scope ON fts_rows(scope_id, projection_generation);

CREATE TABLE IF NOT EXISTS facts_fts (
    fts_row_id INTEGER PRIMARY KEY REFERENCES fts_rows(row_id),
    text TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    actor_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('helpful','irrelevant','possibly_wrong')),
    created_us INTEGER NOT NULL,
    event_id TEXT
);

CREATE TABLE IF NOT EXISTS consents (
    consent_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    processor TEXT NOT NULL,
    purpose TEXT NOT NULL,
    granted_us INTEGER NOT NULL,
    revoked_us INTEGER,
    policy_digest TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consents_scope ON consents(scope_id, processor, purpose);

CREATE TABLE IF NOT EXISTS budget_ledger (
    reservation_id TEXT PRIMARY KEY,
    job_id TEXT,
    day_utc TEXT NOT NULL,
    token_bound INTEGER NOT NULL,
    reserved_cost_microusd INTEGER NOT NULL,
    actual_cost_microusd INTEGER,
    state TEXT NOT NULL CHECK (state IN ('reserved','settled','expired','overrun'))
);
CREATE INDEX IF NOT EXISTS idx_budget_day ON budget_ledger(day_utc, state);

CREATE TABLE IF NOT EXISTS purges (
    purge_id TEXT PRIMARY KEY,
    selection_digest BLOB NOT NULL,
    scope_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('previewed','suppressed','purging','completed')),
    requested_us INTEGER NOT NULL,
    approved_us INTEGER,
    completed_us INTEGER
);

CREATE TABLE IF NOT EXISTS purge_targets (
    purge_id TEXT NOT NULL REFERENCES purges(purge_id),
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    PRIMARY KEY (purge_id, object_kind, object_id)
);
CREATE INDEX IF NOT EXISTS idx_purge_targets_object ON purge_targets(object_kind, object_id);

CREATE TABLE IF NOT EXISTS migration_history (
    version INTEGER PRIMARY KEY,
    applied_us INTEGER NOT NULL,
    migration_digest TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Schema v2 (SPEC_V2 §37-38): additive extensions over v1.
# ---------------------------------------------------------------------------
# Applied on top of DDL_V1 for fresh stores; migrations.py upgrades v1
# databases in place. All new relations follow the same rules: scope or a
# scoped parent on every content row, revisioned interpretation history,
# stable operation keys for idempotence, and purge-aware suppression.
DDL_V2 = """
PRAGMA foreign_keys = ON;

-- Source views: primary + derived views of a source revision (§37).
-- The primary view references source_revisions.payload — no second copy of
-- canonical bytes; derived views may carry their own derived bytes.
CREATE TABLE IF NOT EXISTS source_views (
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    view_id TEXT NOT NULL,
    media_type TEXT NOT NULL DEFAULT 'text/plain',
    view_kind TEXT NOT NULL DEFAULT 'primary'
        CHECK (view_kind IN ('primary','normalized','redacted','extracted','transcript')),
    transformer_revision TEXT,
    locator_json TEXT NOT NULL DEFAULT '{}',
    derived_bytes BLOB,
    integrity_digest BLOB,
    PRIMARY KEY (source_id, revision, view_id),
    FOREIGN KEY (source_id, revision)
        REFERENCES source_revisions(source_id, revision)
);

-- Context groups: harvested evidence units (§12, §37).
CREATE TABLE IF NOT EXISTS context_groups (
    group_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    parser_version TEXT NOT NULL,
    operation_key TEXT NOT NULL,
    completeness TEXT NOT NULL DEFAULT 'complete'
        CHECK (completeness IN ('complete','partial','deferred')),
    recorded_from INTEGER NOT NULL DEFAULT 0,
    recorded_until INTEGER,
    FOREIGN KEY (source_id, revision)
        REFERENCES source_revisions(source_id, revision)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_context_groups_opkey
    ON context_groups(scope_id, operation_key);
CREATE INDEX IF NOT EXISTS idx_context_groups_source
    ON context_groups(source_id, revision);

CREATE TABLE IF NOT EXISTS context_members (
    group_id TEXT NOT NULL REFERENCES context_groups(group_id),
    span_id TEXT NOT NULL REFERENCES spans(span_id),
    role TEXT NOT NULL DEFAULT 'primary'
        CHECK (role IN ('primary','attribution','antecedent','condition','negation','temporal')),
    required INTEGER NOT NULL DEFAULT 1,
    ord INTEGER NOT NULL DEFAULT 0,
    dependency_reason TEXT,
    PRIMARY KEY (group_id, span_id, role)
);

-- Predicate registry: typed, versioned interpretation rules (§13, §37).
CREATE TABLE IF NOT EXISTS predicate_definitions (
    namespace TEXT NOT NULL,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    value_type TEXT NOT NULL DEFAULT 'literal',
    units TEXT,
    cardinality TEXT NOT NULL DEFAULT 'single'
        CHECK (cardinality IN ('single','set')),
    mutable INTEGER NOT NULL DEFAULT 1,
    sensitive INTEGER NOT NULL DEFAULT 0,
    authority_policy TEXT NOT NULL DEFAULT 'default',
    comparison_rules TEXT NOT NULL DEFAULT '{}',
    artifact_digest TEXT,
    PRIMARY KEY (namespace, name, version)
);

-- Evidence families: copies/derivatives of one underlying statement (§18, §37).
CREATE TABLE IF NOT EXISTS evidence_families (
    family_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    origin_kind TEXT NOT NULL,
    origin_id TEXT NOT NULL,
    created_event INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_families_scope ON evidence_families(scope_id);

CREATE TABLE IF NOT EXISTS family_members (
    family_id TEXT NOT NULL REFERENCES evidence_families(family_id),
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'copy',
    PRIMARY KEY (family_id, object_kind, object_id)
);

-- Episodes: task/conversation/event groupings (§21, §37).
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    revision INTEGER NOT NULL DEFAULT 1,
    host_task_id TEXT,
    host_session_id TEXT,
    parent_episode_id TEXT REFERENCES episodes(episode_id),
    kind TEXT NOT NULL DEFAULT 'task',
    label TEXT,
    recorded_from INTEGER NOT NULL DEFAULT 0,
    recorded_until INTEGER,
    row_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_episodes_scope ON episodes(scope_id, host_task_id);

CREATE TABLE IF NOT EXISTS episode_members (
    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    ord INTEGER NOT NULL DEFAULT 0,
    recorded_from INTEGER NOT NULL DEFAULT 0,
    recorded_until INTEGER,
    PRIMARY KEY (episode_id, object_kind, object_id)
);

-- Procedures: evidence-backed task experience (§23, §37).
CREATE TABLE IF NOT EXISTS procedures (
    procedure_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    revision INTEGER NOT NULL DEFAULT 1,
    task_label TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'proposed'
        CHECK (state IN ('proposed','active','review','retired')),
    environment_json TEXT NOT NULL DEFAULT '{}',
    condition_json TEXT,
    recorded_from INTEGER NOT NULL DEFAULT 0,
    recorded_until INTEGER,
    row_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_procedures_scope ON procedures(scope_id, state);

CREATE TABLE IF NOT EXISTS procedure_steps (
    procedure_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    step_no INTEGER NOT NULL,
    span_id TEXT REFERENCES spans(span_id),
    description TEXT NOT NULL,
    precondition_json TEXT,
    hazard TEXT,
    verification TEXT,
    PRIMARY KEY (procedure_id, revision, step_no),
    FOREIGN KEY (procedure_id) REFERENCES procedures(procedure_id)
);

CREATE TABLE IF NOT EXISTS outcome_receipts (
    receipt_id TEXT PRIMARY KEY,
    procedure_id TEXT NOT NULL REFERENCES procedures(procedure_id),
    revision INTEGER NOT NULL DEFAULT 1,
    outcome TEXT NOT NULL DEFAULT 'unknown'
        CHECK (outcome IN ('success','failure','partial','unknown')),
    checker TEXT NOT NULL,
    checked_artifact TEXT,
    detail_json TEXT NOT NULL DEFAULT '{}',
    recorded_us INTEGER NOT NULL DEFAULT 0
);

-- Prospective records: plans, deadlines, recurrence (§22, §37).
CREATE TABLE IF NOT EXISTS prospective_records (
    record_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    revision INTEGER NOT NULL DEFAULT 1,
    claim_id TEXT REFERENCES claims(claim_id),
    episode_id TEXT REFERENCES episodes(episode_id),
    owner_id TEXT NOT NULL,
    intention_text TEXT NOT NULL,
    due_us INTEGER,
    recurrence_json TEXT,
    status TEXT NOT NULL DEFAULT 'planned'
        CHECK (status IN ('planned','in_progress','completed','cancelled','overdue','unknown')),
    recorded_from INTEGER NOT NULL DEFAULT 0,
    recorded_until INTEGER
);
CREATE INDEX IF NOT EXISTS idx_prospective_scope_due ON prospective_records(scope_id, due_us);

-- Artifacts: approved non-text attachments (§37, §48).
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    media_type TEXT NOT NULL,
    digest BLOB NOT NULL,
    locator TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    availability TEXT NOT NULL DEFAULT 'available'
        CHECK (availability IN ('available','suppressed','purged')),
    created_event INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS artifact_links (
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    link_kind TEXT NOT NULL DEFAULT 'attachment',
    PRIMARY KEY (artifact_id, object_kind, object_id)
);

-- Operations: stable idempotence keys (§38, §39).
CREATE TABLE IF NOT EXISTS operations (
    scope_id TEXT NOT NULL,
    operation_key TEXT NOT NULL,
    input_digest BLOB NOT NULL,
    effect_kind TEXT NOT NULL,
    committed_event INTEGER NOT NULL,
    receipt_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (scope_id, operation_key)
);

-- Dependency graph: derived → exact inputs (§06, §38).
CREATE TABLE IF NOT EXISTS dependency_refs (
    derived_kind TEXT NOT NULL,
    derived_id TEXT NOT NULL,
    derived_revision INTEGER NOT NULL DEFAULT 1,
    input_kind TEXT NOT NULL,
    input_id TEXT NOT NULL,
    input_revision INTEGER NOT NULL DEFAULT 1,
    scope_id TEXT NOT NULL,
    invalidation TEXT NOT NULL DEFAULT 'revalidate'
        CHECK (invalidation IN ('revalidate','suppress','rebuild')),
    PRIMARY KEY (derived_kind, derived_id, derived_revision, input_kind, input_id, input_revision)
);
CREATE INDEX IF NOT EXISTS idx_dep_input ON dependency_refs(input_kind, input_id, input_revision);

-- Projection build tracking (§28, §38).
CREATE TABLE IF NOT EXISTS projection_outbox (
    event_seq INTEGER NOT NULL,
    scope_partition TEXT NOT NULL,
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    effect_digest BLOB,
    consumed_by TEXT,
    PRIMARY KEY (event_seq, scope_partition, object_kind, object_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_outbox_pending
    ON projection_outbox(scope_partition, consumed_by);

CREATE TABLE IF NOT EXISTS projection_builds (
    build_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    scope_partition TEXT NOT NULL,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    snapshot_seq INTEGER NOT NULL DEFAULT 0,
    caught_up_seq INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'building'
        CHECK (status IN ('building','validating','ready','published','failed','retired')),
    validation_digest BLOB,
    created_us INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS active_projections (
    scope_partition TEXT NOT NULL,
    projection_kind TEXT NOT NULL,
    build_id TEXT NOT NULL REFERENCES projection_builds(build_id),
    indexed_through_seq INTEGER NOT NULL DEFAULT 0,
    cas_revision INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scope_partition, projection_kind)
);

-- Embedding inputs + policy artifacts (§27, §38).
CREATE TABLE IF NOT EXISTS embedding_inputs (
    input_id TEXT PRIMARY KEY,
    span_id TEXT NOT NULL REFERENCES spans(span_id),
    encoder_id TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL,
    dependency_digest BLOB NOT NULL,
    input_known_seq INTEGER NOT NULL DEFAULT 0,
    UNIQUE (span_id, encoder_id, preprocessing_version)
);

CREATE TABLE IF NOT EXISTS policy_artifacts (
    artifact_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    digest BLOB NOT NULL,
    declared_json TEXT NOT NULL DEFAULT '{}',
    license_id TEXT,
    validation_state TEXT NOT NULL DEFAULT 'unvalidated'
        CHECK (validation_state IN ('unvalidated','validated','revoked')),
    created_us INTEGER NOT NULL DEFAULT 0
);

-- Disclosures: content-minimized egress receipts (§36, §38).
CREATE TABLE IF NOT EXISTS disclosures (
    disclosure_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    processor TEXT NOT NULL,
    purpose TEXT NOT NULL,
    input_refs_json TEXT NOT NULL DEFAULT '[]',
    usage_json TEXT NOT NULL DEFAULT '{}',
    outcome TEXT NOT NULL DEFAULT 'unknown',
    created_us INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_disclosures_scope ON disclosures(scope_id, processor);

-- Erasure ledger: opaque deletion knowledge (§38, §41).
CREATE TABLE IF NOT EXISTS erasure_ledger (
    erasure_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    object_kind TEXT NOT NULL,
    object_digest BLOB NOT NULL,
    purge_id TEXT,
    erased_event INTEGER NOT NULL,
    erasure_epoch INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_erasure_scope ON erasure_ledger(scope_id, object_kind);

-- Handoff capsules (§10, §38).
CREATE TABLE IF NOT EXISTS handoff_capsules (
    capsule_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    recipient_id TEXT NOT NULL,
    issuer_id TEXT NOT NULL,
    permission TEXT NOT NULL DEFAULT 'read_evidence',
    expires_us INTEGER,
    portable INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open','consumed','expired','revoked')),
    snapshot_json TEXT NOT NULL DEFAULT '{}',
    created_event INTEGER NOT NULL DEFAULT 0,
    consumed_event INTEGER
);

CREATE TABLE IF NOT EXISTS capsule_members (
    capsule_id TEXT NOT NULL REFERENCES handoff_capsules(capsule_id),
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (capsule_id, object_kind, object_id)
);

-- Ingest batches + connector cursors (§11, §38, §48).
CREATE TABLE IF NOT EXISTS ingest_batches (
    batch_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    received_count INTEGER NOT NULL DEFAULT 0,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'open'
        CHECK (state IN ('open','partial','complete','aborted')),
    created_us INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS connector_cursors (
    connector_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    cursor_value TEXT NOT NULL,
    updated_us INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (connector_id, scope_id)
);

-- Usage aggregates: bounded counts, never authority (§49, §38).
CREATE TABLE IF NOT EXISTS usage_aggregates (
    scope_id TEXT NOT NULL,
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    window_start_us INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scope_id, object_kind, object_id, kind, window_start_us)
);
"""

# v1 → v2 column additions kept separate so migrations.py can apply the
# same ALTERs to an existing store without replaying DDL_V1.
DDL_V2_ALTER = """
ALTER TABLE scopes ADD COLUMN owner_principal_id TEXT;
ALTER TABLE scopes ADD COLUMN authz_revision INTEGER NOT NULL DEFAULT 0;
ALTER TABLE claims ADD COLUMN interpretation_status TEXT NOT NULL DEFAULT 'structured'
    CHECK (interpretation_status IN ('unstructured','partial','structured'));
ALTER TABLE claim_revisions ADD COLUMN subject_id TEXT;
ALTER TABLE claim_revisions ADD COLUMN predicate TEXT;
ALTER TABLE claim_revisions ADD COLUMN registry_version INTEGER;
ALTER TABLE claim_revisions ADD COLUMN interpretation_status TEXT NOT NULL DEFAULT 'structured'
    CHECK (interpretation_status IN ('unstructured','partial','structured'));
ALTER TABLE claim_revisions ADD COLUMN method TEXT;
ALTER TABLE valid_intervals ADD COLUMN start_kind TEXT NOT NULL DEFAULT 'exact'
    CHECK (start_kind IN ('exact','uncertain_range','unbounded','unknown'));
ALTER TABLE valid_intervals ADD COLUMN end_kind TEXT NOT NULL DEFAULT 'exact'
    CHECK (end_kind IN ('exact','uncertain_range','unbounded','unknown'));
ALTER TABLE valid_intervals ADD COLUMN from_us_hi INTEGER;
ALTER TABLE valid_intervals ADD COLUMN until_us_hi INTEGER;
ALTER TABLE spans ADD COLUMN view_id TEXT;
ALTER TABLE spans ADD COLUMN operation_key TEXT;
ALTER TABLE jobs ADD COLUMN operation_key TEXT;
ALTER TABLE jobs ADD COLUMN lane TEXT NOT NULL DEFAULT 'ordinary'
    CHECK (lane IN ('ordinary','control'));
"""

# The v1 ``jobs.kind`` CHECK lists only the seven original JobKind values;
# SQLite cannot ALTER a CHECK, so the table is rebuilt in place. This must
# run AFTER DDL_V2_ALTER — the rebuilt table expects ``operation_key`` and
# ``lane`` to already exist on the source table. On the migration path the
# statements run with ``PRAGMA defer_foreign_keys`` so ``job_events`` child
# rows do not fault the mid-transaction parent swap; on a fresh store the
# tables are empty and the plain sequence is safe.
DDL_V2_JOBS_REBUILD = """
CREATE TABLE IF NOT EXISTS jobs_rebuilt (
    job_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN
        ('harvest','admit','compare','embed','reindex','review_apply','purge',
         'episode_index','procedure_validate','replay')),
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
        CHECK (lane IN ('ordinary','control'))
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

# FTS5 virtual table lives outside DDL_V1 so capability checks can skip it.
DDL_FTS5 = """
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts_idx USING fts5(
    text,
    content='facts_fts',
    content_rowid='fts_row_id',
    tokenize='unicode61'
);
"""

FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS facts_fts_ai AFTER INSERT ON facts_fts BEGIN
    INSERT INTO facts_fts_idx(rowid, text) VALUES (new.fts_row_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_fts_ad AFTER DELETE ON facts_fts BEGIN
    INSERT INTO facts_fts_idx(facts_fts_idx, rowid, text)
        VALUES ('delete', old.fts_row_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_fts_au AFTER UPDATE ON facts_fts BEGIN
    INSERT INTO facts_fts_idx(facts_fts_idx, rowid, text)
        VALUES ('delete', old.fts_row_id, old.text);
    INSERT INTO facts_fts_idx(rowid, text) VALUES (new.fts_row_id, new.text);
END;
"""
