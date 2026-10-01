"""Schema v3 additions (SPEC_V3 §39): governance, evidence-plane, learning-plane,
vault, and operations tables.

Rules inherited from v1/v2 and preserved here (V3-39.02..39.06):
- Every content table carries a ``scope_id`` or a derivation path to a
  scoped object — purge closure and authorization partitioning stay total.
- Identifiers, revisions, epochs, and sequences are typed columns; JSON is
  for validated payloads only (V3-39.03).
- Enumerations use CHECK constraints; widening a CHECK is a separate fenced
  rebuild (V3-39.01).
- Migrations are additive: no existing identifier, byte, offset, event, or
  receipt is rewritten except approved privacy transformations (V3-57.02).

The ``jobs`` table is rebuilt a second time to widen ``kind`` and ``lane``
CHECKs for the v3 pipeline kinds (§40) — same sanctioned parent-swap
procedure as the v2 rebuild.
"""

SCHEMA_VERSION_V3 = 3

# ---------------------------------------------------------------------------
# New tables. All use CREATE TABLE IF NOT EXISTS so a partial migration can
# be replayed inside one transaction.
# ---------------------------------------------------------------------------
DDL_V3 = """
PRAGMA foreign_keys = ON;

-- ==== Governance (§08–§11) ==============================================

CREATE TABLE IF NOT EXISTS principals (
    principal_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN
        ('human','agent','service','workspace','organization','external_party')),
    display_name TEXT,
    host_binding TEXT,
    created_us INTEGER NOT NULL DEFAULT 0,
    retired INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS perspectives (
    perspective_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    asserter TEXT,
    observer TEXT,
    audience_json TEXT NOT NULL DEFAULT '[]',
    created_event INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_perspectives_scope ON perspectives(scope_id, asserter);

CREATE TABLE IF NOT EXISTS perspective_subjects (
    perspective_id TEXT NOT NULL REFERENCES perspectives(perspective_id),
    subject_id TEXT NOT NULL,
    PRIMARY KEY (perspective_id, subject_id)
);

-- Purpose registry (§11.06): purpose strings come from a registry, never
-- free text, so purpose limitation is enforceable.
CREATE TABLE IF NOT EXISTS purposes (
    purpose TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    registered_us INTEGER NOT NULL DEFAULT 0,
    retired INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS grants_v3 (
    grant_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    verbs_json TEXT NOT NULL,
    purposes_json TEXT NOT NULL DEFAULT '[]',
    caveats_json TEXT NOT NULL DEFAULT '[]',
    delegation_depth INTEGER NOT NULL DEFAULT 0,
    issuer_id TEXT NOT NULL,
    issued_us INTEGER NOT NULL,
    expires_us INTEGER,
    revoked_us INTEGER,
    epoch INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_grants_v3_scope
    ON grants_v3(scope_id, principal_id, revoked_us);

CREATE TABLE IF NOT EXISTS delegations (
    delegation_id TEXT PRIMARY KEY,
    parent_grant_id TEXT NOT NULL REFERENCES grants_v3(grant_id),
    child_grant_id TEXT NOT NULL REFERENCES grants_v3(grant_id),
    delegator_id TEXT NOT NULL,
    delegate_id TEXT NOT NULL,
    created_us INTEGER NOT NULL DEFAULT 0,
    expires_us INTEGER,
    revoked_us INTEGER
);
CREATE INDEX IF NOT EXISTS idx_delegations_delegate ON delegations(delegate_id, revoked_us);

-- Host/operator-issued retention consent for capture (§11.11). Tool
-- permission is not retention consent — this record is.
CREATE TABLE IF NOT EXISTS capture_authorizations (
    authorization_id TEXT PRIMARY KEY,
    issuer_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    allowed_kinds_json TEXT NOT NULL,
    scope_ids_json TEXT NOT NULL DEFAULT '[]',
    retention_policy TEXT NOT NULL,
    policy_revision TEXT NOT NULL,
    issued_us INTEGER NOT NULL,
    expires_us INTEGER,
    revoked_us INTEGER
);
CREATE INDEX IF NOT EXISTS idx_capture_auth_principal
    ON capture_authorizations(principal_id, revoked_us);

-- ==== Evidence plane (§12–§13) ==========================================

-- Full v3 envelope metadata; the ``sources`` row keeps a coarse v2 storage
-- class so v2 read paths stay valid (V3-62 disposition).
CREATE TABLE IF NOT EXISTS source_envelopes (
    envelope_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    scope_id TEXT NOT NULL,
    envelope_kind TEXT NOT NULL CHECK (envelope_kind IN
        ('user_message','assistant_message','tool_call','tool_result',
         'file_diff','file_snapshot_ref','test_result','verification',
         'browser_state','screenshot_ref','plan','subgoal','decision',
         'error','recovery','handoff','delegation','document','import',
         'connector_item','agent_note','lesson','system_event')),
    actor_principal TEXT,
    perspective_id TEXT REFERENCES perspectives(perspective_id),
    event_us INTEGER,
    receipt_us INTEGER NOT NULL DEFAULT 0,
    media_type TEXT NOT NULL DEFAULT 'text/plain',
    trust_class TEXT NOT NULL DEFAULT 'unknown' CHECK (trust_class IN
        ('principal_direct','principal_reported','host_observed',
         'external_content','agent_generated','imported','unknown')),
    capture_proof TEXT,
    redaction_status TEXT NOT NULL DEFAULT 'none'
        CHECK (redaction_status IN ('none','applied','failed_closed')),
    adapter_version TEXT NOT NULL DEFAULT '',
    host_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    step_id TEXT NOT NULL DEFAULT '',
    artifact_ref TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (source_id, revision)
        REFERENCES source_revisions(source_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_envelopes_scope_kind
    ON source_envelopes(scope_id, envelope_kind);
CREATE INDEX IF NOT EXISTS idx_envelopes_task
    ON source_envelopes(scope_id, task_id, step_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_envelopes_source
    ON source_envelopes(source_id, revision, envelope_kind);

CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    host_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    boundary_rule TEXT NOT NULL DEFAULT 'task_id',
    created_event INTEGER NOT NULL DEFAULT 0,
    completed_event INTEGER,
    environment_digest TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_trajectories_task
    ON trajectories(scope_id, task_id);

CREATE TABLE IF NOT EXISTS trajectory_steps (
    step_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
    scope_id TEXT NOT NULL,
    ord INTEGER NOT NULL,
    action_envelope_id TEXT,
    environment_digest TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (trajectory_id, ord)
);
CREATE INDEX IF NOT EXISTS idx_traj_steps ON trajectory_steps(trajectory_id, ord);

CREATE TABLE IF NOT EXISTS step_observations (
    step_id TEXT NOT NULL REFERENCES trajectory_steps(step_id),
    envelope_id TEXT NOT NULL,
    ord INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (step_id, envelope_id)
);

CREATE TABLE IF NOT EXISTS state_anchors (
    anchor_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    ref TEXT NOT NULL,
    digest TEXT NOT NULL,
    step_id TEXT,
    created_event INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_anchors_scope ON state_anchors(scope_id, kind);

-- ==== Learning plane (§17–§25) ==========================================

CREATE TABLE IF NOT EXISTS transitions (
    transition_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
    scope_id TEXT NOT NULL,
    ord INTEGER NOT NULL,
    action_step_id TEXT,
    checker_ref TEXT,
    environment_digest TEXT,
    edge TEXT NOT NULL DEFAULT 'observed_after' CHECK (edge IN
        ('precedes','observed_after','verified_by','causal_hypothesis')),
    created_event INTEGER NOT NULL DEFAULT 0,
    UNIQUE (episode_id, ord, edge)
);
CREATE INDEX IF NOT EXISTS idx_transitions_episode ON transitions(episode_id, ord);

CREATE TABLE IF NOT EXISTS transition_anchors (
    transition_id TEXT NOT NULL REFERENCES transitions(transition_id),
    role TEXT NOT NULL CHECK (role IN ('pre','post')),
    anchor_id TEXT NOT NULL REFERENCES state_anchors(anchor_id),
    ord INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (transition_id, role, anchor_id)
);

CREATE TABLE IF NOT EXISTS procedure_signatures (
    procedure_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    signature_digest TEXT NOT NULL,
    ordered_ops_json TEXT NOT NULL DEFAULT '[]',
    intent_key TEXT NOT NULL DEFAULT '',
    scope_id TEXT NOT NULL,
    PRIMARY KEY (procedure_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_proc_sig ON procedure_signatures(scope_id, signature_digest);

CREATE TABLE IF NOT EXISTS procedure_exposures (
    exposure_id TEXT PRIMARY KEY,
    procedure_id TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    scope_id TEXT NOT NULL,
    task_id TEXT,
    session_id TEXT,
    outcome TEXT NOT NULL DEFAULT 'unknown'
        CHECK (outcome IN ('exposed','applicable','adopted','success','failure','unknown')),
    environment_digest TEXT,
    recorded_us INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_proc_exposures
    ON procedure_exposures(procedure_id, outcome);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    text TEXT NOT NULL,
    proof_count INTEGER NOT NULL DEFAULT 0,
    perspective_id TEXT REFERENCES perspectives(perspective_id),
    freshness TEXT NOT NULL DEFAULT 'unknown' CHECK (freshness IN
        ('stable','revalidate_after','volatile','unknown')),
    stale_since_seq INTEGER,
    producer TEXT NOT NULL DEFAULT 'slot_aggregate_v1',
    recorded_from INTEGER NOT NULL DEFAULT 0,
    recorded_until INTEGER
);
CREATE INDEX IF NOT EXISTS idx_observations_scope ON observations(scope_id, revision);

CREATE TABLE IF NOT EXISTS observation_evidence (
    observation_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('supports','contradicts')),
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    object_revision INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (observation_id, revision, role, object_kind, object_id)
);

-- Materialized dependency graph (§06.06): derived → exact inputs; purge,
-- invalidation, and influence traverse it. Edges are immutable (§17.02).
CREATE TABLE IF NOT EXISTS derivations (
    child_kind TEXT NOT NULL,
    child_id TEXT NOT NULL,
    child_revision INTEGER NOT NULL,
    parent_kind TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    parent_revision INTEGER NOT NULL,
    producer_kind TEXT NOT NULL,
    producer_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    scope_id TEXT NOT NULL,
    PRIMARY KEY (child_kind, child_id, child_revision,
                 parent_kind, parent_id, parent_revision)
);
CREATE INDEX IF NOT EXISTS idx_deriv_child
    ON derivations(child_kind, child_id, child_revision);
CREATE INDEX IF NOT EXISTS idx_deriv_parent
    ON derivations(parent_kind, parent_id, parent_revision);
CREATE INDEX IF NOT EXISTS idx_deriv_scope ON derivations(scope_id);

-- ==== Security metadata, screening, quarantine (§14, §34) ================

CREATE TABLE IF NOT EXISTS security_labels (
    label_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    source_trust TEXT NOT NULL DEFAULT 'unknown' CHECK (source_trust IN
        ('principal_direct','principal_reported','host_observed',
         'external_content','agent_generated','imported','unknown')),
    content_form TEXT NOT NULL DEFAULT 'unknown'
        CHECK (content_form IN ('descriptive','instructional','mixed','unknown')),
    attack_risk TEXT NOT NULL DEFAULT 'unassessed'
        CHECK (attack_risk IN ('unassessed','no_findings','suspicious','blocked')),
    review_state TEXT NOT NULL DEFAULT 'not_required' CHECK (review_state IN
        ('not_required','pending','released','quarantined','rejected')),
    findings_json TEXT NOT NULL DEFAULT '[]',
    method TEXT NOT NULL DEFAULT 'rules',
    rules_revision TEXT NOT NULL DEFAULT '',
    created_event INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sec_labels_scope
    ON security_labels(scope_id, review_state, attack_risk);

CREATE TABLE IF NOT EXISTS quarantine (
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    scope_id TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL DEFAULT '[]',
    findings_json TEXT NOT NULL DEFAULT '[]',
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN
        ('pending','released','suppressed','purged')),
    opened_event INTEGER NOT NULL DEFAULT 0,
    decided_event INTEGER,
    decided_by TEXT,
    decision_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (object_kind, object_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_quarantine_scope ON quarantine(scope_id, state);

-- ==== Vault, consents, hydration (§35, §11) ==============================

CREATE TABLE IF NOT EXISTS vault_entries (
    entry_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    sensitivity TEXT NOT NULL CHECK (sensitivity IN ('s1','s2','s3','s4')),
    placeholder TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    key_version INTEGER NOT NULL,
    wrap_key_version INTEGER NOT NULL,
    nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    wrapped_key BLOB NOT NULL,
    aad_digest BLOB NOT NULL,
    detection TEXT NOT NULL DEFAULT 'detected'
        CHECK (detection IN ('detected','owner_declared','no_detection')),
    created_event INTEGER NOT NULL DEFAULT 0,
    erased_event INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_vault_placeholder
    ON vault_entries(scope_id, placeholder);
CREATE INDEX IF NOT EXISTS idx_vault_scope ON vault_entries(scope_id, sensitivity);

-- Placeholder spans inside accepted views (§35.07): sanitized text stays
-- searchable through typed placeholders while original offsets remain
-- distinct — quotations cite the accepted view, never the removed value.
CREATE TABLE IF NOT EXISTS vault_refs (
    placeholder TEXT NOT NULL,
    entry_id TEXT NOT NULL REFERENCES vault_entries(entry_id),
    scope_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    PRIMARY KEY (scope_id, view_id, start_byte, end_byte)
);
CREATE INDEX IF NOT EXISTS idx_vault_refs_entry ON vault_refs(entry_id);

CREATE TABLE IF NOT EXISTS action_tickets (
    ticket_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    recipient_id TEXT NOT NULL,
    action_digest BLOB NOT NULL,
    purpose TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    nonce TEXT NOT NULL,
    issued_us INTEGER NOT NULL,
    expires_us INTEGER NOT NULL,
    gateway_id TEXT NOT NULL,
    consumed_us INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tickets_nonce ON action_tickets(nonce);

CREATE TABLE IF NOT EXISTS ticket_objects (
    ticket_id TEXT NOT NULL REFERENCES action_tickets(ticket_id),
    object_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    PRIMARY KEY (ticket_id, object_id, revision)
);

CREATE TABLE IF NOT EXISTS value_handles (
    handle_id TEXT PRIMARY KEY,
    vault_entry_id TEXT NOT NULL REFERENCES vault_entries(entry_id),
    scope_id TEXT NOT NULL,
    recipient_id TEXT NOT NULL,
    action_digest BLOB NOT NULL,
    consent_id TEXT NOT NULL,
    expires_us INTEGER NOT NULL,
    consumed_us INTEGER
);

-- Non-content redaction bookkeeping (§35.07): original and sanitized
-- offsets stay distinct so quotations remain honest.
CREATE TABLE IF NOT EXISTS redaction_spans (
    scope_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    orig_start INTEGER NOT NULL,
    orig_end INTEGER NOT NULL,
    accepted_start INTEGER NOT NULL,
    accepted_end INTEGER NOT NULL,
    sensitivity TEXT NOT NULL,
    entry_id TEXT,
    PRIMARY KEY (scope_id, view_id, orig_start, orig_end)
);

-- ==== Propagation ledger (§10.04) ========================================

CREATE TABLE IF NOT EXISTS propagations (
    propagation_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    recipient_id TEXT NOT NULL,
    verbs_json TEXT NOT NULL,
    purpose TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    created_us INTEGER NOT NULL,
    capsule_id TEXT,
    revoked_seq INTEGER,
    acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_propagations_object
    ON propagations(object_kind, object_id, revoked_seq);
CREATE INDEX IF NOT EXISTS idx_propagations_recipient
    ON propagations(recipient_id, revoked_seq);

-- ==== Controller, routing, influence (§26, §32, §43) =====================

CREATE TABLE IF NOT EXISTS routing_decisions (
    decision_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    state_key TEXT NOT NULL,
    routes_json TEXT NOT NULL,
    lane_set_json TEXT NOT NULL DEFAULT '[]',
    budgets_json TEXT NOT NULL DEFAULT '{}',
    state_json TEXT NOT NULL DEFAULT '{}',
    policy_revision TEXT NOT NULL,
    result_sizes_json TEXT NOT NULL DEFAULT '{}',
    outcome_credit TEXT,
    created_us INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_routing_scope ON routing_decisions(scope_id, state_key);

CREATE TABLE IF NOT EXISTS routing_stats (
    scope_id TEXT NOT NULL,
    state_key TEXT NOT NULL,
    action_key TEXT NOT NULL,
    exposures INTEGER NOT NULL DEFAULT 0,
    credit_json TEXT NOT NULL DEFAULT '{}',
    policy_revision TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (scope_id, state_key, action_key)
);

CREATE TABLE IF NOT EXISTS influence (
    handle_id TEXT PRIMARY KEY,
    receipt_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    caller_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    pack TEXT NOT NULL,
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    feedback_kind TEXT,
    action_receipt TEXT,
    created_us INTEGER NOT NULL DEFAULT 0,
    redacted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_influence_object ON influence(object_kind, object_id);
CREATE INDEX IF NOT EXISTS idx_influence_receipt ON influence(receipt_id);

-- ==== Freshness, environment, working sets, social (§24–§25) =============

CREATE TABLE IF NOT EXISTS freshness (
    scope_id TEXT NOT NULL,
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    class TEXT NOT NULL CHECK (class IN
        ('stable','revalidate_after','volatile','unknown')),
    revalidate_after_us INTEGER,
    anchor_refs_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (scope_id, object_kind, object_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_freshness_scope ON freshness(scope_id, class);

CREATE TABLE IF NOT EXISTS environment_state (
    scope_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    anchor_id TEXT,
    observed_us INTEGER NOT NULL DEFAULT 0,
    volatile INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (scope_id, key)
);

CREATE TABLE IF NOT EXISTS working_sets (
    set_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_us INTEGER NOT NULL DEFAULT 0,
    expires_us INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_working_sets ON working_sets(scope_id, session_id);

CREATE TABLE IF NOT EXISTS working_set_items (
    item_id TEXT PRIMARY KEY,
    set_id TEXT NOT NULL REFERENCES working_sets(set_id),
    kind TEXT NOT NULL,
    object_ref TEXT,
    text TEXT,
    ord INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS social_memory (
    record_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    observer_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    value_json TEXT NOT NULL DEFAULT '{}',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    revision INTEGER NOT NULL DEFAULT 1,
    created_us INTEGER NOT NULL DEFAULT 0,
    UNIQUE (scope_id, observer_id, subject_id, kind)
);

-- ==== Learning snapshots, index generations, replay (§34, §41, §43) ======

CREATE TABLE IF NOT EXISTS learning_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    digest BLOB NOT NULL,
    created_us INTEGER NOT NULL,
    validation TEXT NOT NULL DEFAULT 'unvalidated'
        CHECK (validation IN ('unvalidated','validated','rejected'))
);

CREATE TABLE IF NOT EXISTS index_generations (
    generation_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN
        ('fts','dense','sparse','late','signature')),
    scope_partition TEXT NOT NULL,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    snapshot_seq INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'building'
        CHECK (status IN ('building','ready','published','retired','failed')),
    created_us INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_index_gen ON index_generations(kind, scope_partition, status);

CREATE TABLE IF NOT EXISTS replay_runs (
    run_id TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    report_json TEXT NOT NULL DEFAULT '{}',
    sandbox_ref TEXT,
    created_us INTEGER NOT NULL DEFAULT 0
);
"""

# ---------------------------------------------------------------------------
# Additive column additions on pre-v3 tables (§39, §57.02). Nothing is
# rewritten; new columns carry safe defaults.
# ---------------------------------------------------------------------------
DDL_V3_ALTER = """
ALTER TABLE consents ADD COLUMN data_classes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE consents ADD COLUMN sanitization TEXT NOT NULL DEFAULT 'none'
    CHECK (sanitization IN ('none','sanitized','evidence_excerpt','full'));
ALTER TABLE consents ADD COLUMN retention_promise TEXT;
ALTER TABLE consents ADD COLUMN expires_us INTEGER;
ALTER TABLE consents ADD COLUMN budget_microusd INTEGER NOT NULL DEFAULT 0;
ALTER TABLE episodes ADD COLUMN boundary_rule TEXT;
ALTER TABLE episodes ADD COLUMN outcome TEXT NOT NULL DEFAULT 'unknown'
    CHECK (outcome IN ('success','failure','partial','unknown'));
ALTER TABLE episodes ADD COLUMN environment_digest TEXT;
ALTER TABLE procedures ADD COLUMN intent_signature_json TEXT;
ALTER TABLE procedures ADD COLUMN operations_json TEXT;
ALTER TABLE procedures ADD COLUMN bindings_json TEXT;
ALTER TABLE procedures ADD COLUMN hazards_json TEXT;
ALTER TABLE procedures ADD COLUMN verification_json TEXT;
ALTER TABLE procedures ADD COLUMN expected_outcome TEXT;
ALTER TABLE procedures ADD COLUMN failure_modes_json TEXT;
ALTER TABLE procedures ADD COLUMN applicability_json TEXT;
ALTER TABLE procedures ADD COLUMN preconditions_json TEXT;
ALTER TABLE procedures ADD COLUMN provenance_json TEXT;
ALTER TABLE procedures ADD COLUMN risk_class TEXT NOT NULL DEFAULT 'medium'
    CHECK (risk_class IN ('low','medium','high'));
ALTER TABLE procedures ADD COLUMN reuse_stats_json TEXT;
ALTER TABLE procedures ADD COLUMN compiler_manifest TEXT;
ALTER TABLE procedures ADD COLUMN evidence_family_id TEXT;
ALTER TABLE procedures ADD COLUMN security_label_id TEXT;
ALTER TABLE procedures ADD COLUMN freshness TEXT NOT NULL DEFAULT 'unknown'
    CHECK (freshness IN ('stable','revalidate_after','volatile','unknown'));
ALTER TABLE claim_revisions ADD COLUMN perspective_id TEXT;
ALTER TABLE claim_revisions ADD COLUMN security_label_id TEXT;
ALTER TABLE claim_revisions ADD COLUMN freshness TEXT;
ALTER TABLE prospective_records ADD COLUMN evidence_ref TEXT;
ALTER TABLE prospective_records ADD COLUMN precision TEXT;
ALTER TABLE prospective_records ADD COLUMN event_id TEXT;
ALTER TABLE prospective_records ADD COLUMN subgoal_of TEXT;
ALTER TABLE handoff_capsules ADD COLUMN verbs_json TEXT;
ALTER TABLE handoff_capsules ADD COLUMN purpose TEXT;
ALTER TABLE handoff_capsules ADD COLUMN delegation_chain_json TEXT;
ALTER TABLE handoff_capsules ADD COLUMN source_epoch INTEGER;
"""

# The v2 ``procedures.state`` CHECK is ('proposed','active','review',
# 'retired'); v3's promotion ladder is candidate → reviewed → active with
# deprecated/retired history (§21). The rebuilt CHECK accepts the union so
# v2 code paths stay valid while v3 writes the new states; migration maps
# 'proposed'→'candidate' and 'review'→'reviewed' at the data level
# (V3-57.05). Same fenced parent-swap procedure as the jobs rebuild.
DDL_V3_PROCEDURES_REBUILD = """
CREATE TABLE IF NOT EXISTS procedures_rebuilt (
    procedure_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id),
    revision INTEGER NOT NULL DEFAULT 1,
    task_label TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'candidate' CHECK (state IN
        ('candidate','reviewed','active','deprecated','retired',
         'proposed','review')),
    environment_json TEXT NOT NULL DEFAULT '{}',
    condition_json TEXT,
    recorded_from INTEGER NOT NULL DEFAULT 0,
    recorded_until INTEGER,
    row_version INTEGER NOT NULL DEFAULT 1,
    intent_signature_json TEXT,
    operations_json TEXT,
    bindings_json TEXT,
    hazards_json TEXT,
    verification_json TEXT,
    expected_outcome TEXT,
    failure_modes_json TEXT,
    applicability_json TEXT,
    preconditions_json TEXT,
    provenance_json TEXT,
    risk_class TEXT NOT NULL DEFAULT 'medium'
        CHECK (risk_class IN ('low','medium','high')),
    reuse_stats_json TEXT,
    compiler_manifest TEXT,
    evidence_family_id TEXT,
    security_label_id TEXT,
    freshness TEXT NOT NULL DEFAULT 'unknown'
        CHECK (freshness IN ('stable','revalidate_after','volatile','unknown'))
);
INSERT INTO procedures_rebuilt
    (procedure_id, scope_id, revision, task_label, state, environment_json,
     condition_json, recorded_from, recorded_until, row_version,
     intent_signature_json, operations_json, bindings_json, hazards_json,
     verification_json, expected_outcome, failure_modes_json,
     applicability_json, preconditions_json, provenance_json, risk_class,
     reuse_stats_json, compiler_manifest, evidence_family_id,
     security_label_id, freshness)
SELECT procedure_id, scope_id, revision, task_label,
       CASE state WHEN 'proposed' THEN 'candidate'
                  WHEN 'review' THEN 'reviewed'
                  ELSE state END,
       environment_json, condition_json, recorded_from, recorded_until,
       row_version, intent_signature_json, operations_json, bindings_json,
       hazards_json, verification_json, expected_outcome, failure_modes_json,
       applicability_json, preconditions_json, provenance_json, risk_class,
       reuse_stats_json, compiler_manifest, evidence_family_id,
       security_label_id, freshness
FROM procedures;
DROP TABLE procedures;
ALTER TABLE procedures_rebuilt RENAME TO procedures;
CREATE INDEX IF NOT EXISTS idx_procedures_scope ON procedures(scope_id, state);
"""

# Third rebuild of ``jobs``: widen ``kind`` to all v3 pipeline kinds and
# ``lane`` to the four-lane model (§40). 'control' stays valid for rows
# written by v2.
DDL_V3_JOBS_REBUILD = """
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
         'connector_pull')),
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
