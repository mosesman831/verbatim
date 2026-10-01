"""V4 schema (SPEC_V4 §41) — SCHEMA_VERSION=4.

The relations below are the v4 target contract's *kernel* tables: object
registry + revisions, dependency edges, operation receipts, delivery and
dispatch permits, readiness obligations, closure runs/frontier, producer
manifests, view support, and schema operations. All use
``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS`` so a partial
migration can be replayed inside the same transaction boundary.

v4 reuses existing v1–v3 tables where their invariants already suffice
(sources, spans, claims, grants_v3, quarantine, vault_*, derivations, jobs,
…). The new tables here carry the contracts no earlier table can express:
fenced atomic application, disclosure linearization, resumable deletion
closure, per-receipt readiness, and verifiable phased migrations.
"""

SCHEMA_VERSION_V4 = 4

DDL_V4 = """
-- §41: one registered owner + lifecycle per durable object kind.
CREATE TABLE IF NOT EXISTS objects (
    object_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    current_revision INTEGER NOT NULL,
    disposition TEXT NOT NULL DEFAULT 'active',
    created_event INTEGER NOT NULL,
    PRIMARY KEY (kind, object_id)
);
CREATE INDEX IF NOT EXISTS idx_objects_scope ON objects(scope_id, kind);

-- Immutable revision identity; digest binds object type, identity,
-- revision, locator, algorithm, provenance (V4-13.04).
CREATE TABLE IF NOT EXISTS object_revisions (
    kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    digest TEXT NOT NULL,
    recorded_from INTEGER NOT NULL,
    recorded_until INTEGER,
    producer_ref TEXT,
    metadata_json TEXT,
    PRIMARY KEY (kind, object_id, revision),
    FOREIGN KEY (kind, object_id) REFERENCES objects(kind, object_id)
);
CREATE INDEX IF NOT EXISTS idx_object_revisions_id
    ON object_revisions(object_id, revision);

-- §41 dependency_edges: exact child/parent revision keys — complete,
-- validated ancestry for derivation and deletion closure (V4-19.01).
CREATE TABLE IF NOT EXISTS dependency_edges (
    child_kind TEXT NOT NULL,
    child_id TEXT NOT NULL,
    child_revision INTEGER NOT NULL,
    parent_kind TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    parent_revision INTEGER NOT NULL,
    role TEXT NOT NULL,
    producer_id TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    PRIMARY KEY (child_kind, child_id, child_revision,
                 parent_kind, parent_id, parent_revision, role)
);
CREATE INDEX IF NOT EXISTS idx_dep_edges_parent
    ON dependency_edges(parent_kind, parent_id, parent_revision);
CREATE INDEX IF NOT EXISTS idx_dep_edges_child
    ON dependency_edges(child_kind, child_id, child_revision);

-- §41 operation_receipts: idempotent atomic application (V4-09.05).
CREATE TABLE IF NOT EXISTS operation_receipts (
    operation_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    result_ref TEXT,
    effects_applied INTEGER NOT NULL,
    jobs_json TEXT,
    applied_seq INTEGER NOT NULL,
    created_us INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_op_receipts_scope
    ON operation_receipts(scope_id, created_us);

-- §41 delivery_permits: disclosure linearization (V4-09.06/08).
CREATE TABLE IF NOT EXISTS delivery_permits (
    permit_id TEXT PRIMARY KEY,
    caller_id TEXT NOT NULL,
    purpose TEXT NOT NULL,
    epoch_vector_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    dependency_versions_json TEXT NOT NULL,
    state TEXT NOT NULL,
    issued_us INTEGER NOT NULL,
    expires_us INTEGER NOT NULL,
    receipt_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_delivery_permits_caller
    ON delivery_permits(caller_id, issued_us);

-- §41 dispatch_permits: no ungated transport (V4-12.02).
CREATE TABLE IF NOT EXISTS dispatch_permits (
    permit_id TEXT PRIMARY KEY,
    recipient TEXT NOT NULL,
    purpose TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    scope_ids_json TEXT NOT NULL,
    consent_refs_json TEXT NOT NULL,
    reservation_id TEXT NOT NULL,
    max_spend REAL NOT NULL,
    issued_us INTEGER NOT NULL,
    expires_us INTEGER NOT NULL,
    state TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dispatch_permits_recipient
    ON dispatch_permits(recipient, issued_us);

-- §41 readiness_obligations: per-receipt capability DAG (V4-14.02/03).
CREATE TABLE IF NOT EXISTS readiness_obligations (
    obligation_id TEXT PRIMARY KEY,
    receipt_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    capability TEXT NOT NULL,
    depends_on_json TEXT NOT NULL DEFAULT '[]',
    state TEXT NOT NULL,
    error TEXT,
    created_us INTEGER NOT NULL,
    updated_us INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_readiness_receipt
    ON readiness_obligations(receipt_id, capability);
CREATE INDEX IF NOT EXISTS idx_readiness_state
    ON readiness_obligations(state, capability);

-- §41 closure_runs: truthful deletion lifecycle (V4-38.03).
CREATE TABLE IF NOT EXISTS closure_runs (
    run_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    roots_json TEXT NOT NULL,
    erasure_epoch INTEGER NOT NULL,
    phase TEXT NOT NULL,
    boundary_json TEXT NOT NULL,
    verification_json TEXT,
    error TEXT,
    created_us INTEGER NOT NULL,
    updated_us INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_closure_runs_phase
    ON closure_runs(phase, scope_id);

-- §41 closure_frontier: crash-safe continuation (V4-38.03/04).
CREATE TABLE IF NOT EXISTS closure_frontier (
    run_id TEXT NOT NULL,
    cursor INTEGER NOT NULL,
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    revision INTEGER,
    action TEXT NOT NULL,
    state TEXT NOT NULL,
    detail_json TEXT,
    PRIMARY KEY (run_id, cursor),
    FOREIGN KEY (run_id) REFERENCES closure_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_closure_frontier_state
    ON closure_frontier(state, run_id);

-- §41 producer_manifests: reproducible interpretation (V4-16, V4-20).
CREATE TABLE IF NOT EXISTS producer_manifests (
    producer_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    artifact_digest TEXT,
    rubric_digest TEXT,
    config_digest TEXT,
    schema_version INTEGER NOT NULL,
    license_ref TEXT,
    health TEXT NOT NULL DEFAULT 'unverified',
    registered_us INTEGER NOT NULL
);

-- §41 view_support: grounding without source impersonation (V4-20.03/04).
CREATE TABLE IF NOT EXISTS view_support (
    view_kind TEXT NOT NULL,
    view_id TEXT NOT NULL,
    view_revision INTEGER NOT NULL,
    proposition_locator TEXT NOT NULL,
    evidence_kind TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    evidence_revision INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    PRIMARY KEY (view_kind, view_id, view_revision, proposition_locator,
                 evidence_kind, evidence_id, evidence_revision)
);
CREATE INDEX IF NOT EXISTS idx_view_support_view
    ON view_support(view_kind, view_id, view_revision);

-- §41 schema_operations: verifiable phased migrations (V4-41.04/06).
CREATE TABLE IF NOT EXISTS schema_operations (
    operation_id TEXT PRIMARY KEY,
    version_from INTEGER NOT NULL,
    version_to INTEGER NOT NULL,
    phase TEXT NOT NULL,
    checksum TEXT NOT NULL,
    cursor TEXT,
    state TEXT NOT NULL,
    owner_lease TEXT,
    created_us INTEGER NOT NULL,
    updated_us INTEGER NOT NULL
);

-- V4-11.01: explicit tagged purpose constraint on v3 grants. NULL on a
-- pre-v4 row means legacy-ambiguous (handled by the 3→4 migration).
ALTER TABLE grants_v3 ADD COLUMN purpose_tag TEXT;
"""

DDL_V4_ALTER = """
-- Grants created before v4 used bare-set purposes semantics where an empty
-- set meant ANY. That ambiguity is the F4-03 defect: at v4 the empty set
-- normalizes to NONE (V4-62.05 owner-reviewed remediation maps genuinely
-- unrestricted grants to tag='any' explicitly, never silently).
UPDATE grants_v3 SET purpose_tag =
    CASE WHEN purposes_json IS NULL OR purposes_json IN ('[]', 'null')
         THEN 'legacy_empty'
         ELSE 'set' END
WHERE purpose_tag IS NULL;
"""


def _split(script: str) -> list[str]:
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


def v4_statements() -> tuple[str, ...]:
    """DDL_V4 tables + grants purpose_tag ALTER + the legacy-tag UPDATE."""
    stmts = _split(DDL_V4)
    stmts.extend(s for s in DDL_V4_ALTER.split(";") if s.strip())
    return tuple(stmts)


# The phase-ledger table is part of the migration machinery itself:
# migrations.py creates it ahead of the v4 migration on multi-step upgrades
# so earlier pending migrations can record their phases too (V4-41.05/06).
# Derived from DDL_V4 — never a second, divergent definition.
SCHEMA_OPERATIONS_DDL = next(
    s
    for s in _split(DDL_V4)
    if "CREATE TABLE IF NOT EXISTS schema_operations" in s
)
