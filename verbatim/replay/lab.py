"""Replay laboratory — offline policy replay with pinned manifests (§43).

``ReplayLab.replay_routing`` re-executes recorded ``routing_decisions``
inside a **sandbox store** — never production (V3-43.01) — under a
candidate policy table, and persists a what-if report into
``replay_runs`` (V3-43.03).

``ReplayLab.paired_execution`` runs full delivery-path arms (V3-43.05):
each policy arm restarts from the same snapshot in its own sandbox,
receives the identical declared replay authorization, and runs real
``recall_v3`` deliveries — divergence is measured on delivered item
sets, never on counterfactual outcome labels (V3-43.04).

``ReplayLab.paired_admission`` runs the write-side counterpart
(V3-43.02): identical payload streams through the real ``Ingester`` +
``run_pending`` drain — harvesting, screening, admission, relation
discovery, and consolidation jobs execute inside each arm's isolated
store, and claims/edges/quarantine/review burden are compared.

Honesty boundaries (declared, not claimed):

- Every V3-43.02 stage now executes somewhere: controller routing,
  lane sets/weights, budgets, abstention thresholds and packing via
  ``replay_routing``/``paired_execution``; harvesting, screening,
  admission, relation discovery and consolidation via
  ``paired_admission`` — all through the real production code paths
  inside sandboxes, never re-implemented.
- Replay runs local rules only — ``model_calls`` is always 0; fresh
  inference would need explicit intent, consent, and budget (V3-43.06).
- Decisions recorded before ``state_json`` existed are reported
  ``not_replayable`` — inputs are never fabricated.
- The sandbox is seeded with the scope's evidence state (claims,
  revisions, spans, FTS rows, edges, procedures, grants, projection
  meta); production is read once, never written by a replay.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import sqlite3
import time
from typing import Any, Iterable, Optional

from ..core.time import now_us
from ..config import VerbatimConfig
from ..core.types import (
    ErrorCode,
    Scope,
    VerbatimError,
    json_dumps,
    require_id,
    safe_json_loads,
)
from ..core.types_v3 import (
    ActionIntent,
    BudgetTier,
    MemoryKindV3,
    QueryClass,
    RecallRequestV3,
    Route,
)
from ..retrieval.v3 import controller as _ctrl
from ..retrieval.v3 import fusion_v3 as _fuse
from ..storage.repos import scope_id_for
from ..storage.store import Store
from .report import DecisionDivergence, WhatIfReport


def _require_state(raw: Any, decision_id: str) -> _ctrl._State:
    """Rebuild the pinned controller state; raise if unreconstructable."""
    if not isinstance(raw, dict) or "state" not in raw or "limits" not in raw:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"decision {decision_id} has no pinned replay inputs "
            "(state_json) — recorded before the replayable-decision "
            "contract; reported not_replayable, never fabricated",
        )
    s = raw["state"]
    return _ctrl._State(
        query_class=QueryClass(s["qc"]),
        action_intent=ActionIntent(s["intent"]),
        size_band=s["size"],
        procedures=bool(s["procedures"]),
        stuck=bool(s["stuck"]),
        session_phase=s["phase"],
        freshness_required=bool(s["fresh"]),
        tier=BudgetTier(s["tier"]),
        has_identifiers=bool(s["ids"]),
        memory_kinds=tuple(MemoryKindV3(k) for k in s.get("kinds", ())),
        prefetch=bool(s["prefetch"]),
    )


def _replay_request(raw: dict, scope_id: str) -> RecallRequestV3:
    """Minimal request carrying the recorded caller ceilings.

    ``query`` is a marker, never the recorded text — query text is not
    persisted in decision rows (metadata-only auditing, V3-04.11) and
    ``plan_routes`` reads only ceilings + the pinned state.
    """
    limits = raw["limits"]
    return RecallRequestV3(
        query="<replay>",
        scope_id=scope_id,
        caller_id="replay-lab",
        purpose="replay",
        max_items=int(limits["max_items"]),
        max_bytes=int(limits["max_bytes"]),
        target_tokens=int(limits["target_tokens"]),
        deadline_ms=int(limits["deadline_ms"]),
    )


def _variant_tables(variant: Optional[dict]) -> Optional[dict]:
    """Resolve a variant spec into plan_routes' table overrides.

    Spec shape::

        {"name": str,
         "tier_budgets": {"<tier>": {field: value, ...}, ...},   # merged
         "class_lanes": {"<query_class>": [lane, ...]},          # replace
         "class_routes": {"<query_class>": [route, ...]}},       # replace
         "lane_weights": {"<lane>": float, ...},                 # merged
         "abstain": {"term_floor_ratio": float}}                 # merged

    Returns ``None`` for an empty/absent variant — baseline re-execution
    is the determinism check (identical inputs must yield identical plans).
    """
    if not variant:
        return None
    tables: dict[str, Any] = {}
    if variant.get("tier_budgets"):
        merged = {t: dict(b) for t, b in _ctrl.TIER_BUDGETS.items()}
        for tier, overrides in variant["tier_budgets"].items():
            merged[BudgetTier(tier)].update(overrides)
        tables["tier_budgets"] = merged
    if variant.get("class_lanes"):
        lanes = dict(_ctrl._CLASS_LANES)
        for qc, lane_list in variant["class_lanes"].items():
            lanes[QueryClass(qc)] = tuple(lane_list)
        tables["class_lanes"] = lanes
    if variant.get("class_routes"):
        routes = dict(_ctrl._CLASS_ROUTES)
        for qc, route_list in variant["class_routes"].items():
            routes[QueryClass(qc)] = tuple(Route(r) for r in route_list)
        tables["class_routes"] = routes
    if variant.get("lane_weights"):
        merged_w = dict(_fuse.LANE_WEIGHTS)
        merged_w.update(
            {str(k): float(v) for k, v in variant["lane_weights"].items()}
        )
        tables["lane_weights"] = merged_w
    if variant.get("abstain"):
        # abstention thresholds — e.g. {"term_floor_ratio": 0.75}
        tables["abstain"] = dict(variant["abstain"])
    if variant.get("controller"):
        # learned/forced controller selection — e.g.
        # {"kind": "replay_learned", "artifact": {...}} or
        # {"kind": "forced", "action": "tier_tighter"} (V3-43.04
        # documented action support: the executed action is explicit)
        tables["controller"] = dict(variant["controller"])
    return tables or None


#: Snapshot copy rules for paired replay arms (V3-43.05) — every table
#: the read OR write path can consult, plus the FK parents those rows
#: reference. FK-ordered so parents land first even without the deferred
#: enforcement the write phase also uses. A ``None`` predicate copies the
#: whole (small/global) table; ``?`` binds the scope id. A rule that fails
#: is recorded ``skipped`` in the seed report — honest seeding, never
#: silent.
#:
#: Deliberately excluded: ``operations`` (HMAC-keyed receipts — the
#: sandbox's key differs from production's, so copied receipts would read
#: as phantom idempotence conflicts on the write path), ``replay_runs``
#: (replay artifacts, not evidence), and ``purposes``-style globals are
#: copied whole.
_SNAPSHOT_RULES: tuple[tuple[str, Optional[str]], ...] = (
    # --- roots + global registries ---
    ("principals", None),
    ("purposes", None),
    ("predicate_definitions", None),
    ("encoder_manifests", None),
    ("policy_artifacts", None),
    ("learning_snapshots", None),
    ("migration_history", None),
    ("scopes", "scope_id = ?"),
    # --- source chain (perspectives precede source_envelopes: FK) ---
    ("sources", "scope_id = ?"),
    ("perspectives", "scope_id = ?"),
    (
        "perspective_subjects",
        "perspective_id IN (SELECT perspective_id FROM perspectives"
        " WHERE scope_id = ?)",
    ),
    (
        "source_revisions",
        "source_id IN (SELECT source_id FROM sources WHERE scope_id = ?)",
    ),
    ("source_envelopes", "scope_id = ?"),
    (
        "source_views",
        "source_id IN (SELECT source_id FROM sources WHERE scope_id = ?)",
    ),
    ("redaction_spans", "scope_id = ?"),
    (
        "spans",
        "source_id IN (SELECT source_id FROM sources WHERE scope_id = ?)",
    ),
    # --- security/integrity state: holds + suppression ride along or a
    #     sandbox could deliver content production withholds ---
    ("security_labels", "scope_id = ?"),
    ("freshness", "scope_id = ?"),
    ("quarantine", "scope_id = ?"),
    ("purges", "scope_id = ?"),
    (
        "purge_targets",
        "purge_id IN (SELECT purge_id FROM purges WHERE scope_id = ?)",
    ),
    ("erasure_ledger", "scope_id = ?"),
    ("consents", "scope_id = ?"),
    ("disclosures", "scope_id = ?"),
    # --- claims + evidence ---
    ("claims", "scope_id = ?"),
    (
        "claim_revisions",
        "claim_id IN (SELECT claim_id FROM claims WHERE scope_id = ?)",
    ),
    (
        "claim_evidence",
        "claim_id IN (SELECT claim_id FROM claims WHERE scope_id = ?)",
    ),
    (
        "claim_entities",
        "claim_id IN (SELECT claim_id FROM claims WHERE scope_id = ?)",
    ),
    (
        "valid_intervals",
        "claim_id IN (SELECT claim_id FROM claims WHERE scope_id = ?)",
    ),
    (
        "feedback",
        "claim_id IN (SELECT claim_id FROM claims WHERE scope_id = ?)",
    ),
    ("entities", "scope_id = ?"),
    (
        "entity_aliases",
        "entity_id IN (SELECT entity_id FROM entities WHERE scope_id = ?)",
    ),
    ("edges", "scope_id = ?"),
    ("derivations", "scope_id = ?"),
    ("dependency_refs", "scope_id = ?"),
    ("evidence_families", "scope_id = ?"),
    (
        "family_members",
        "family_id IN (SELECT family_id FROM evidence_families"
        " WHERE scope_id = ?)",
    ),
    ("conflict_groups", "scope_id = ?"),
    (
        "conflict_members",
        "group_id IN (SELECT group_id FROM conflict_groups"
        " WHERE scope_id = ?)",
    ),
    ("context_groups", "scope_id = ?"),
    (
        "context_members",
        "group_id IN (SELECT group_id FROM context_groups"
        " WHERE scope_id = ?)",
    ),
    # --- review/decision chain (reviews FK decisions) ---
    ("decisions", "scope_id = ?"),
    (
        "decision_inputs",
        "decision_id IN (SELECT decision_id FROM decisions"
        " WHERE scope_id = ?)",
    ),
    ("reviews", "scope_id = ?"),
    ("routing_decisions", "scope_id = ?"),
    ("routing_stats", "scope_id = ?"),
    # --- authorization chain (delegations FK grants_v3) ---
    ("grants_v3", "scope_id = ?"),
    ("scope_grants", "scope_id = ?"),
    (
        "delegations",
        "parent_grant_id IN (SELECT grant_id FROM grants_v3"
        " WHERE scope_id = ?)"
        " OR child_grant_id IN (SELECT grant_id FROM grants_v3"
        " WHERE scope_id = ?)",
    ),
    ("capture_authorizations", None),
    ("action_tickets", "scope_id = ?"),
    (
        "ticket_objects",
        "ticket_id IN (SELECT ticket_id FROM action_tickets"
        " WHERE scope_id = ?)",
    ),
    # --- procedures ---
    ("procedures", "scope_id = ?"),
    (
        "procedure_steps",
        "procedure_id IN"
        " (SELECT procedure_id FROM procedures WHERE scope_id = ?)",
    ),
    (
        "procedure_signatures",
        "procedure_id IN"
        " (SELECT procedure_id FROM procedures WHERE scope_id = ?)",
    ),
    (
        "outcome_receipts",
        "procedure_id IN"
        " (SELECT procedure_id FROM procedures WHERE scope_id = ?)",
    ),
    ("procedure_exposures", "scope_id = ?"),
    # --- experience ---
    ("episodes", "scope_id = ?"),
    (
        "episode_members",
        "episode_id IN"
        " (SELECT episode_id FROM episodes WHERE scope_id = ?)",
    ),
    ("state_anchors", "scope_id = ?"),
    ("transitions", "scope_id = ?"),
    (
        "transition_anchors",
        "transition_id IN (SELECT transition_id FROM transitions"
        " WHERE scope_id = ?)",
    ),
    ("trajectories", "scope_id = ?"),
    ("trajectory_steps", "scope_id = ?"),
    (
        "step_observations",
        "step_id IN (SELECT step_id FROM trajectory_steps"
        " WHERE scope_id = ?)",
    ),
    # --- working/env/prospective/social ---
    ("working_sets", "scope_id = ?"),
    (
        "working_set_items",
        "set_id IN"
        " (SELECT set_id FROM working_sets WHERE scope_id = ?)",
    ),
    ("environment_state", "scope_id = ?"),
    ("prospective_records", "scope_id = ?"),
    ("observations", "scope_id = ?"),
    (
        "observation_evidence",
        "observation_id IN (SELECT observation_id FROM observations"
        " WHERE scope_id = ?)",
    ),
    ("social_memory", "scope_id = ?"),
    # --- vectors + lexical projection ---
    (
        "embeddings",
        "span_id IN (SELECT s.span_id FROM spans s"
        " JOIN sources so ON so.source_id = s.source_id"
        " WHERE so.scope_id = ?)",
    ),
    (
        "embedding_inputs",
        "span_id IN (SELECT s.span_id FROM spans s"
        " JOIN sources so ON so.source_id = s.source_id"
        " WHERE so.scope_id = ?)",
    ),
    ("fts_rows", "scope_id = ?"),
    (
        "facts_fts",
        "fts_row_id IN"
        " (SELECT row_id FROM fts_rows WHERE scope_id = ?)",
    ),
    # --- vault + sharing ---
    ("vault_entries", "scope_id = ?"),
    ("value_handles", "scope_id = ?"),
    ("vault_refs", "scope_id = ?"),
    ("handoff_capsules", "scope_id = ?"),
    (
        "capsule_members",
        "capsule_id IN (SELECT capsule_id FROM handoff_capsules"
        " WHERE scope_id = ?)",
    ),
    # --- ops/derived + in-flight pipeline state ---
    ("influence", "scope_id = ?"),
    ("propagations", "scope_id = ?"),
    ("artifacts", "scope_id = ?"),
    (
        "artifact_links",
        "artifact_id IN (SELECT artifact_id FROM artifacts"
        " WHERE scope_id = ?)",
    ),
    ("usage_aggregates", "scope_id = ?"),
    ("ingest_batches", "scope_id = ?"),
    ("connector_cursors", "scope_id = ?"),
    # seq-fenced requests (min_ready_seq/known_at_seq) evaluate against
    # the same event numbering as production only when the ledger rides
    # along — without it a sandbox would under-satisfy every fence.
    ("events", "scope_id = ?"),
    ("jobs", "scope_id = ?"),
    (
        "job_events",
        "job_id IN (SELECT job_id FROM jobs WHERE scope_id = ?)",
    ),
    (
        "budget_ledger",
        "job_id IN (SELECT job_id FROM jobs WHERE scope_id = ?)",
    ),
    # --- projection state (last: active_projections FKs builds) ---
    ("index_generations", None),
    ("projection_builds", None),
    ("projection_outbox", None),
    ("active_projections", None),
    ("meta", None),
)

#: Tables whose fresh-sandbox rows must yield to the source's — ``meta``
#: carries ``projection_generation``; generation-scoped FTS rows are
#: invisible to search unless the sandbox adopts the source's generation.
# Store.create pre-seeds rows in these tables — snapshot writes must
# replace, not collide.
_UPSERT_TABLES = frozenset({"meta", "migration_history"})


def _read_snapshot(
    src: sqlite3.Connection, scope_id: str
) -> tuple[dict[str, tuple[list[str], list]], dict[str, Any], str]:
    """Read the scope's full evidence state ONCE under the caller's read
    snapshot — every arm writes this identical image, so seed equality
    is structural, not re-measured (V3-43.05).

    Returns ``(tables, counts, digest)``: ``tables`` maps table →
    ``(columns, rows)``; ``counts`` is the per-table row report (or a
    ``skipped:*`` marker); ``digest`` is a content hash of the whole
    image — the honest identity of what every arm started from."""
    tables: dict[str, tuple[list[str], list]] = {}
    counts: dict[str, Any] = {}
    h = hashlib.sha256()
    for table, pred in _SNAPSHOT_RULES:
        try:
            cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})")]
            if not cols:
                counts[table] = "skipped:no_table"
                continue
            if pred is None:
                rows = src.execute(
                    f"SELECT {', '.join(cols)} FROM {table}"
                ).fetchall()
            else:
                rows = src.execute(
                    f"SELECT {', '.join(cols)} FROM {table} WHERE {pred}",
                    (scope_id,),
                ).fetchall()
            tables[table] = (cols, rows)
            counts[table] = len(rows)
            h.update(table.encode())
            h.update(b"\0")
            h.update(",".join(cols).encode())
            h.update(b"\0")
            for row in rows:
                h.update(repr(tuple(row)).encode())
                h.update(b"\1")
        except sqlite3.Error as exc:
            counts[table] = f"skipped:{exc}"
    return tables, counts, h.hexdigest()


def _write_snapshot(
    dst: sqlite3.Connection,
    tables: dict[str, tuple[list[str], list]],
) -> dict[str, Any]:
    """Write the shared snapshot image into one sandbox inside the
    caller's write tx. ``defer_foreign_keys`` makes row order irrelevant
    (incl. the episodes self-FK) while integrity is still enforced at
    commit — and ``foreign_key_check`` surfaces any violation honestly
    instead of committing a broken seed."""
    dst.execute("PRAGMA defer_foreign_keys = ON")
    counts: dict[str, Any] = {}
    for table, (cols, rows) in tables.items():
        marks = ",".join("?" * len(cols))
        mode = (
            "INSERT OR REPLACE" if table in _UPSERT_TABLES
            else "INSERT"
        )
        dst.executemany(
            f"{mode} INTO {table} ({', '.join(cols)})"
            f" VALUES ({marks})",
            rows,
        )
        counts[table] = len(rows)
    violations = dst.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT,
            "sandbox seed violated foreign keys — the snapshot would "
            "start arms on inconsistent evidence: "
            + repr(violations[:8]),
        )
    return counts


def _seed_snapshot(
    src: sqlite3.Connection, dst: sqlite3.Connection, scope_id: str
) -> dict[str, Any]:
    """Copy the scope's full evidence state into the sandbox — the
    delivery path reads claims, revisions, spans, payloads, edges, FTS
    rows (external-content triggers rebuild ``facts_fts_idx``), vectors,
    procedures, grants, and projection meta. Read-only on production."""
    tables, _read_counts, _digest = _read_snapshot(src, scope_id)
    return _write_snapshot(dst, tables)


def _provision_replay_caller(conn: sqlite3.Connection, scope_id: str) -> None:
    """Declared replay authorization: a synthetic principal + READ/QUOTE
    grant inside the sandbox only — every arm gets the identical context
    so paired comparisons see equal authorization (V3-43.05)."""
    conn.execute(
        "INSERT OR IGNORE INTO principals"
        "(principal_id,kind,display_name,created_us)"
        " VALUES('replay-lab','service','Replay laboratory',1)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO grants_v3"
        "(grant_id,scope_id,principal_id,verbs_json,purposes_json,"
        " caveats_json,delegation_depth,issuer_id,issued_us)"
        " VALUES(?,?,?,?,?,?,0,'replay-lab',1)",
        (
            f"grant-replay-{scope_id}", scope_id, "replay-lab",
            '["read","quote"]', '["evaluate"]', '[]',
        ),
    )


def _propagate_runtime(src_store: Store, sandbox: Store) -> None:
    """Carry the production runtime surface into the sandbox — config,
    encoder, and capability probes. A sandbox without them would
    silently degrade lanes the production arm had live (semantic/dense
    measuring differently is a replay defect, not a policy delta)."""
    for attr in ("cfg", "encoder", "encode_query", "encoder_id"):
        if hasattr(src_store, attr):
            setattr(sandbox, attr, getattr(src_store, attr))


def _create_sandbox(src_store: Store, path: str) -> Store:
    """Sandbox store bound to the production HMAC key.

    The snapshot copies ``source_revisions.payload_hmac`` and
    ``spans.excerpt_hmac`` verbatim — both are digests under the
    production key. A sandbox minted with a fresh key could never
    deliver what production delivered: the fail-closed integrity
    re-check in packaging would read every copied payload as corrupt.
    The key file is COPIED alongside the sandbox (never linked) so the
    artifact stays self-contained — ``Store.open(sandbox_path)`` on the
    recorded ``sandbox_ref`` keeps working even if production's key
    rotates. Key sharing is safe here: a sandbox is a same-operator
    replay double, and ``operations`` receipts stay excluded from the
    snapshot regardless.
    """
    key_path = getattr(src_store, "_key_path", None)
    if isinstance(key_path, str) and os.path.exists(key_path):
        dst_key = os.path.abspath(path) + ".key"
        if not os.path.exists(dst_key):
            with open(key_path, "rb") as fh:
                key = fh.read()
            fd = os.open(dst_key, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                         0o600)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(key)
            except BaseException:
                os.unlink(dst_key)
                raise
    return Store.create(path)


_ARM_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _check_arm_names(arm_list: list[dict]) -> list[str]:
    """Arm names become sandbox filenames — bounded charset, unique."""
    names = [a.get("name") for a in arm_list]
    if not all(names) or len(set(names)) != len(names):
        raise VerbatimError(
            ErrorCode.VALIDATION, "each arm needs a unique name"
        )
    bad = [n for n in names if not _ARM_NAME_RE.match(str(n))]
    if bad:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"arm names must match {_ARM_NAME_RE.pattern!r}: {bad!r}",
        )
    return names


def _check_sandbox_dir(store: Store, sandbox_dir: str) -> str:
    """Refuse sandbox dirs that intermingle with the live store: the
    store's own directory (files would mix) or any directory containing
    the live store file (a cleanup sweep would delete it). A sibling
    subdirectory under the store's parent is fine (V3-43.01)."""
    if not sandbox_dir:
        raise VerbatimError(
            ErrorCode.VALIDATION, "sandbox_dir is required"
        )
    live = os.path.abspath(getattr(store, "path", "") or "")
    target = os.path.abspath(sandbox_dir)
    if live:
        try:
            contains_live = os.path.commonpath((target, live)) == target
        except ValueError:
            contains_live = False
        if contains_live or target == os.path.dirname(live):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "paired sandbox_dir intermingles with the live store — "
                "replays run against sandbox stores only (V3-43.01)",
            )
    return target


def _delivered_ids(result: Any) -> tuple[str, ...]:
    """Delivered item identities (kind:id@rev) across all packs."""
    out: list[str] = []
    for pack in result.packs:
        for item in pack.items:
            h = item.handle
            out.append(f"{h.object_kind}:{h.object_id}@{h.revision}")
    return tuple(sorted(set(out)))


def _delivered_bytes(result: Any) -> dict[str, int]:
    """Per-delivered-item text bytes — the token-cost accounting substrate
    paired evaluators score against (G8's total-task-token reduction)."""
    out: dict[str, int] = {}
    for pack in result.packs:
        for item in pack.items:
            h = item.handle
            key = f"{h.object_kind}:{h.object_id}@{h.revision}"
            out[key] = len((item.text or "").encode("utf-8"))
    return out


def _request_fingerprint(request: RecallRequestV3) -> str:
    """Pin a request without storing query text — the same metadata-only
    posture routing_decisions takes (V3-04.11)."""
    canon = "|".join(
        f"{f.name}={getattr(request, f.name)!r}"
        for f in dataclasses.fields(request)
        if f.name != "query"
    )
    q = hashlib.sha256(str(request.query).encode("utf-8")).hexdigest()[:16]
    return hashlib.sha256(f"{canon}|q={q}".encode("utf-8")).hexdigest()[:24]


def _payload_bytes(p: dict) -> bytes:
    """Normalize a replay payload dict to the bytes the envelope ingests —
    ``payload`` bytes pass through, anything else (or ``text``) encodes
    UTF-8."""
    raw = p.get("payload")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return str(raw if raw is not None else p.get("text", "")).encode(
        "utf-8"
    )


class ReplayLab:
    """Offline policy replay against a sandbox store (§43.01)."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def replay_routing(
        self,
        scope_id: str,
        *,
        sandbox_path: str,
        variant: Optional[dict] = None,
        decision_ids: Optional[Iterable[str]] = None,
    ) -> WhatIfReport:
        """Re-execute recorded routing decisions under ``variant``.

        Baseline replay (``variant=None``) is the determinism check: the
        same pinned inputs under the same tables must reproduce the
        recorded plan exactly — a divergence there is a defect, not a
        policy delta.
        """
        require_id(scope_id, "scope_id")
        if not sandbox_path:
            raise VerbatimError(
                ErrorCode.VALIDATION, "sandbox_path is required"
            )
        live = os.path.abspath(getattr(self.store, "path", "") or "")
        if live and os.path.abspath(sandbox_path) == live:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "replay sandbox_path names the live store — replays run "
                "against a sandbox store only (V3-43.01)",
            )
        tables = _variant_tables(variant)
        id_list = None if decision_ids is None else list(decision_ids)

        with self.store.read() as conn:
            rows = conn.execute(
                "SELECT decision_id, state_key, routes_json, lane_set_json,"
                " budgets_json, state_json, policy_revision, created_us"
                " FROM routing_decisions WHERE scope_id = ?"
                + (
                    ""
                    if id_list is None
                    else " AND decision_id IN (%s)"
                    % ",".join("?" * len(id_list))
                )
                + " ORDER BY created_us",
                (scope_id,)
                if id_list is None
                else (scope_id, *id_list),
            ).fetchall()
            seeds = None

        t0 = time.monotonic()
        sandbox = _create_sandbox(self.store, sandbox_path)
        try:
            _propagate_runtime(self.store, sandbox)
            with self.store.read() as src, sandbox.tx() as dst:
                seeds = _seed_snapshot(src, dst, scope_id)

            divergences: list[DecisionDivergence] = []
            routes_changed = lanes_added = lanes_removed = 0
            budget_divergences = replayed = not_replayable = 0
            revisions: set[str] = set()
            for row in rows:
                (
                    decision_id, state_key, routes_json, lanes_json,
                    budgets_json, state_json, policy_rev, _created,
                ) = row
                revisions.add(policy_rev)
                try:
                    state = _require_state(
                        safe_json_loads(state_json), decision_id
                    )
                    request = _replay_request(
                        safe_json_loads(state_json), scope_id
                    )
                except (VerbatimError, KeyError, TypeError, ValueError):
                    not_replayable += 1
                    continue
                rs = _ctrl.plan_routes(
                    request, None, None, state, tables=tables
                )
                recorded_routes = tuple(safe_json_loads(routes_json))
                recorded_lanes = tuple(safe_json_loads(lanes_json))
                recorded_budgets = safe_json_loads(budgets_json)
                variant_routes = tuple(r.value for r in rs.routes)
                added = tuple(
                    l for l in rs.lanes if l not in recorded_lanes
                )
                removed = tuple(
                    l for l in recorded_lanes if l not in rs.lanes
                )
                deltas = {
                    k: [recorded_budgets.get(k), rs.budgets.get(k)]
                    for k in set(recorded_budgets) | set(rs.budgets)
                    if recorded_budgets.get(k) != rs.budgets.get(k)
                }
                replayed += 1
                if (
                    variant_routes != recorded_routes
                    or added
                    or removed
                    or deltas
                ):
                    routes_changed += int(
                        variant_routes != recorded_routes
                    )
                    lanes_added += len(added)
                    lanes_removed += len(removed)
                    budget_divergences += int(bool(deltas))
                    divergences.append(
                        DecisionDivergence(
                            decision_id=decision_id,
                            state_key=state_key,
                            baseline_routes=recorded_routes,
                            variant_routes=variant_routes,
                            lanes_added=added,
                            lanes_removed=removed,
                            budget_deltas=deltas,
                        )
                    )
        finally:
            sandbox.close()

        latency_ms = (time.monotonic() - t0) * 1000.0
        run_id = "rr_" + hashlib.sha256(
            f"{scope_id}|{len(rows)}|{now_us()}".encode("utf-8")
        ).hexdigest()[:24]
        report = WhatIfReport(
            run_id=run_id,
            scope_id=scope_id,
            sandbox_ref=sandbox_path,
            baseline_revision=",".join(sorted(revisions)),
            variant=dict(variant or {}),
            decisions_total=len(rows),
            replayed=replayed,
            not_replayable=not_replayable,
            routes_changed=routes_changed,
            lanes_added=lanes_added,
            lanes_removed=lanes_removed,
            budget_divergences=budget_divergences,
            divergences=tuple(divergences),
            latency_ms=round(latency_ms, 3),
        )

        manifest = {
            "scope_id": scope_id,
            "decisions": len(rows),
            "baseline_revisions": sorted(revisions),
            "variant": dict(variant or {}),
            "seeded": seeds or {},
            "sandbox_ref": sandbox_path,
            "simulated_clock": "recorded created_us",
            "stages": list(report.stages_executed),
        }
        with self.store.tx() as conn:
            conn.execute(
                "INSERT INTO replay_runs"
                " (run_id, manifest_json, report_json, sandbox_ref,"
                "  created_us) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    json_dumps(manifest),
                    json_dumps(report.to_dict()),
                    sandbox_path,
                    now_us(),
                ),
            )
        return report

    def paired_execution(
        self,
        scope_id: str,
        requests: Iterable[RecallRequestV3],
        *,
        arms: Iterable[dict],
        sandbox_dir: str,
    ) -> dict[str, Any]:
        """Paired sandbox executions that actually DELIVER each policy's
        context (V3-43.04/43.05) — the only evidence class that supports
        a G8-style quality comparison.

        Every arm restarts from the same production snapshot in its own
        sandbox store (isolated writes), receives the identical declared
        replay authorization, and runs the real ``recall_v3`` delivery
        path over the same request list with equal budgets. What-if
        reports compare *delivered item sets*, not counterfactual labels.
        """
        require_id(scope_id, "scope_id")
        arm_list = [dict(a) for a in arms]
        if not arm_list:
            raise VerbatimError(
                ErrorCode.VALIDATION, "paired_execution needs >= 1 arm"
            )
        names = _check_arm_names(arm_list)
        sandbox_dir = _check_sandbox_dir(self.store, sandbox_dir)
        req_list = list(requests)
        for req in req_list:
            if req.scope_id != scope_id:
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    f"request scope {req.scope_id!r} != replay scope "
                    f"{scope_id!r} — deliveries would see a different "
                    "partition than the seeded snapshot",
                )
        os.makedirs(sandbox_dir, exist_ok=True)
        run_id = "rr_" + hashlib.sha256(
            f"{scope_id}|paired|{now_us()}".encode("utf-8")
        ).hexdigest()[:24]

        from ..retrieval.v3.recall import recall_v3

        # Read the production snapshot ONCE — every arm writes the
        # identical image, so seed equality is structural (V3-43.05),
        # not re-measured per arm and never drifts between arms.
        with self.store.read() as src:
            snapshot, seed_counts, snapshot_digest = _read_snapshot(
                src, scope_id
            )

        per_arm: dict[str, Any] = {}
        for arm in arm_list:
            t0 = time.monotonic()
            path = os.path.join(
                sandbox_dir, f"arm-{arm['name']}-{run_id[3:15]}.db"
            )
            sandbox = _create_sandbox(self.store, path)
            try:
                _propagate_runtime(self.store, sandbox)
                with sandbox.tx() as dst:
                    _write_snapshot(dst, snapshot)
                    _provision_replay_caller(dst, scope_id)
                tables = _variant_tables(arm.get("variant"))
                deliveries: list[dict[str, Any]] = []
                for req in req_list:
                    # identical declared authorization across arms —
                    # the sandbox provisioned replay-lab, not the
                    # original caller (V3-43.05 equal context)
                    replay_req = dataclasses.replace(
                        req,
                        caller_id="replay-lab",
                        purpose="evaluate",
                    )
                    r_t0 = time.monotonic()
                    out = recall_v3(
                        sandbox, replay_req, policy_tables=tables
                    )
                    deliveries.append({
                        "fingerprint": _request_fingerprint(req),
                        "items": list(_delivered_ids(out)),
                        "item_bytes": _delivered_bytes(out),
                        "packs": len(out.packs),
                        "omitted": out.omitted,
                        "abstained": out.abstained,
                        "warnings": list(out.warnings),
                        "latency_ms": round(
                            (time.monotonic() - r_t0) * 1000.0, 3
                        ),
                    })
                with sandbox.read() as sconn:
                    burden = sconn.execute(
                        "SELECT COUNT(*) FROM reviews"
                        " WHERE state = 'open'"
                    ).fetchone()[0]
                per_arm[arm["name"]] = {
                    "variant": arm.get("variant"),
                    "sandbox_ref": path,
                    "deliveries": deliveries,
                    "review_burden": burden,
                    "latency_ms": round((time.monotonic() - t0) * 1000.0, 3),
                }
            finally:
                sandbox.close()

        pairwise: list[dict[str, Any]] = []
        changed = 0
        for i in range(len(arm_list)):
            for j in range(i + 1, len(arm_list)):
                a_name, b_name = names[i], names[j]
                a_del = per_arm[a_name]["deliveries"]
                b_del = per_arm[b_name]["deliveries"]
                for idx, (ra, rb) in enumerate(zip(a_del, b_del)):
                    a_items, b_items = set(ra["items"]), set(rb["items"])
                    diff = {
                        "arms": [a_name, b_name],
                        "request_index": idx,
                        "items_only_a": sorted(a_items - b_items),
                        "items_only_b": sorted(b_items - a_items),
                        "bytes_diff": (
                            [ra["item_bytes"], rb["item_bytes"]]
                            if ra["item_bytes"] != rb["item_bytes"]
                            else None
                        ),
                        "omitted_diff": (
                            [ra["omitted"], rb["omitted"]]
                            if ra["omitted"] != rb["omitted"]
                            else None
                        ),
                        "warnings_diff": (
                            [
                                sorted(set(ra["warnings"]) - set(rb["warnings"])),
                                sorted(set(rb["warnings"]) - set(ra["warnings"])),
                            ]
                            if set(ra["warnings"]) != set(rb["warnings"])
                            else None
                        ),
                        "abstain_diff": (
                            [ra["abstained"], rb["abstained"]]
                            if ra["abstained"] != rb["abstained"]
                            else None
                        ),
                    }
                    if any(v for k, v in diff.items() if k not in ("arms", "request_index")):
                        changed += 1
                        pairwise.append(diff)

        report = {
            "run_id": run_id,
            "kind": "paired_execution",
            "scope_id": scope_id,
            "stages_executed": [
                "snapshot_read", "sandbox_seed", "provision_caller",
                "paired_delivery", "pairwise_diff",
            ],
            "arms": [
                {"name": a["name"], "variant": a.get("variant")}
                for a in arm_list
            ],
            "requests": len(req_list),
            "per_arm": per_arm,
            "pairwise_divergences": pairwise,
            "changed_deliveries": changed,
            "review_burden": {
                n: per_arm[n]["review_burden"] for n in names
            },
            "model_calls": 0,
            "notes": (
                "paired sandbox executions (V3-43.05): every arm restarted "
                "from the same snapshot with isolated writes, identical "
                "declared replay authorization, equal request budgets. "
                "Delivered item sets compared — never counterfactual "
                "outcome labels (V3-43.04). model_calls=0 by construction "
                "(V3-43.06)."
            ),
        }
        manifest = {
            "scope_id": scope_id,
            "kind": "paired_execution",
            "arms": report["arms"],
            "request_fingerprints": [
                d["fingerprint"]
                for d in per_arm[names[0]]["deliveries"]
            ],
            "sandbox_dir": sandbox_dir,
            "snapshot_digest": snapshot_digest,
            "seed_counts": seed_counts,
            "seeds_identical": True,
            "seeds_basis": (
                "shared_snapshot — every arm wrote the identical "
                "read-once image; equality is structural, not measured"
            ),
            "simulated_clock": "live wall clock (no model calls)",
            "stages": report["stages_executed"],
        }
        with self.store.tx() as conn:
            conn.execute(
                "INSERT INTO replay_runs"
                " (run_id, manifest_json, report_json, sandbox_ref,"
                "  created_us) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    json_dumps(manifest),
                    json_dumps(report),
                    sandbox_dir,
                    now_us(),
                ),
            )
        return report

    def paired_admission(
        self,
        scope: Scope,
        payloads: Iterable[dict],
        *,
        arms: Iterable[dict],
        sandbox_dir: str,
        drain_limit: int = 256,
    ) -> dict[str, Any]:
        """Paired WRITE-side arms (V3-43.02): each sandbox ingests the
        same payload stream through the real ``Ingester`` + ``run_pending``
        drain — harvesting, screening, admission, relation discovery, and
        consolidation jobs all execute inside the arm's isolated store.

        Arms differ by ``admission_context`` (the PolicyContext's
        three-valued condition facts — §17); the comparison reports
        admitted claims, lifecycle states, relation edges, quarantine,
        and open review burden per arm (V3-43.03).
        """
        require_id(getattr(scope, "profile_id", None), "scope.profile_id")
        arm_list = [dict(a) for a in arms]
        if not arm_list:
            raise VerbatimError(
                ErrorCode.VALIDATION, "paired_admission needs >= 1 arm"
            )
        names = _check_arm_names(arm_list)
        sandbox_dir = _check_sandbox_dir(self.store, sandbox_dir)
        payload_list = [dict(p) for p in payloads]
        os.makedirs(sandbox_dir, exist_ok=True)
        scope_id = scope_id_for(self.store, scope)
        run_id = "rr_" + hashlib.sha256(
            f"{scope_id}|admission|{now_us()}".encode("utf-8")
        ).hexdigest()[:24]

        from ..core.types import Provenance, SourceKind, SourceEnvelope
        from ..core.types_v3 import (
            EnvelopeKind,
            Perspective,
            SourceEnvelopeV3,
        )
        from ..evidence.envelopes import ingest_envelope
        from ..ingest import Ingester

        # Read the production snapshot ONCE — every arm writes the
        # identical image, so seed equality is structural (V3-43.05),
        # not re-measured per arm and never drifts between arms.
        with self.store.read() as src:
            snapshot, seed_counts, snapshot_digest = _read_snapshot(
                src, scope_id
            )

        per_arm: dict[str, Any] = {}
        for arm in arm_list:
            t0 = time.monotonic()
            path = os.path.join(
                sandbox_dir, f"arm-{arm['name']}-{run_id[3:15]}.db"
            )
            sandbox = _create_sandbox(self.store, path)
            try:
                _propagate_runtime(self.store, sandbox)
                with sandbox.tx() as dst:
                    _write_snapshot(dst, snapshot)
                    _provision_replay_caller(dst, scope_id)
                cfg = getattr(self.store, "cfg", None) or VerbatimConfig()
                ingester = Ingester(sandbox, cfg, judge=None)
                actx = (arm.get("variant") or {}).get("admission_context")
                if actx is not None:
                    ingester.policy = dataclasses.replace(
                        ingester.policy, admission_context=dict(actx)
                    )
                with sandbox.read() as sconn:
                    pre_job_ids = {
                        r[0] for r in sconn.execute(
                            "SELECT job_id FROM jobs"
                        )
                    }
                receipts = []
                channels: list[str] = []
                accepted_sources: list[str] = []
                for p in payload_list:
                    if p.get("payload") is None and p.get("text") is None:
                        raise VerbatimError(
                            ErrorCode.VALIDATION,
                            "each payload needs 'text' or 'payload' bytes",
                        )
                    payload_bytes = _payload_bytes(p)
                    v3_kind = p.get("kind")
                    if v3_kind is not None:
                        # V3 channel (M2): screening, security labels, and
                        # quarantine holds run inside ingest_envelope; the
                        # HARVEST/ADMIT chain lands on the job queue and
                        # drains below — the production write path end to
                        # end, not a replay shortcut.
                        env3 = SourceEnvelopeV3(
                            kind=EnvelopeKind(v3_kind),
                            scope_id=scope_id,
                            actor_principal=(
                                p.get("actor_principal") or "replay-lab"
                            ),
                            perspective=Perspective(
                                asserter=p.get("speaker_id"),
                            ),
                            event_us=int(p.get("event_us", 1)),
                            receipt_us=int(p.get("receipt_us", 0)),
                            content=payload_bytes,
                            media_type=p.get("media_type", "text/plain"),
                            external_id=p.get("external_id"),
                            metadata=dict(p.get("metadata") or {}),
                        )
                        with sandbox.tx() as wconn:
                            r3 = ingest_envelope(wconn, sandbox, env3)
                        receipts.append(r3)
                        accepted_sources.append(r3.source_id)
                        channels.append("v3")
                    else:
                        env = SourceEnvelope(
                            origin="replay",
                            source_kind=SourceKind.IMPORT,
                            scope=scope,
                            speaker_id=p.get("speaker_id"),
                            payload=payload_bytes,
                            event_us=int(p.get("event_us", 1)),
                            captured_us=int(p.get("captured_us", 1)),
                            provenance=Provenance(
                                p.get("provenance", "direct_user")
                            ),
                            source_id=p.get("source_id"),
                            external_id=p.get("external_id"),
                        )
                        r2 = ingester.ingest(env)
                        receipts.append(r2)
                        accepted_sources.extend(r2.accepted)
                        channels.append("v2")
                drained = ingester.run_pending(limit=drain_limit)
                with sandbox.read() as sconn:
                    claims = sconn.execute(
                        "SELECT c.claim_id, cr.state, cr.object_json"
                        " FROM claims c"
                        " JOIN claim_revisions cr"
                        "   ON cr.claim_id = c.claim_id"
                        "  AND cr.revision = ("
                        "    SELECT MAX(revision) FROM claim_revisions"
                        "    WHERE claim_id = c.claim_id)"
                        " WHERE c.scope_id = ? ORDER BY c.claim_id",
                        (scope_id,),
                    ).fetchall()
                    edges = sconn.execute(
                        "SELECT edge_type, COUNT(*) FROM edges"
                        " WHERE scope_id = ? GROUP BY edge_type",
                        (scope_id,),
                    ).fetchall()
                    reviews = sconn.execute(
                        "SELECT COUNT(*) FROM reviews"
                        " WHERE scope_id = ? AND state = 'open'",
                        (scope_id,),
                    ).fetchone()[0]
                    quarantined = sconn.execute(
                        "SELECT object_kind || ':' || object_id"
                        " FROM quarantine WHERE scope_id = ?",
                        (scope_id,),
                    ).fetchall()
                    # jobs created by THIS arm's ingest that the drain
                    # executed — seeded snapshot jobs can't masquerade
                    # as arm work (stage honesty, V3-43.02).
                    new_jobs = sconn.execute(
                        "SELECT kind, state FROM jobs"
                        + (
                            " WHERE job_id NOT IN (%s)"
                            % ",".join("?" * len(pre_job_ids))
                            if pre_job_ids else ""
                        ),
                        tuple(pre_job_ids),
                    ).fetchall()
                per_arm[arm["name"]] = {
                    "variant": arm.get("variant"),
                    "sandbox_ref": path,
                    "channels": channels,
                    "sources_accepted": len(accepted_sources),
                    "duplicates": sum(
                        1 for r in receipts
                        if getattr(r, "duplicate", False)
                    ),
                    "jobs_drained": drained,
                    "job_kinds_executed": sorted(
                        {k for k, s in new_jobs if s == "succeeded"}
                    ),
                    "job_kinds_failed": sorted(
                        {k for k, s in new_jobs if s == "failed"}
                    ),
                    "claims": [
                        {
                            "claim_id": c,
                            "state": s,
                            # ids are minted per-arm — content digest is
                            # the comparable identity across arms
                            "content_digest": hashlib.sha256(
                                (obj or "").encode()
                            ).hexdigest()[:12],
                        }
                        for c, s, obj in claims
                    ],
                    "edges": {t: n for t, n in edges},
                    "review_burden": reviews,
                    "quarantined": sorted(q[0] for q in quarantined),
                    "latency_ms": round((time.monotonic() - t0) * 1000.0, 3),
                }
            finally:
                sandbox.close()

        pairwise: list[dict[str, Any]] = []
        changed = 0
        for i in range(len(arm_list)):
            for j in range(i + 1, len(arm_list)):
                a_name, b_name = names[i], names[j]
                a, b = per_arm[a_name], per_arm[b_name]
                # claim ids are minted per-arm — compare on
                # (content_digest, state) pairs
                a_claims = {
                    c["content_digest"]: c["state"] for c in a["claims"]
                }
                b_claims = {
                    c["content_digest"]: c["state"] for c in b["claims"]
                }
                diff = {
                    "arms": [a_name, b_name],
                    "claims_only_a": sorted(
                        set(a_claims) - set(b_claims)
                    ),
                    "claims_only_b": sorted(
                        set(b_claims) - set(a_claims)
                    ),
                    "state_diffs": sorted(
                        cid for cid in set(a_claims) & set(b_claims)
                        if a_claims[cid] != b_claims[cid]
                    ),
                    "edge_diff": (
                        {"a": a["edges"], "b": b["edges"]}
                        if a["edges"] != b["edges"] else None
                    ),
                    "review_burden_diff": (
                        [a["review_burden"], b["review_burden"]]
                        if a["review_burden"] != b["review_burden"]
                        else None
                    ),
                    "quarantine_diff": (
                        {"a": a["quarantined"], "b": b["quarantined"]}
                        if a["quarantined"] != b["quarantined"]
                        else None
                    ),
                }
                if any(v for k, v in diff.items()
                       if k != "arms" and v):
                    changed += 1
                    pairwise.append(diff)

        report = {
            "run_id": run_id,
            "kind": "paired_admission",
            "scope_id": scope_id,
            "stages_executed": [
                "snapshot_read", "sandbox_seed", "provision_caller",
                "envelope_ingest", "job_drain", "pairwise_diff",
            ],
            "arms": [
                {"name": a["name"], "variant": a.get("variant")}
                for a in arm_list
            ],
            "payloads": len(payload_list),
            "per_arm": per_arm,
            "pairwise_divergences": pairwise,
            "changed_admissions": changed,
            "review_burden": {
                n: per_arm[n]["review_burden"] for n in names
            },
            "model_calls": 0,
            "notes": (
                "paired write-side arms (V3-43.02): identical payload "
                "streams through the real Ingester + run_pending drain "
                "in isolated sandboxes. Admissions, edges, quarantine and "
                "review burden measured post-drain — never counterfactual "
                "labels (V3-43.04). model_calls=0 (V3-43.06)."
            ),
        }
        manifest = {
            "scope_id": scope_id,
            "kind": "paired_admission",
            "arms": report["arms"],
            "payload_digests": [
                hashlib.sha256(_payload_bytes(p)).hexdigest()[:16]
                for p in payload_list
            ],
            "sandbox_dir": sandbox_dir,
            "snapshot_digest": snapshot_digest,
            "seed_counts": seed_counts,
            "seeds_identical": True,
            "seeds_basis": (
                "shared_snapshot — every arm wrote the identical "
                "read-once image; equality is structural, not measured"
            ),
            "job_kinds_executed": sorted(
                set().union(
                    *(
                        set(per_arm[n]["job_kinds_executed"])
                        for n in names
                    )
                )
            ),
            "stages": report["stages_executed"],
        }
        with self.store.tx() as conn:
            conn.execute(
                "INSERT INTO replay_runs"
                " (run_id, manifest_json, report_json, sandbox_ref,"
                "  created_us) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    json_dumps(manifest),
                    json_dumps(report),
                    sandbox_dir,
                    now_us(),
                ),
            )
        return report


__all__ = ["ReplayLab"]
