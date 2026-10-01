"""Repositories for schema v2 relations (SPEC_V2 §37-38).

Same contract as ``repos.py``: mutators take the caller's transaction
``conn`` so higher layers commit effects atomically; reads return plain
``dict`` snapshots — mutable rows never escape (SPEC §8). All SQL is
parameterized over fixed column allowlists (SPEC §41).
"""

from __future__ import annotations

import hmac
import sqlite3
from typing import Any, Optional, Sequence

from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
)
from .repos import (
    _dump,
    _json_parse,
    _json_text,
    _next_event_seq,
    _require_int,
    _require_str,
    _row,
    _rows,
)
from .store import Store


# ----------------------------------------------------------------------
# source views (SPEC_V2 §37)
# ----------------------------------------------------------------------


class SourceViewsRepo:
    """Primary + derived views of a source revision.

    The ``primary`` view carries no payload copy — it points at
    ``source_revisions.payload`` (SPEC_V2 §37: canonical bytes are stored
    exactly once). Derived views may carry their own derived bytes.
    """

    _KINDS = ("primary", "normalized", "redacted", "extracted", "transcript")

    def __init__(self, store: Store) -> None:
        self._store = store

    def ensure_primary(
        self, conn: sqlite3.Connection, source_id: str, revision: int
    ) -> None:
        """Register the primary view for a persisted revision (idempotent)."""
        require_id(source_id, "source_id")
        _require_int(revision, "revision", minimum=1)
        conn.execute(
            "INSERT OR IGNORE INTO source_views"
            "(source_id, revision, view_id, media_type, view_kind, locator_json)"
            " VALUES (?, ?, 'primary', 'text/plain', 'primary', '{}')",
            (source_id, revision),
        )

    def insert_derived(
        self,
        conn: sqlite3.Connection,
        source_id: str,
        revision: int,
        view_id: str,
        *,
        media_type: str,
        view_kind: str,
        transformer_revision: str,
        derived_bytes: Optional[bytes] = None,
        locator_json: Any = None,
    ) -> str:
        require_id(source_id, "source_id")
        require_id(view_id, "view_id")
        _require_int(revision, "revision", minimum=1)
        _require_str(media_type, "media_type")
        _require_str(transformer_revision, "transformer_revision")
        if view_kind not in self._KINDS:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid view_kind {view_kind!r}")
        if view_kind == "primary":
            raise VerbatimError(
                ErrorCode.VALIDATION, "primary view is registered via ensure_primary"
            )
        digest = self._store.hmac(derived_bytes) if derived_bytes is not None else None
        conn.execute(
            "INSERT INTO source_views"
            "(source_id, revision, view_id, media_type, view_kind,"
            " transformer_revision, locator_json, derived_bytes, integrity_digest)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                revision,
                view_id,
                media_type,
                view_kind,
                transformer_revision,
                _json_text(locator_json, "locator_json") or "{}",
                derived_bytes,
                digest,
            ),
        )
        return view_id

    def _verified(self, row: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Re-check ``integrity_digest`` over the row's derived bytes.

        A NULL digest marks a legacy or byte-less view row (primary views
        carry no ``derived_bytes``; purged views have both columns nulled)
        — nothing to check against, so it is returned unverified rather
        than condemned. Bytes that no longer match their digest are
        corruption, never content (``STORE_CORRUPT``).
        """
        if row is None:
            return None
        derived = row.get("derived_bytes")
        digest = row.get("integrity_digest")
        if derived is not None and digest is not None and not hmac.compare_digest(
            self._store.hmac(bytes(derived)), bytes(digest)
        ):
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"source view {row.get('view_id')}"
                f" on {row.get('source_id')}@{row.get('revision')}"
                " fails integrity check",
            )
        return row

    def get(
        self, source_id: str, revision: int, view_id: str
    ) -> Optional[dict[str, Any]]:
        with self._store.read() as conn:
            row = _row(
                conn.execute(
                    "SELECT * FROM source_views"
                    " WHERE source_id = ? AND revision = ? AND view_id = ?",
                    (source_id, revision, view_id),
                )
            )
        return self._verified(row)

    def for_revision(
        self, source_id: str, revision: int
    ) -> list[dict[str, Any]]:
        with self._store.read() as conn:
            rows = _rows(
                conn.execute(
                    "SELECT * FROM source_views"
                    " WHERE source_id = ? AND revision = ? ORDER BY view_id",
                    (source_id, revision),
                )
            )
        return [self._verified(r) for r in rows if r is not None]


# ----------------------------------------------------------------------
# context groups (SPEC_V2 §12, §37)
# ----------------------------------------------------------------------


class ContextGroupsRepo:
    """Harvested evidence groups: primary spans plus required context."""

    _ROLES = ("primary", "attribution", "antecedent", "condition", "negation", "temporal")
    _COMPLETENESS = ("complete", "partial", "deferred")

    def __init__(self, store: Store) -> None:
        self._store = store

    def create(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        source_id: str,
        revision: int,
        *,
        parser_version: str,
        operation_key: str,
        completeness: str = "complete",
        group_id: Optional[str] = None,
    ) -> str:
        """Insert a context group; idempotent on (scope_id, operation_key)."""
        require_id(scope_id, "scope_id")
        require_id(source_id, "source_id")
        _require_int(revision, "revision", minimum=1)
        _require_str(parser_version, "parser_version")
        _require_str(operation_key, "operation_key")
        if completeness not in self._COMPLETENESS:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"invalid completeness {completeness!r}"
            )
        gid = group_id or new_id()
        require_id(gid, "group_id")
        cur = conn.execute(
            "INSERT OR IGNORE INTO context_groups"
            "(group_id, scope_id, source_id, revision, parser_version,"
            " operation_key, completeness, recorded_from)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                gid,
                scope_id,
                source_id,
                revision,
                parser_version,
                operation_key,
                completeness,
                _next_event_seq(conn),
            ),
        )
        if cur.rowcount == 0:
            row = conn.execute(
                "SELECT group_id FROM context_groups"
                " WHERE scope_id = ? AND operation_key = ?",
                (scope_id, operation_key),
            ).fetchone()
            return row[0]
        return gid

    def add_member(
        self,
        conn: sqlite3.Connection,
        group_id: str,
        span_id: str,
        *,
        role: str = "primary",
        required: bool = True,
        ord: int = 0,
        dependency_reason: Optional[str] = None,
    ) -> None:
        require_id(group_id, "group_id")
        require_id(span_id, "span_id")
        if role not in self._ROLES:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid member role {role!r}")
        conn.execute(
            "INSERT OR REPLACE INTO context_members"
            "(group_id, span_id, role, required, ord, dependency_reason)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (group_id, span_id, role, 1 if required else 0, ord, dependency_reason),
        )

    def get(self, group_id: str) -> Optional[dict[str, Any]]:
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM context_groups WHERE group_id = ?", (group_id,)
                )
            )

    def members(self, group_id: str) -> list[dict[str, Any]]:
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM context_members WHERE group_id = ?"
                    " ORDER BY ord, span_id",
                    (group_id,),
                )
            )

    def for_source(self, source_id: str, revision: int) -> list[dict[str, Any]]:
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM context_groups"
                    " WHERE source_id = ? AND revision = ? ORDER BY group_id",
                    (source_id, revision),
                )
            )

    def supersede(self, conn: sqlite3.Connection, group_id: str, event_seq: int) -> None:
        """Close the group's known-at interval (source superseded/edited)."""
        require_id(group_id, "group_id")
        _require_int(event_seq, "event_seq", minimum=1)
        conn.execute(
            "UPDATE context_groups SET recorded_until = ?"
            " WHERE group_id = ? AND recorded_until IS NULL",
            (event_seq, group_id),
        )


# ----------------------------------------------------------------------
# predicate registry (SPEC_V2 §13, §37)
# ----------------------------------------------------------------------


class PredicateRegistryRepo:
    """Versioned predicate interpretation rules."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def register(
        self,
        conn: sqlite3.Connection,
        namespace: str,
        name: str,
        *,
        version: int = 1,
        value_type: str = "literal",
        units: Optional[str] = None,
        cardinality: str = "single",
        mutable: bool = True,
        sensitive: bool = False,
        authority_policy: str = "default",
        comparison_rules: Any = None,
        artifact_digest: Optional[bytes] = None,
    ) -> None:
        _require_str(namespace, "namespace")
        _require_str(name, "name")
        _require_int(version, "version", minimum=1)
        if cardinality not in ("single", "set"):
            raise VerbatimError(
                ErrorCode.VALIDATION, f"invalid cardinality {cardinality!r}"
            )
        conn.execute(
            "INSERT INTO predicate_definitions"
            "(namespace, name, version, value_type, units, cardinality, mutable,"
            " sensitive, authority_policy, comparison_rules, artifact_digest)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                namespace,
                name,
                version,
                _require_str(value_type, "value_type"),
                units,
                cardinality,
                1 if mutable else 0,
                1 if sensitive else 0,
                _require_str(authority_policy, "authority_policy"),
                _json_text(comparison_rules, "comparison_rules") or "{}",
                artifact_digest,
            ),
        )

    def get(
        self, namespace: str, name: str, version: Optional[int] = None
    ) -> Optional[dict[str, Any]]:
        """Latest version when ``version`` is None."""
        with self._store.read() as conn:
            if version is not None:
                return _row(
                    conn.execute(
                        "SELECT * FROM predicate_definitions"
                        " WHERE namespace = ? AND name = ? AND version = ?",
                        (namespace, name, version),
                    )
                )
            return _row(
                conn.execute(
                    "SELECT * FROM predicate_definitions"
                    " WHERE namespace = ? AND name = ?"
                    " ORDER BY version DESC LIMIT 1",
                    (namespace, name),
                )
            )

    def all_current(self) -> list[dict[str, Any]]:
        """One row per (namespace, name) at its max version."""
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT p.* FROM predicate_definitions p"
                    " JOIN (SELECT namespace, name, MAX(version) v"
                    "       FROM predicate_definitions GROUP BY namespace, name) m"
                    "   ON p.namespace = m.namespace AND p.name = m.name"
                    "  AND p.version = m.v"
                    " ORDER BY p.namespace, p.name"
                )
            )


# ----------------------------------------------------------------------
# evidence families (SPEC_V2 §18, §37)
# ----------------------------------------------------------------------


class EvidenceFamiliesRepo:
    """Copies/derivatives of one underlying statement."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def create(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        origin_kind: str,
        origin_id: str,
        *,
        family_id: Optional[str] = None,
    ) -> str:
        require_id(scope_id, "scope_id")
        _require_str(origin_kind, "origin_kind")
        require_id(origin_id, "origin_id")
        fid = family_id or new_id()
        require_id(fid, "family_id")
        conn.execute(
            "INSERT INTO evidence_families"
            "(family_id, scope_id, origin_kind, origin_id, created_event)"
            " VALUES (?, ?, ?, ?, ?)",
            (fid, scope_id, origin_kind, origin_id, _next_event_seq(conn)),
        )
        return fid

    def add_member(
        self,
        conn: sqlite3.Connection,
        family_id: str,
        object_kind: str,
        object_id: str,
        *,
        role: str = "copy",
    ) -> None:
        require_id(family_id, "family_id")
        _require_str(object_kind, "object_kind")
        require_id(object_id, "object_id")
        conn.execute(
            "INSERT OR IGNORE INTO family_members"
            "(family_id, object_kind, object_id, role) VALUES (?, ?, ?, ?)",
            (family_id, object_kind, object_id, role),
        )

    def members(self, family_id: str) -> list[dict[str, Any]]:
        require_id(family_id, "family_id")
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM family_members WHERE family_id = ?",
                    (family_id,),
                )
            )

    def families_for(self, object_kind: str, object_id: str) -> list[dict[str, Any]]:
        """Families containing this object — used for duplicate-collapse."""
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT f.* FROM evidence_families f"
                    " JOIN family_members m ON m.family_id = f.family_id"
                    " WHERE m.object_kind = ? AND m.object_id = ?",
                    (object_kind, object_id),
                )
            )


# ----------------------------------------------------------------------
# episodes (SPEC_V2 §21, §37)
# ----------------------------------------------------------------------


class EpisodesRepo:
    """Task/conversation/event groupings with temporal membership."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def create(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        *,
        kind: str = "task",
        host_task_id: Optional[str] = None,
        host_session_id: Optional[str] = None,
        parent_episode_id: Optional[str] = None,
        label: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        require_id(scope_id, "scope_id")
        _require_str(kind, "kind")
        eid = episode_id or new_id()
        require_id(eid, "episode_id")
        conn.execute(
            "INSERT INTO episodes"
            "(episode_id, scope_id, revision, host_task_id, host_session_id,"
            " parent_episode_id, kind, label, recorded_from)"
            " VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)",
            (
                eid,
                scope_id,
                host_task_id,
                host_session_id,
                parent_episode_id,
                kind,
                label,
                _next_event_seq(conn),
            ),
        )
        return eid

    def get(self, episode_id: str) -> Optional[dict[str, Any]]:
        require_id(episode_id, "episode_id")
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
                )
            )

    def add_member(
        self,
        conn: sqlite3.Connection,
        episode_id: str,
        object_kind: str,
        object_id: str,
        *,
        ord: int = 0,
    ) -> None:
        require_id(episode_id, "episode_id")
        _require_str(object_kind, "object_kind")
        require_id(object_id, "object_id")
        conn.execute(
            "INSERT OR REPLACE INTO episode_members"
            "(episode_id, object_kind, object_id, ord, recorded_from)"
            " VALUES (?, ?, ?, ?, ?)",
            (episode_id, object_kind, object_id, ord, _next_event_seq(conn)),
        )

    def remove_member(
        self,
        conn: sqlite3.Connection,
        episode_id: str,
        object_kind: str,
        object_id: str,
    ) -> None:
        conn.execute(
            "UPDATE episode_members SET recorded_until = ?"
            " WHERE episode_id = ? AND object_kind = ? AND object_id = ?"
            "   AND recorded_until IS NULL",
            (_next_event_seq(conn), episode_id, object_kind, object_id),
        )

    def members(
        self, episode_id: str, *, as_of_seq: Optional[int] = None
    ) -> list[dict[str, Any]]:
        require_id(episode_id, "episode_id")
        with self._store.read() as conn:
            if as_of_seq is None:
                return _rows(
                    conn.execute(
                        "SELECT * FROM episode_members WHERE episode_id = ?"
                        " AND recorded_until IS NULL ORDER BY ord, object_id",
                        (episode_id,),
                    )
                )
            return _rows(
                conn.execute(
                    "SELECT * FROM episode_members WHERE episode_id = ?"
                    " AND recorded_from <= ?"
                    " AND (recorded_until IS NULL OR recorded_until > ?)"
                    " ORDER BY ord, object_id",
                    (episode_id, as_of_seq, as_of_seq),
                )
            )

    def for_scope(self, scope_id: str, *, kind: Optional[str] = None) -> list[dict[str, Any]]:
        require_id(scope_id, "scope_id")
        with self._store.read() as conn:
            if kind:
                return _rows(
                    conn.execute(
                        "SELECT * FROM episodes WHERE scope_id = ? AND kind = ?"
                        " AND recorded_until IS NULL ORDER BY episode_id",
                        (scope_id, kind),
                    )
                )
            return _rows(
                conn.execute(
                    "SELECT * FROM episodes WHERE scope_id = ?"
                    " AND recorded_until IS NULL ORDER BY episode_id",
                    (scope_id,),
                )
            )


# ----------------------------------------------------------------------
# procedures (SPEC_V2 §23, §37)
# ----------------------------------------------------------------------


class ProceduresRepo:
    """Evidence-backed task experience: steps, environment, outcomes."""

    _STATES = ("proposed", "active", "review", "retired")
    _OUTCOMES = ("success", "failure", "partial", "unknown")

    def __init__(self, store: Store) -> None:
        self._store = store

    def create(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        task_label: str,
        *,
        environment: Any = None,
        condition: Any = None,
        state: str = "proposed",
        procedure_id: Optional[str] = None,
    ) -> str:
        require_id(scope_id, "scope_id")
        _require_str(task_label, "task_label")
        if state not in self._STATES:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid state {state!r}")
        pid = procedure_id or new_id()
        require_id(pid, "procedure_id")
        conn.execute(
            "INSERT INTO procedures"
            "(procedure_id, scope_id, revision, task_label, state,"
            " environment_json, condition_json, recorded_from)"
            " VALUES (?, ?, 1, ?, ?, ?, ?, ?)",
            (
                pid,
                scope_id,
                task_label,
                state,
                _json_text(environment, "environment_json") or "{}",
                _json_text(condition, "condition_json"),
                _next_event_seq(conn),
            ),
        )
        return pid

    def get(self, procedure_id: str) -> Optional[dict[str, Any]]:
        require_id(procedure_id, "procedure_id")
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM procedures WHERE procedure_id = ?", (procedure_id,)
                )
            )

    def add_step(
        self,
        conn: sqlite3.Connection,
        procedure_id: str,
        revision: int,
        step_no: int,
        description: str,
        *,
        span_id: Optional[str] = None,
        precondition: Any = None,
        hazard: Optional[str] = None,
        verification: Optional[str] = None,
    ) -> None:
        require_id(procedure_id, "procedure_id")
        _require_int(revision, "revision", minimum=1)
        _require_int(step_no, "step_no")
        _require_str(description, "description")
        conn.execute(
            "INSERT INTO procedure_steps"
            "(procedure_id, revision, step_no, span_id, description,"
            " precondition_json, hazard, verification)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                procedure_id,
                revision,
                step_no,
                span_id,
                description,
                _json_text(precondition, "precondition_json"),
                hazard,
                verification,
            ),
        )

    def steps(self, procedure_id: str, revision: int) -> list[dict[str, Any]]:
        require_id(procedure_id, "procedure_id")
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM procedure_steps"
                    " WHERE procedure_id = ? AND revision = ? ORDER BY step_no",
                    (procedure_id, revision),
                )
            )

    def set_state(
        self, conn: sqlite3.Connection, procedure_id: str, state: str, row_version: int
    ) -> None:
        """Expected-version state transition (fencing)."""
        require_id(procedure_id, "procedure_id")
        if state not in self._STATES:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid state {state!r}")
        _require_int(row_version, "row_version", minimum=1)
        cur = conn.execute(
            "UPDATE procedures SET state = ?, row_version = row_version + 1"
            " WHERE procedure_id = ? AND row_version = ?",
            (state, procedure_id, row_version),
        )
        if cur.rowcount == 0:
            raise VerbatimError(
                ErrorCode.STALE_PROPOSAL, "procedure row_version stale or missing"
            )

    def record_outcome(
        self,
        conn: sqlite3.Connection,
        procedure_id: str,
        revision: int,
        *,
        outcome: str,
        checker: str,
        checked_artifact: Optional[str] = None,
        detail: Any = None,
        recorded_us: int = 0,
    ) -> str:
        require_id(procedure_id, "procedure_id")
        if outcome not in self._OUTCOMES:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid outcome {outcome!r}")
        _require_str(checker, "checker")
        rid = new_id()
        conn.execute(
            "INSERT INTO outcome_receipts"
            "(receipt_id, procedure_id, revision, outcome, checker,"
            " checked_artifact, detail_json, recorded_us)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                procedure_id,
                revision,
                outcome,
                checker,
                checked_artifact,
                _json_text(detail, "detail_json") or "{}",
                recorded_us,
            ),
        )
        return rid

    def outcomes(self, procedure_id: str) -> list[dict[str, Any]]:
        require_id(procedure_id, "procedure_id")
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM outcome_receipts WHERE procedure_id = ?"
                    " ORDER BY recorded_us",
                    (procedure_id,),
                )
            )

    def for_scope(self, scope_id: str, *, state: Optional[str] = None) -> list[dict[str, Any]]:
        require_id(scope_id, "scope_id")
        with self._store.read() as conn:
            if state:
                return _rows(
                    conn.execute(
                        "SELECT * FROM procedures WHERE scope_id = ? AND state = ?"
                        " AND recorded_until IS NULL ORDER BY procedure_id",
                        (scope_id, state),
                    )
                )
            return _rows(
                conn.execute(
                    "SELECT * FROM procedures WHERE scope_id = ?"
                    " AND recorded_until IS NULL ORDER BY procedure_id",
                    (scope_id,),
                )
            )


# ----------------------------------------------------------------------
# prospective records (SPEC_V2 §22, §37)
# ----------------------------------------------------------------------


class ProspectiveRepo:
    """Plans, deadlines, and recurring intentions — never completed facts."""

    _STATUSES = ("planned", "in_progress", "completed", "cancelled", "overdue", "unknown")

    def __init__(self, store: Store) -> None:
        self._store = store

    def create(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        owner_id: str,
        intention_text: str,
        *,
        due_us: Optional[int] = None,
        recurrence: Any = None,
        claim_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> str:
        require_id(scope_id, "scope_id")
        require_id(owner_id, "owner_id")
        _require_str(intention_text, "intention_text")
        rid = record_id or new_id()
        require_id(rid, "record_id")
        conn.execute(
            "INSERT INTO prospective_records"
            "(record_id, scope_id, revision, claim_id, episode_id, owner_id,"
            " intention_text, due_us, recurrence_json, status, recorded_from)"
            " VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, 'planned', ?)",
            (
                rid,
                scope_id,
                claim_id,
                episode_id,
                owner_id,
                intention_text,
                due_us,
                _json_text(recurrence, "recurrence_json"),
                _next_event_seq(conn),
            ),
        )
        return rid

    def get(self, record_id: str) -> Optional[dict[str, Any]]:
        require_id(record_id, "record_id")
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM prospective_records WHERE record_id = ?",
                    (record_id,),
                )
            )

    def set_status(
        self, conn: sqlite3.Connection, record_id: str, status: str
    ) -> None:
        require_id(record_id, "record_id")
        if status not in self._STATUSES:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid status {status!r}")
        cur = conn.execute(
            "UPDATE prospective_records SET status = ?"
            " WHERE record_id = ? AND recorded_until IS NULL",
            (status, record_id),
        )
        if cur.rowcount == 0:
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "record not found")

    def due_before(
        self, scope_id: str, before_us: int, *, statuses: Sequence[str] = ("planned", "in_progress", "overdue")
    ) -> list[dict[str, Any]]:
        require_id(scope_id, "scope_id")
        marks = ",".join("?" for _ in statuses)
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM prospective_records"
                    " WHERE scope_id = ? AND due_us IS NOT NULL AND due_us <= ?"
                    f" AND status IN ({marks}) AND recorded_until IS NULL"
                    " ORDER BY due_us",
                    (scope_id, before_us, *statuses),
                )
            )


# ----------------------------------------------------------------------
# artifacts (SPEC_V2 §37, §48)
# ----------------------------------------------------------------------


class ArtifactsRepo:
    """Approved non-text attachments; availability fencing for purge."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def register(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        media_type: str,
        digest: bytes,
        locator: str,
        *,
        size_bytes: int = 0,
        artifact_id: Optional[str] = None,
    ) -> str:
        require_id(scope_id, "scope_id")
        _require_str(media_type, "media_type")
        _require_str(locator, "locator")
        aid = artifact_id or new_id()
        require_id(aid, "artifact_id")
        conn.execute(
            "INSERT INTO artifacts"
            "(artifact_id, scope_id, media_type, digest, locator, size_bytes,"
            " availability, created_event)"
            " VALUES (?, ?, ?, ?, ?, ?, 'available', ?)",
            (aid, scope_id, media_type, digest, locator, size_bytes, _next_event_seq(conn)),
        )
        return aid

    def get(self, artifact_id: str) -> Optional[dict[str, Any]]:
        require_id(artifact_id, "artifact_id")
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
                )
            )

    def set_availability(
        self, conn: sqlite3.Connection, artifact_id: str, availability: str
    ) -> None:
        require_id(artifact_id, "artifact_id")
        if availability not in ("available", "suppressed", "purged"):
            raise VerbatimError(
                ErrorCode.VALIDATION, f"invalid availability {availability!r}"
            )
        conn.execute(
            "UPDATE artifacts SET availability = ? WHERE artifact_id = ?",
            (availability, artifact_id),
        )

    def link(
        self,
        conn: sqlite3.Connection,
        artifact_id: str,
        object_kind: str,
        object_id: str,
        *,
        link_kind: str = "attachment",
    ) -> None:
        require_id(artifact_id, "artifact_id")
        _require_str(object_kind, "object_kind")
        require_id(object_id, "object_id")
        conn.execute(
            "INSERT OR IGNORE INTO artifact_links"
            "(artifact_id, object_kind, object_id, link_kind) VALUES (?, ?, ?, ?)",
            (artifact_id, object_kind, object_id, link_kind),
        )

    def links_for(self, object_kind: str, object_id: str) -> list[dict[str, Any]]:
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM artifact_links"
                    " WHERE object_kind = ? AND object_id = ?",
                    (object_kind, object_id),
                )
            )


# ----------------------------------------------------------------------
# operations — stable idempotence keys (SPEC_V2 §38, §39)
# ----------------------------------------------------------------------


class OperationsRepo:
    """The idempotence ledger: one durable receipt per (scope, operation_key).

    Protocol: ``lookup`` first — a committed row means the effect already
    landed (replay its receipt, never re-apply). ``record`` writes the row
    inside the SAME transaction as the effect itself, so an operation and
    its receipt commit or roll back together (V2-39: a retry after a lost
    response must observe the committed receipt).
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def lookup(
        self, conn: sqlite3.Connection, scope_id: str, operation_key: str
    ) -> Optional[dict[str, Any]]:
        require_id(scope_id, "scope_id")
        _require_str(operation_key, "operation_key")
        return _row(
            conn.execute(
                "SELECT * FROM operations"
                " WHERE scope_id = ? AND operation_key = ?",
                (scope_id, operation_key),
            )
        )

    def check(
        self, conn: sqlite3.Connection, scope_id: str, operation_key: str, input_digest: bytes
    ) -> Optional[dict[str, Any]]:
        """Return the existing receipt for replay, or raise on key conflict.

        Same key + same digest → the caller replays the stored receipt.
        Same key + different digest → a genuine idempotence conflict: the
        caller must not silently re-run under a recycled key (V2-39.07).
        """
        row = self.lookup(conn, scope_id, operation_key)
        if row is None:
            return None
        if bytes(row["input_digest"]) != bytes(input_digest):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "operation_key reused with a different input digest",
            )
        return row

    def record(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        operation_key: str,
        *,
        input_digest: bytes,
        effect_kind: str,
        receipt: Any = None,
        committed_event: Optional[int] = None,
    ) -> None:
        """Persist the receipt inside the effect's own transaction."""
        require_id(scope_id, "scope_id")
        _require_str(operation_key, "operation_key")
        _require_str(effect_kind, "effect_kind")
        conn.execute(
            "INSERT INTO operations"
            "(scope_id, operation_key, input_digest, effect_kind,"
            " committed_event, receipt_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                scope_id,
                operation_key,
                bytes(input_digest),
                effect_kind,
                committed_event if committed_event is not None else _next_event_seq(conn),
                _json_text(receipt, "receipt_json") or "{}",
            ),
        )

    def for_scope(self, scope_id: str) -> list[dict[str, Any]]:
        require_id(scope_id, "scope_id")
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM operations WHERE scope_id = ?"
                    " ORDER BY committed_event",
                    (scope_id,),
                )
            )


# ----------------------------------------------------------------------
# dependency graph (SPEC_V2 §06, §38)
# ----------------------------------------------------------------------


class DependencyRepo:
    """derived → exact-input edges for staleness fencing."""

    _INVALIDATIONS = ("revalidate", "suppress", "rebuild")

    def __init__(self, store: Store) -> None:
        self._store = store

    def add(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        derived_kind: str,
        derived_id: str,
        input_kind: str,
        input_id: str,
        *,
        derived_revision: int = 1,
        input_revision: int = 1,
        invalidation: str = "revalidate",
    ) -> None:
        require_id(scope_id, "scope_id")
        _require_str(derived_kind, "derived_kind")
        require_id(derived_id, "derived_id")
        _require_str(input_kind, "input_kind")
        require_id(input_id, "input_id")
        if invalidation not in self._INVALIDATIONS:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"invalid invalidation {invalidation!r}"
            )
        conn.execute(
            "INSERT OR REPLACE INTO dependency_refs"
            "(derived_kind, derived_id, derived_revision, input_kind, input_id,"
            " input_revision, scope_id, invalidation)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                derived_kind,
                derived_id,
                derived_revision,
                input_kind,
                input_id,
                input_revision,
                scope_id,
                invalidation,
            ),
        )

    def dependents_of(
        self, input_kind: str, input_id: str, input_revision: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Everything derived from an input — walked on edits/purges."""
        with self._store.read() as conn:
            if input_revision is not None:
                return _rows(
                    conn.execute(
                        "SELECT * FROM dependency_refs"
                        " WHERE input_kind = ? AND input_id = ? AND input_revision = ?",
                        (input_kind, input_id, input_revision),
                    )
                )
            return _rows(
                conn.execute(
                    "SELECT * FROM dependency_refs"
                    " WHERE input_kind = ? AND input_id = ?",
                    (input_kind, input_id),
                )
            )

    def inputs_of(self, derived_kind: str, derived_id: str) -> list[dict[str, Any]]:
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM dependency_refs"
                    " WHERE derived_kind = ? AND derived_id = ?",
                    (derived_kind, derived_id),
                )
            )


# ----------------------------------------------------------------------
# projection tracking (SPEC_V2 §28, §38)
# ----------------------------------------------------------------------


class ProjectionRepo:
    """Outbox, builds, and the active-projection pointer (CAS publish)."""

    _BUILD_STATES = ("building", "validating", "ready", "published", "failed", "retired")

    def __init__(self, store: Store) -> None:
        self._store = store

    def outbox_add(
        self,
        conn: sqlite3.Connection,
        scope_partition: str,
        object_kind: str,
        object_id: str,
        revision: int,
        *,
        effect_digest: Optional[bytes] = None,
    ) -> None:
        """Queue a projection update inside the effect's own transaction."""
        _require_str(scope_partition, "scope_partition")
        _require_str(object_kind, "object_kind")
        require_id(object_id, "object_id")
        _require_int(revision, "revision", minimum=1)
        conn.execute(
            "INSERT OR REPLACE INTO projection_outbox"
            "(event_seq, scope_partition, object_kind, object_id, revision,"
            " effect_digest, consumed_by)"
            " VALUES (?, ?, ?, ?, ?, ?, NULL)",
            (
                _next_event_seq(conn),
                scope_partition,
                object_kind,
                object_id,
                revision,
                effect_digest,
            ),
        )

    def outbox_pending(
        self, scope_partition: str, *, limit: int = 256
    ) -> list[dict[str, Any]]:
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM projection_outbox"
                    " WHERE scope_partition = ? AND consumed_by IS NULL"
                    " ORDER BY event_seq LIMIT ?",
                    (scope_partition, limit),
                )
            )

    def outbox_consume(
        self,
        conn: sqlite3.Connection,
        scope_partition: str,
        build_id: str,
        through_seq: int,
    ) -> int:
        """Mark pending rows consumed by a build; returns count."""
        require_id(build_id, "build_id")
        cur = conn.execute(
            "UPDATE projection_outbox SET consumed_by = ?"
            " WHERE scope_partition = ? AND consumed_by IS NULL AND event_seq <= ?",
            (build_id, scope_partition, through_seq),
        )
        return cur.rowcount

    def build_create(
        self,
        conn: sqlite3.Connection,
        kind: str,
        scope_partition: str,
        *,
        manifest: Any = None,
        snapshot_seq: int = 0,
        build_id: Optional[str] = None,
    ) -> str:
        _require_str(kind, "kind")
        _require_str(scope_partition, "scope_partition")
        bid = build_id or new_id()
        require_id(bid, "build_id")
        from ..core.time import wall_us

        conn.execute(
            "INSERT INTO projection_builds"
            "(build_id, kind, scope_partition, manifest_json, snapshot_seq,"
            " status, created_us)"
            " VALUES (?, ?, ?, ?, ?, 'building', ?)",
            (bid, kind, scope_partition, _json_text(manifest, "manifest") or "{}",
             snapshot_seq, wall_us()),
        )
        return bid

    def build_set_state(
        self,
        conn: sqlite3.Connection,
        build_id: str,
        status: str,
        *,
        caught_up_seq: Optional[int] = None,
        validation_digest: Optional[bytes] = None,
    ) -> None:
        require_id(build_id, "build_id")
        if status not in self._BUILD_STATES:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid build status {status!r}")
        conn.execute(
            "UPDATE projection_builds"
            " SET status = ?,"
            "     caught_up_seq = COALESCE(?, caught_up_seq),"
            "     validation_digest = COALESCE(?, validation_digest)"
            " WHERE build_id = ?",
            (status, caught_up_seq, validation_digest, build_id),
        )

    def build_get(self, build_id: str) -> Optional[dict[str, Any]]:
        require_id(build_id, "build_id")
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM projection_builds WHERE build_id = ?", (build_id,)
                )
            )

    def active_get(self, scope_partition: str, projection_kind: str) -> Optional[dict[str, Any]]:
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM active_projections"
                    " WHERE scope_partition = ? AND projection_kind = ?",
                    (scope_partition, projection_kind),
                )
            )

    def active_publish(
        self,
        conn: sqlite3.Connection,
        scope_partition: str,
        projection_kind: str,
        build_id: str,
        indexed_through_seq: int,
        *,
        expected_cas: int,
    ) -> None:
        """Compare-and-swap publish; expected_cas fences stale publishers.

        ``expected_cas=-1`` is the first-publish sentinel: the row must be
        absent. Otherwise the row must exist at exactly that cas_revision.
        """
        _require_str(scope_partition, "scope_partition")
        _require_str(projection_kind, "projection_kind")
        require_id(build_id, "build_id")
        _require_int(expected_cas, "expected_cas", minimum=-1)
        cur = conn.execute(
            "UPDATE active_projections"
            " SET build_id = ?, indexed_through_seq = ?, cas_revision = cas_revision + 1"
            " WHERE scope_partition = ? AND projection_kind = ? AND cas_revision = ?",
            (build_id, indexed_through_seq, scope_partition, projection_kind, expected_cas),
        )
        if cur.rowcount == 0:
            # Insert path only valid when no row exists (cas_revision starts 0).
            existing = conn.execute(
                "SELECT cas_revision FROM active_projections"
                " WHERE scope_partition = ? AND projection_kind = ?",
                (scope_partition, projection_kind),
            ).fetchone()
            if existing is not None:
                raise VerbatimError(
                    ErrorCode.STALE_PROPOSAL, "active projection CAS failed — stale publisher"
                )
            if expected_cas != -1:
                raise VerbatimError(
                    ErrorCode.STALE_PROPOSAL, "expected active projection row missing"
                )
            conn.execute(
                "INSERT INTO active_projections"
                "(scope_partition, projection_kind, build_id, indexed_through_seq,"
                " cas_revision)"
                " VALUES (?, ?, ?, ?, 0)",
                (scope_partition, projection_kind, build_id, indexed_through_seq),
            )


# ----------------------------------------------------------------------
# embedding inputs + policy artifacts (SPEC_V2 §27, §38)
# ----------------------------------------------------------------------


class EmbeddingInputsRepo:
    """Exact inputs behind each embedding row — replay/audit support."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def record(
        self,
        conn: sqlite3.Connection,
        span_id: str,
        encoder_id: str,
        preprocessing_version: str,
        dependency_digest: bytes,
        input_known_seq: int,
        *,
        input_id: Optional[str] = None,
    ) -> str:
        require_id(span_id, "span_id")
        _require_str(encoder_id, "encoder_id")
        _require_str(preprocessing_version, "preprocessing_version")
        iid = input_id or new_id()
        conn.execute(
            "INSERT INTO embedding_inputs"
            "(input_id, span_id, encoder_id, preprocessing_version,"
            " dependency_digest, input_known_seq)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(span_id, encoder_id, preprocessing_version)"
            " DO UPDATE SET dependency_digest = excluded.dependency_digest,"
            "              input_known_seq = excluded.input_known_seq",
            (iid, span_id, encoder_id, preprocessing_version,
             bytes(dependency_digest), input_known_seq),
        )
        return iid

    def for_span(self, span_id: str) -> list[dict[str, Any]]:
        require_id(span_id, "span_id")
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM embedding_inputs WHERE span_id = ?", (span_id,)
                )
            )


class PolicyArtifactsRepo:
    """Model/policy artifact registry with validation state."""

    _STATES = ("unvalidated", "validated", "revoked")

    def __init__(self, store: Store) -> None:
        self._store = store

    def register(
        self,
        conn: sqlite3.Connection,
        kind: str,
        digest: bytes,
        declared: Any,
        *,
        license_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> str:
        _require_str(kind, "kind")
        aid = artifact_id or new_id()
        require_id(aid, "artifact_id")
        from ..core.time import wall_us

        conn.execute(
            "INSERT INTO policy_artifacts"
            "(artifact_id, kind, digest, declared_json, license_id,"
            " validation_state, created_us)"
            " VALUES (?, ?, ?, ?, ?, 'unvalidated', ?)",
            (aid, kind, bytes(digest), _json_text(declared, "declared") or "{}",
             license_id, wall_us()),
        )
        return aid

    def get(self, artifact_id: str) -> Optional[dict[str, Any]]:
        require_id(artifact_id, "artifact_id")
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM policy_artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                )
            )

    def set_state(
        self, conn: sqlite3.Connection, artifact_id: str, state: str
    ) -> None:
        require_id(artifact_id, "artifact_id")
        if state not in self._STATES:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid state {state!r}")
        conn.execute(
            "UPDATE policy_artifacts SET validation_state = ? WHERE artifact_id = ?",
            (state, artifact_id),
        )

    def by_kind(self, kind: str, *, validated_only: bool = False) -> list[dict[str, Any]]:
        _require_str(kind, "kind")
        with self._store.read() as conn:
            if validated_only:
                return _rows(
                    conn.execute(
                        "SELECT * FROM policy_artifacts"
                        " WHERE kind = ? AND validation_state = 'validated'",
                        (kind,),
                    )
                )
            return _rows(
                conn.execute(
                    "SELECT * FROM policy_artifacts WHERE kind = ?", (kind,)
                )
            )


# ----------------------------------------------------------------------
# disclosures (SPEC_V2 §36, §38)
# ----------------------------------------------------------------------


class DisclosuresRepo:
    """Content-minimized egress receipts — never payload copies."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def record(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        processor: str,
        purpose: str,
        *,
        input_refs: Any = None,
        usage: Any = None,
        outcome: str = "unknown",
    ) -> str:
        require_id(scope_id, "scope_id")
        _require_str(processor, "processor")
        _require_str(purpose, "purpose")
        did = new_id()
        from ..core.time import wall_us

        conn.execute(
            "INSERT INTO disclosures"
            "(disclosure_id, scope_id, processor, purpose, input_refs_json,"
            " usage_json, outcome, created_us)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                did,
                scope_id,
                processor,
                purpose,
                _json_text(input_refs, "input_refs") or "[]",
                _json_text(usage, "usage") or "{}",
                outcome,
                wall_us(),
            ),
        )
        return did

    def for_scope(
        self, scope_id: str, *, processor: Optional[str] = None
    ) -> list[dict[str, Any]]:
        require_id(scope_id, "scope_id")
        with self._store.read() as conn:
            if processor:
                return _rows(
                    conn.execute(
                        "SELECT * FROM disclosures WHERE scope_id = ? AND processor = ?"
                        " ORDER BY created_us",
                        (scope_id, processor),
                    )
                )
            return _rows(
                conn.execute(
                    "SELECT * FROM disclosures WHERE scope_id = ? ORDER BY created_us",
                    (scope_id,),
                )
            )


# ----------------------------------------------------------------------
# erasure ledger (SPEC_V2 §38, §41)
# ----------------------------------------------------------------------


class ErasureRepo:
    """Opaque deletion knowledge — object digests, never live references.

    Restoring an old backup must not resurrect erased content: the digest
    of a purged object is checked on restore/import (V2-41). Digests are
    profile-keyed HMACs so the ledger cannot confirm low-entropy content
    by dictionary attack.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def digest_for(self, object_kind: str, object_id: str) -> bytes:
        """Profile-keyed pseudonymous object identity."""
        return self._store.hmac(f"{object_kind}:{object_id}".encode("utf-8"))

    def record(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        object_kind: str,
        object_id: str,
        *,
        purge_id: Optional[str] = None,
        erasure_epoch: int = 0,
    ) -> str:
        require_id(scope_id, "scope_id")
        _require_str(object_kind, "object_kind")
        require_id(object_id, "object_id")
        eid = new_id()
        conn.execute(
            "INSERT INTO erasure_ledger"
            "(erasure_id, scope_id, object_kind, object_digest, purge_id,"
            " erased_event, erasure_epoch)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                eid,
                scope_id,
                object_kind,
                self.digest_for(object_kind, object_id),
                purge_id,
                _next_event_seq(conn),
                erasure_epoch,
            ),
        )
        return eid

    def is_erased(
        self, conn: sqlite3.Connection, scope_id: str, object_kind: str, object_id: str
    ) -> bool:
        """Check inside the caller's transaction — restore paths rely on it."""
        require_id(scope_id, "scope_id")
        row = conn.execute(
            "SELECT 1 FROM erasure_ledger"
            " WHERE scope_id = ? AND object_kind = ? AND object_digest = ?"
            " LIMIT 1",
            (scope_id, object_kind, self.digest_for(object_kind, object_id)),
        ).fetchone()
        return row is not None

    def for_scope(self, scope_id: str) -> list[dict[str, Any]]:
        require_id(scope_id, "scope_id")
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM erasure_ledger WHERE scope_id = ?"
                    " ORDER BY erased_event",
                    (scope_id,),
                )
            )


# ----------------------------------------------------------------------
# handoff capsules (SPEC_V2 §10, §38)
# ----------------------------------------------------------------------


class HandoffRepo:
    """Scoped cross-agent handoff capsules; reauthorized at consumption."""

    _STATUSES = ("open", "consumed", "expired", "revoked")
    _PERMISSIONS = ("read_evidence", "propose", "share")

    def __init__(self, store: Store) -> None:
        self._store = store

    def create(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        recipient_id: str,
        issuer_id: str,
        *,
        permission: str = "read_evidence",
        expires_us: Optional[int] = None,
        portable: bool = False,
        snapshot: Any = None,
        capsule_id: Optional[str] = None,
    ) -> str:
        require_id(scope_id, "scope_id")
        require_id(recipient_id, "recipient_id")
        require_id(issuer_id, "issuer_id")
        if permission not in self._PERMISSIONS:
            raise VerbatimError(
                ErrorCode.VALIDATION, f"invalid permission {permission!r}"
            )
        cid = capsule_id or new_id()
        require_id(cid, "capsule_id")
        conn.execute(
            "INSERT INTO handoff_capsules"
            "(capsule_id, scope_id, recipient_id, issuer_id, permission,"
            " expires_us, portable, status, snapshot_json, created_event)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)",
            (
                cid,
                scope_id,
                recipient_id,
                issuer_id,
                permission,
                expires_us,
                1 if portable else 0,
                _json_text(snapshot, "snapshot") or "{}",
                _next_event_seq(conn),
            ),
        )
        return cid

    def get(self, capsule_id: str) -> Optional[dict[str, Any]]:
        require_id(capsule_id, "capsule_id")
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM handoff_capsules WHERE capsule_id = ?",
                    (capsule_id,),
                )
            )

    def add_member(
        self,
        conn: sqlite3.Connection,
        capsule_id: str,
        object_kind: str,
        object_id: str,
        *,
        revision: int = 1,
    ) -> None:
        require_id(capsule_id, "capsule_id")
        _require_str(object_kind, "object_kind")
        require_id(object_id, "object_id")
        conn.execute(
            "INSERT OR REPLACE INTO capsule_members"
            "(capsule_id, object_kind, object_id, revision) VALUES (?, ?, ?, ?)",
            (capsule_id, object_kind, object_id, revision),
        )

    def members(self, capsule_id: str) -> list[dict[str, Any]]:
        require_id(capsule_id, "capsule_id")
        with self._store.read() as conn:
            return _rows(
                conn.execute(
                    "SELECT * FROM capsule_members WHERE capsule_id = ?",
                    (capsule_id,),
                )
            )

    def set_status(
        self,
        conn: sqlite3.Connection,
        capsule_id: str,
        status: str,
        *,
        consumed_event: Optional[int] = None,
    ) -> None:
        require_id(capsule_id, "capsule_id")
        if status not in self._STATUSES:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid status {status!r}")
        conn.execute(
            "UPDATE handoff_capsules SET status = ?,"
            " consumed_event = COALESCE(?, consumed_event)"
            " WHERE capsule_id = ?",
            (status, consumed_event, capsule_id),
        )


# ----------------------------------------------------------------------
# ingest batches + connector cursors (SPEC_V2 §11, §38, §48)
# ----------------------------------------------------------------------


class IngestBatchesRepo:
    """Batch manifests and per-connector durable cursors."""

    _STATES = ("open", "partial", "complete", "aborted")

    def __init__(self, store: Store) -> None:
        self._store = store

    def create_batch(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        *,
        manifest: Any = None,
        batch_id: Optional[str] = None,
    ) -> str:
        require_id(scope_id, "scope_id")
        bid = batch_id or new_id()
        require_id(bid, "batch_id")
        from ..core.time import wall_us

        conn.execute(
            "INSERT INTO ingest_batches"
            "(batch_id, scope_id, manifest_json, state, created_us)"
            " VALUES (?, ?, ?, 'open', ?)",
            (bid, scope_id, _json_text(manifest, "manifest") or "{}", wall_us()),
        )
        return bid

    def update_batch(
        self,
        conn: sqlite3.Connection,
        batch_id: str,
        *,
        received: Optional[int] = None,
        accepted: Optional[int] = None,
        state: Optional[str] = None,
    ) -> None:
        require_id(batch_id, "batch_id")
        if state is not None and state not in self._STATES:
            raise VerbatimError(ErrorCode.VALIDATION, f"invalid batch state {state!r}")
        conn.execute(
            "UPDATE ingest_batches SET"
            " received_count = COALESCE(?, received_count),"
            " accepted_count = COALESCE(?, accepted_count),"
            " state = COALESCE(?, state)"
            " WHERE batch_id = ?",
            (received, accepted, state, batch_id),
        )

    def get_batch(self, batch_id: str) -> Optional[dict[str, Any]]:
        require_id(batch_id, "batch_id")
        with self._store.read() as conn:
            return _row(
                conn.execute(
                    "SELECT * FROM ingest_batches WHERE batch_id = ?", (batch_id,)
                )
            )

    def cursor_get(self, conn: sqlite3.Connection, connector_id: str, scope_id: str) -> Optional[str]:
        _require_str(connector_id, "connector_id")
        require_id(scope_id, "scope_id")
        row = conn.execute(
            "SELECT cursor_value FROM connector_cursors"
            " WHERE connector_id = ? AND scope_id = ?",
            (connector_id, scope_id),
        ).fetchone()
        return row[0] if row else None

    def cursor_set(
        self, conn: sqlite3.Connection, connector_id: str, scope_id: str, cursor: str
    ) -> None:
        _require_str(connector_id, "connector_id")
        require_id(scope_id, "scope_id")
        _require_str(cursor, "cursor")
        from ..core.time import wall_us

        conn.execute(
            "INSERT INTO connector_cursors(connector_id, scope_id, cursor_value, updated_us)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(connector_id, scope_id) DO UPDATE SET"
            " cursor_value = excluded.cursor_value, updated_us = excluded.updated_us",
            (connector_id, scope_id, cursor, wall_us()),
        )


# ----------------------------------------------------------------------
# usage aggregates (SPEC_V2 §49, §38)
# ----------------------------------------------------------------------


class UsageRepo:
    """Bounded usage counts — ranking signals, never authority."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def bump(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        object_kind: str,
        object_id: str,
        kind: str,
        *,
        window_start_us: int = 0,
        delta: int = 1,
    ) -> None:
        require_id(scope_id, "scope_id")
        _require_str(object_kind, "object_kind")
        require_id(object_id, "object_id")
        _require_str(kind, "kind")
        conn.execute(
            "INSERT INTO usage_aggregates"
            "(scope_id, object_kind, object_id, kind, count, window_start_us)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(scope_id, object_kind, object_id, kind, window_start_us)"
            " DO UPDATE SET count = count + excluded.count",
            (scope_id, object_kind, object_id, kind, delta, window_start_us),
        )

    def counts(self, scope_id: str, object_kind: str, object_id: str) -> dict[str, int]:
        require_id(scope_id, "scope_id")
        with self._store.read() as conn:
            rows = conn.execute(
                "SELECT kind, SUM(count) FROM usage_aggregates"
                " WHERE scope_id = ? AND object_kind = ? AND object_id = ?"
                " GROUP BY kind",
                (scope_id, object_kind, object_id),
            ).fetchall()
        return {r[0]: int(r[1]) for r in rows}


class GrantsRepo:
    """Recorded principal permissions + the scope authorization epoch.

    ``scope_grants`` is the durable grant surface a bound caller's
    authority is recorded against (SPEC_V2 §09): revocation sets
    ``revoked_event`` and bumps ``scopes.authz_revision``, so a caller
    pinned to an older epoch — or holding a since-revoked permission —
    is fenced on its next operation (V2-09.15). ``INSERT OR IGNORE``
    semantics mean recording can never resurrect a revoked grant; only
    an explicit ``grant()`` clears the revocation.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def authz_revision(self, conn: sqlite3.Connection, scope_id: str) -> int:
        row = conn.execute(
            "SELECT authz_revision FROM scopes WHERE scope_id = ?",
            (scope_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def bump_authz(self, conn: sqlite3.Connection, scope_id: str) -> int:
        """Advance the scope's authorization epoch; returns the new value."""
        conn.execute(
            "UPDATE scopes SET authz_revision = authz_revision + 1"
            " WHERE scope_id = ?",
            (scope_id,),
        )
        return self.authz_revision(conn, scope_id)

    def record_bound(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        principal_id: str,
        permissions: Sequence[str],
        issuer: str,
        granted_event: int,
    ) -> int:
        """Record host-bound grants — idempotent, never un-revokes."""
        n = 0
        for perm in permissions:
            cur = conn.execute(
                "INSERT OR IGNORE INTO scope_grants"
                " (scope_id, principal_id, permission, issuer, granted_event)"
                " VALUES (?, ?, ?, ?, ?)",
                (scope_id, principal_id, perm, issuer, granted_event),
            )
            n += cur.rowcount
        return n

    def revoked(
        self, conn: sqlite3.Connection, scope_id: str, principal_id: str
    ) -> set[str]:
        rows = conn.execute(
            "SELECT permission FROM scope_grants"
            " WHERE scope_id = ? AND principal_id = ?"
            " AND revoked_event IS NOT NULL",
            (scope_id, principal_id),
        ).fetchall()
        return {r[0] for r in rows}

    def active(
        self, conn: sqlite3.Connection, scope_id: str, principal_id: str
    ) -> set[str]:
        rows = conn.execute(
            "SELECT permission FROM scope_grants"
            " WHERE scope_id = ? AND principal_id = ?"
            " AND revoked_event IS NULL",
            (scope_id, principal_id),
        ).fetchall()
        return {r[0] for r in rows}

    def grant(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        principal_id: str,
        permission: str,
        issuer: str,
        granted_event: int,
    ) -> None:
        """Explicit grant — the only path that clears a revocation."""
        conn.execute(
            "INSERT INTO scope_grants"
            " (scope_id, principal_id, permission, issuer, granted_event,"
            "  revoked_event)"
            " VALUES (?, ?, ?, ?, ?, NULL)"
            " ON CONFLICT(scope_id, principal_id, permission) DO UPDATE SET"
            "   issuer = excluded.issuer,"
            "   granted_event = excluded.granted_event,"
            "   revoked_event = NULL",
            (scope_id, principal_id, permission, issuer, granted_event),
        )

    def revoke(
        self,
        conn: sqlite3.Connection,
        scope_id: str,
        principal_id: str,
        permission: str,
        revoked_event: int,
    ) -> bool:
        """Revoke a grant; returns False if no active grant existed."""
        cur = conn.execute(
            "UPDATE scope_grants SET revoked_event = ?"
            " WHERE scope_id = ? AND principal_id = ? AND permission = ?"
            " AND revoked_event IS NULL",
            (revoked_event, scope_id, principal_id, permission),
        )
        return cur.rowcount > 0


__all__ = [
    "ArtifactsRepo",
    "ContextGroupsRepo",
    "DependencyRepo",
    "DisclosuresRepo",
    "EmbeddingInputsRepo",
    "EpisodesRepo",
    "ErasureRepo",
    "EvidenceFamiliesRepo",
    "GrantsRepo",
    "HandoffRepo",
    "IngestBatchesRepo",
    "OperationsRepo",
    "PolicyArtifactsRepo",
    "PredicateRegistryRepo",
    "ProceduresRepo",
    "ProjectionRepo",
    "ProspectiveRepo",
    "SourceViewsRepo",
    "UsageRepo",
]
