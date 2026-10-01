"""Persistent at-least-once job queue (SPEC §22).

Delivery guarantee: a job is never lost between ``enqueue`` and a terminal
state, but it may be *leased more than once* — lease expiry returns the job to
``retry_wait`` and a replacement worker picks it up. Correctness therefore
rests on two mechanisms rather than on timing:

* generation fencing — every lease hand-off increments ``generation``; a
  paused or expired worker holding an older generation can no longer commit,
  so stale results can never overwrite a replacement worker's output
  (SPEC §22 "a paused old worker cannot later overwrite the replacement
  worker's result").
* idempotent effects — callers deduplicate via ``dedup_key`` (UNIQUE) and
  expected versions in their own domain tables; the queue itself only
  guarantees redelivery.

Time: ``lease_until_us`` and ``not_before_us`` are persisted wall-clock
microseconds. Workers are expected to enforce their own deadlines on a
monotonic clock; the persisted expiry exists so that *other* workers can
reclaim abandoned work, and it tolerates restarts because it is durable.

Ordering: ``purge`` jobs always lease ahead of enrichment kinds. Purge work
is a privacy obligation, not throughput work — a saturated enrichment queue
must never starve deletion (SPEC §22 reserved maintenance lane). Within a
kind, FIFO by insertion rowid is preferred; execution order is never used
to infer validity.
"""

from __future__ import annotations

import random
import sqlite3
from typing import Any, Iterable, Optional, Union

from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    JobKind,
    JobState,
    Scope,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)

# Backoff policy (SPEC §22 "bounded exponential backoff with jitter").
RETRY_BASE_S = 2.0
RETRY_CAP_S = 300.0
DEFAULT_LEASE_S = 60.0
# Reserved control lane (SPEC_V2 §39.11): suppression, rebuild, and
# review-application work is privacy/correctness work, not throughput work —
# it drains ahead of enrichment and never counts against the ordinary cap.
# V3 (§40) splits that reserved lane into four: privacy_control (deletion,
# revocation, quarantine, vault), maintenance (rebuilds, projections,
# connectors), background (learning: compile/consolidate/episode/index
# builds), and ordinary (capture-path work). 'control' stays valid for rows
# enqueued by v2 and sorts with privacy_control.
LANES = ("ordinary", "background", "privacy_control", "maintenance", "control")
CONTROL_KINDS = frozenset(
    {JobKind.PURGE.value, JobKind.REINDEX.value, JobKind.REVIEW_APPLY.value}
)
# Default lane for v3 kinds; anything unlisted falls back to 'ordinary'.
_LANE_DEFAULTS = {
    # privacy/control — deletion, revocation, and admission screening are
    # correctness work: they must never starve behind enrichment (§40.02,
    # §34.01). Screening ahead of harvest keeps flagged content from
    # becoming durable claims before its label exists.
    JobKind.PURGE_DERIVED.value: "privacy_control",
    JobKind.PURGE_VAULT.value: "privacy_control",
    JobKind.QUARANTINE_REVIEW.value: "privacy_control",
    JobKind.REVOCATION_NOTIFY.value: "privacy_control",
    JobKind.VAULT_ROTATE.value: "privacy_control",
    JobKind.SCREEN.value: "privacy_control",
    # maintenance — rebuildable projection/index work
    JobKind.SPARSE_INDEX.value: "maintenance",
    JobKind.LATE_INDEX.value: "maintenance",
    JobKind.SIGNATURE_INDEX.value: "maintenance",
    JobKind.PROJECTION_SYNC.value: "maintenance",
    JobKind.CONNECTOR_PULL.value: "maintenance",
    # background — learning pipeline
    JobKind.EPISODE_BUILD.value: "background",
    JobKind.TRANSITION_BUILD.value: "background",
    JobKind.PROCEDURE_COMPILE.value: "background",
    JobKind.PROCEDURE_REFINE.value: "background",
    JobKind.CONSOLIDATE.value: "background",
}
# Dequeue priority: privacy/correctness work first, then maintenance, then
# learning, then ordinary throughput.
_LANE_ORDER_SQL = (
    "CASE lane WHEN 'control' THEN 0 WHEN 'privacy_control' THEN 1 "
    "WHEN 'maintenance' THEN 2 WHEN 'background' THEN 3 ELSE 4 END"
)
# Within a lane, ``source_project`` sorts ahead of sibling work: it is the
# only job that settles ``source_lexical_ready`` — the capability every
# session-consistency barrier waits on — while ``source_embed``/claim-
# pipeline jobs settle capabilities no read path blocks on. Dequeue order
# is a scheduling decision, not a semantics change: the same jobs run, the
# same lease/generation fencing applies, and barrier-visible readiness
# simply resolves earlier.
_KIND_ORDER_SQL = "(kind = 'source_project') DESC"
# job_events is an operational log, not an audit trail: keep it bounded so a
# crash-looping job cannot grow the table without limit (SPEC §20 "bounded
# operational history without payload duplication").
JOB_HISTORY_LIMIT = 64
# Private key stashed inside input_refs_json. The schema has no created_us
# column, yet stats() must report queue age; smuggling the enqueue timestamp
# inside the caller's own refs keeps the table contract intact. Keys with a
# leading underscore are stripped again on read.
_ENQUEUED_US_KEY = "_enqueued_us"

_TERMINAL = (JobState.SUCCEEDED.value, JobState.FAILED.value, JobState.CANCELLED.value)


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    """Rows → dict snapshots via cursor description.

    The store does not promise a ``row_factory`` — connections may yield
    plain tuples — so column access always goes through the cursor's own
    description (same convention as storage.repos).
    """
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    rows = _rows(cur)
    return rows[0] if rows else None


def scope_key(scope: Union[Scope, str]) -> str:
    """Canonical ``scope_id`` text for a :class:`Scope`; strings pass through.

    Delegates to ``core.identity.scope_key`` — the same canonicalization
    ingest and the CLI use when writing jobs and granting consents — so all
    scope-keyed tables share one id namespace. Raw strings are validated as
    identifiers and passed through unchanged.
    """
    if isinstance(scope, Scope):
        from ..core.identity import scope_key as _canonical

        return _canonical(scope)
    return require_id(scope, "scope_id")


def _kind_value(kind: Union[JobKind, str]) -> str:
    """Normalize a job kind; unknown kinds are rejected at the door.

    The schema CHECK also enforces this, but failing here produces the
    correct error class (VALIDATION, not a raw IntegrityError).
    """
    try:
        return JobKind(kind).value if not isinstance(kind, JobKind) else kind.value
    except ValueError as exc:
        raise VerbatimError(ErrorCode.VALIDATION, f"unknown job kind {kind!r}") from exc


def _row_to_job(row: dict[str, Any]) -> dict[str, Any]:
    job = dict(row)
    refs = safe_json_loads(job.get("input_refs_json") or "{}")
    if isinstance(refs, dict):
        job["input_refs"] = {k: v for k, v in refs.items() if not k.startswith("_")}
    else:
        job["input_refs"] = {}
    job.pop("input_refs_json", None)
    return job


class JobQueue:
    """Durable FIFO-ish queue over the ``jobs``/``job_events`` tables.

    ``store`` must expose ``tx()`` / ``read()`` context managers yielding an
    ``sqlite3.Connection`` and is used only for transaction scoping — all
    statements stay parameterized (SPEC §41). ``max_pending`` bounds ordinary
    pending jobs; purge work is exempt because deletion must not be starved
    by enrichment backpressure.
    """

    def __init__(
        self,
        store: Any,
        max_pending: int = 10_000,
        *,
        cap: Optional[int] = None,
        clock: Optional[Any] = None,
        rng: Optional[random.Random] = None,
        policy_epoch: int = 0,
        history_limit: int = JOB_HISTORY_LIMIT,
    ) -> None:
        # ``max_pending`` mirrors JobsConfig.max_pending; ``cap`` is a
        # tolerated alias for callers wired before the config names settled.
        limit = cap if cap is not None else max_pending
        if limit < 1:
            raise VerbatimError(ErrorCode.CONFIG_INVALID, "queue cap must be positive")
        self._store = store
        self._cap = limit
        # Schema-v2 columns are optional: a pre-migration store (or the v1
        # test shim) has no ``lane``/``operation_key`` columns. Probing once
        # keeps every SQL path working on both shapes — on v1 the lane
        # semantics degrade to the kind set (control kinds still sort first)
        # and ``operation_key`` enqueue is refused explicitly instead of
        # silently dropping a durability key.
        with self._store.read() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
        self._has_lane = "lane" in cols
        self._has_opkey = "operation_key" in cols
        # Injectable clock/RNG keep tests deterministic; production defaults
        # to wall clock and a system RNG for jitter.
        self._now = clock or now_us
        self._rand = rng or random.SystemRandom()
        self._policy_epoch = policy_epoch
        self._history_limit = history_limit

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _append_event(self, conn: sqlite3.Connection, job_id: str, state: str, error_code: Optional[str]) -> None:
        nxt = conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM job_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO job_events (job_id, event_seq, state, error_code) VALUES (?,?,?,?)",
            (job_id, nxt, state, error_code),
        )
        conn.execute(
            "DELETE FROM job_events WHERE job_id = ? AND event_seq <= "
            "(SELECT MAX(event_seq) FROM job_events WHERE job_id = ?) - ?",
            (job_id, job_id, self._history_limit),
        )

    def _pending_count(self, conn: sqlite3.Connection, scope_id: str) -> int:
        # Control work neither triggers nor consumes the ordinary-job cap —
        # the reserved lane runs both ways (SPEC §22, V2-39.11). On a v1
        # schema (no lane column) the control kinds are exempt by name.
        if self._has_lane:
            return conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE scope_id = ? AND lane = 'ordinary'"
                " AND state IN ('queued','retry_wait')",
                (scope_id,),
            ).fetchone()[0]
        return conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE scope_id = ?"
            " AND kind NOT IN ('purge','reindex','review_apply')"
            " AND state IN ('queued','retry_wait')",
            (scope_id,),
        ).fetchone()[0]

    def _resolve_lane(self, kind_value: str, lane: Optional[str]) -> str:
        """Assign the job's lane; control kinds default to ``'control'``.

        v3 kinds resolve through ``_LANE_DEFAULTS`` (§40); anything else is
        ordinary throughput work.
        """
        resolved = lane if lane is not None else _LANE_DEFAULTS.get(
            kind_value, "control" if kind_value in CONTROL_KINDS else "ordinary"
        )
        if resolved not in LANES:
            raise VerbatimError(ErrorCode.VALIDATION, f"unknown job lane {resolved!r}")
        return resolved

    @staticmethod
    def _is_current(row: Optional[dict[str, Any]], owner: str, generation: int) -> bool:
        return (
            row is not None
            and row["state"] == JobState.LEASED.value
            and row["lease_owner"] == owner
            and row["generation"] == generation
        )

    def _fetch(self, conn: sqlite3.Connection, job_id: str) -> Optional[dict[str, Any]]:
        return _row(conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)))

    # ------------------------------------------------------------------
    # producer API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        conn: sqlite3.Connection,
        scope_id: Union[Scope, str],
        kind: Union[JobKind, str],
        input_refs: dict[str, Any],
        dedup_key: Optional[bytes] = None,
        deadline_s: Optional[float] = None,
        not_before_us: int = 0,
        policy_epoch: Optional[int] = None,
        operation_key: Optional[str] = None,
        lane: Optional[str] = None,
    ) -> str:
        """Insert a durable job; returns the job_id.

        ``dedup_key`` makes enqueue idempotent: on a UNIQUE conflict the
        *existing* job_id is returned, so retried producers cannot create
        duplicate effects downstream. Backpressure is reported *before*
        unbounded growth (SPEC §22, §43 BACKPRESSURE — retryable because the
        caller may succeed later). Control-lane jobs bypass the cap on
        purpose: privacy and correctness work must never be starved by
        enrichment backpressure (SPEC_V2 §39.11/39.12).

        ``operation_key`` (schema v2) binds the job to the ``operations``
        receipt ledger so a redelivered job replays its prior receipt
        instead of reapplying effects (V2-39.10). ``lane`` overrides the
        default lane assignment; the reserved control kinds default to
        ``'control'``. Both are refused explicitly — never silently
        dropped — on a store whose jobs table predates the columns.
        """
        sid = scope_key(scope_id)
        k = _kind_value(kind)
        if not isinstance(input_refs, dict):
            raise VerbatimError(ErrorCode.VALIDATION, "input_refs must be a mapping")
        if dedup_key is not None and not isinstance(dedup_key, (bytes, bytearray)):
            raise VerbatimError(ErrorCode.VALIDATION, "dedup_key must be bytes or None")
        if not_before_us < 0:
            raise VerbatimError(ErrorCode.VALIDATION, "not_before_us must be >= 0")
        resolved_lane = self._resolve_lane(k, lane)
        if lane is not None and not self._has_lane:
            raise VerbatimError(
                ErrorCode.SCHEMA_UNSUPPORTED, "job lane requires schema v2"
            )
        if operation_key is not None:
            require_id(operation_key, "operation_key")
            if not self._has_opkey:
                raise VerbatimError(
                    ErrorCode.SCHEMA_UNSUPPORTED,
                    "job operation_key requires schema v2",
                )

        if dedup_key is not None:
            # V4-42.06: a deduplicated retry resolves the EXISTING
            # obligation — it is not new work, so new-item backpressure
            # must never make the lookup non-idempotent. Same terminal-set
            # semantics as the IntegrityError path below: a live or
            # succeeded row IS the receipt; a dead one falls through to
            # the vacate-and-reinsert path so the key is re-drivable.
            existing = _row(
                conn.execute(
                    "SELECT job_id, state FROM jobs WHERE dedup_key = ?",
                    (bytes(dedup_key),),
                )
            )
            if existing is not None and existing["state"] not in (
                JobState.FAILED.value,
                JobState.CANCELLED.value,
            ):
                return existing["job_id"]

        if resolved_lane == "ordinary" and self._pending_count(conn, sid) >= self._cap:
            raise VerbatimError(
                ErrorCode.BACKPRESSURE,
                f"queue saturated: {self._cap} pending jobs in scope",
                retryable=True,
            )

        now = self._now()
        refs = dict(input_refs)
        refs[_ENQUEUED_US_KEY] = now
        try:
            refs_json = json_dumps(refs)
        except (TypeError, ValueError) as exc:
            raise VerbatimError(
                ErrorCode.VALIDATION, "input_refs must be JSON-serializable"
            ) from exc
        deadline_us = now + int(deadline_s * 1_000_000) if deadline_s is not None else None

        job_id = new_id()
        extra_cols = ""
        extra_vals: list[Any] = []
        if self._has_opkey:
            extra_cols += ", operation_key"
            extra_vals.append(operation_key)
        if self._has_lane:
            extra_cols += ", lane"
            extra_vals.append(resolved_lane)
        insert_sql = (
            "INSERT INTO jobs (job_id, scope_id, kind, state, dedup_key,"
            " input_refs_json, policy_epoch, attempts, not_before_us,"
            f" deadline_us, generation{extra_cols})"
            f" VALUES (?,?,?,?,?,?,?,0,?,?,0{',?' * len(extra_vals)})"
        )
        insert_params = (
            job_id,
            sid,
            k,
            JobState.QUEUED.value,
            bytes(dedup_key) if dedup_key is not None else None,
            refs_json,
            self._policy_epoch if policy_epoch is None else policy_epoch,
            not_before_us,
            deadline_us,
            *extra_vals,
        )
        try:
            conn.execute(insert_sql, insert_params)
        except sqlite3.IntegrityError as exc:
            if "CHECK constraint failed" in str(exc):
                # A valid JobKind/state rejected by the table CHECK means the
                # schema predates the enum (e.g. the v1 seven-kind list
                # still on this store). That is a schema capability gap —
                # report it loudly rather than letting the job vanish.
                raise VerbatimError(
                    ErrorCode.SCHEMA_UNSUPPORTED,
                    f"jobs table CHECK rejected kind {k!r}: {exc}",
                ) from exc
            # dedup_key collision: converge on the already-durable job so
            # retries never duplicate effects (at-least-once + idempotent).
            if dedup_key is None:
                raise
            existing = _row(
                conn.execute(
                    "SELECT job_id, state FROM jobs WHERE dedup_key = ?",
                    (bytes(dedup_key),),
                )
            )
            if existing is None:
                raise
            if existing["state"] not in (
                JobState.FAILED.value,
                JobState.CANCELLED.value,
            ):
                # In-flight (queued/leased/retry_wait) or already-completed
                # work: the dedup contract returns the durable job — a
                # succeeded row *is* the receipt for this dedup key.
                return existing["job_id"]
            # Terminal but unfinished (failed/cancelled): the dead attempt
            # must not pin the dedup key forever — a permanent failure
            # would otherwise make this work item un-re-drivable. Vacate
            # the key on the dead row (suffixed with its job_id, so the
            # lineage stays inspectable and uniqueness is preserved) and
            # insert a fresh attempt under the canonical key.
            conn.execute(
                "UPDATE jobs SET dedup_key = ? WHERE job_id = ?",
                (
                    bytes(dedup_key)
                    + b"\x00dead\x00"
                    + existing["job_id"].encode("utf-8"),
                    existing["job_id"],
                ),
            )
            conn.execute(insert_sql, insert_params)

        self._append_event(conn, job_id, JobState.QUEUED.value, None)
        return job_id

    # ------------------------------------------------------------------
    # worker API
    # ------------------------------------------------------------------

    def lease(
        self,
        scope_id: Union[Scope, str, None],
        kinds: Iterable[Union[JobKind, str]],
        owner: str,
        limit: int = 1,
        lease_s: float = DEFAULT_LEASE_S,
        now_us: Optional[int] = None,
        lane: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Atomically lease due jobs for ``owner``; returns leased job dicts.

        One transaction performs selection + UPDATE so two workers can never
        lease the same row. Each lease bumps ``generation`` and ``attempts`` —
        the generation is the fencing token the worker must present at commit
        time. Control-lane jobs sort ahead of ordinary work unconditionally —
        suppression, rebuild, and review application are privacy/correctness
        obligations, not throughput work, so a saturated enrichment queue can
        never starve them (SPEC §22, V2-39.11). ``lane='control'`` /
        ``lane='ordinary'`` restricts the selection to one lane; on a
        pre-v2 schema the lane degrades to the control kind set.

        ``scope_id=None`` drains across scopes — intended for trusted
        standalone workers; per-job authorization is still re-verified by the
        worker against the job's own scope at dispatch/commit (SPEC §9, §22).

        Jobs past ``deadline_us`` are still leased: silently skipping them
        would strand them in ``queued`` forever. The worker sees the deadline
        in the returned dict and can fail fast — expiry is a visible
        transition, not a silent drop (SPEC §43).
        """
        sid = None if scope_id is None else scope_key(scope_id)
        kind_values = [_kind_value(k) for k in kinds]
        if not kind_values:
            raise VerbatimError(ErrorCode.VALIDATION, "kinds must be non-empty")
        if limit < 1:
            raise VerbatimError(ErrorCode.VALIDATION, "limit must be >= 1")
        if lease_s <= 0:
            raise VerbatimError(ErrorCode.VALIDATION, "lease_s must be positive")
        if lane is not None and lane not in LANES:
            raise VerbatimError(ErrorCode.VALIDATION, f"unknown job lane {lane!r}")
        require_id(owner, "owner")
        now = self._now() if now_us is None else now_us
        lease_until = now + int(lease_s * 1_000_000)

        placeholders = ",".join("?" for _ in kind_values)
        clauses = [f"kind IN ({placeholders})"]
        params_list: list[Any] = list(kind_values)
        if sid is not None:
            clauses.insert(0, "scope_id = ?")
            params_list.insert(0, sid)
        clauses.append("state IN ('queued','retry_wait')")
        clauses.append("not_before_us <= ?")
        params_list.append(now)
        if lane is not None:
            if self._has_lane:
                clauses.append("lane = ?")
                params_list.append(lane)
            else:
                ctrl = ",".join(f"'{k}'" for k in sorted(CONTROL_KINDS))
                op = "IN" if lane == "control" else "NOT IN"
                clauses.append(f"kind {op} ({ctrl})")
        if self._has_lane:
            # v3 four-lane priority (§40): privacy/control work first, then
            # maintenance, then background learning, then ordinary. Legacy
            # 'control' rows still sort ahead of everything.
            lead = _LANE_ORDER_SQL
        else:
            ctrl = ",".join(f"'{k}'" for k in sorted(CONTROL_KINDS))
            lead = f"(kind IN ({ctrl})) DESC"
        order = (
            f"{lead}, {_KIND_ORDER_SQL}, (kind = 'purge') DESC,"
            " not_before_us ASC, rowid ASC"
        )
        params_list.append(limit)
        where = " AND ".join(clauses)
        with self._store.tx() as conn:
            rows = _rows(
                conn.execute(
                    f"SELECT job_id FROM jobs WHERE {where} ORDER BY {order} LIMIT ?",
                    params_list,
                )
            )
            ids = [r["job_id"] for r in rows]
            if not ids:
                return []
            id_ph = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE jobs SET state = 'leased', lease_owner = ?,"
                f" lease_until_us = ?, generation = generation + 1,"
                f" attempts = attempts + 1 WHERE job_id IN ({id_ph})",
                (owner, lease_until, *ids),
            )
            for jid in ids:
                self._append_event(conn, jid, JobState.LEASED.value, None)
            leased = _rows(
                conn.execute(
                    f"SELECT * FROM jobs WHERE job_id IN ({id_ph})"
                    f" ORDER BY {lead}, {_KIND_ORDER_SQL},"
                    " (kind = 'purge') DESC, rowid ASC",
                    ids,
                )
            )
        return [_row_to_job(r) for r in leased]

    def lease_priority(
        self,
        scope_id: Union[Scope, str, None],
        kinds: Iterable[Union[JobKind, str]],
        *,
        source_ids: Optional[Iterable[str]] = None,
        owner: str,
        limit: int = 1,
        lease_s: float = DEFAULT_LEASE_S,
        lane: Union[str, Iterable[str], None] = None,
    ) -> list[dict[str, Any]]:
        """Lease due jobs filtered to a payload ``source_id`` set and/or a
        lane set — the unblock-first pass's private helper (V6-02.08,
        v6_contracts §3).

        Identical transaction shape and fencing as :meth:`lease`: one
        transaction performs selection + ``UPDATE``, every leased row gets
        the ``generation``/``attempts`` bump and ``lease_owner``/
        ``lease_until_us`` stamp, and the same lane/kind/not_before/rowid
        ordering applies. Two extra, optional filters:

        * ``source_ids`` — restrict to rows whose ``input_refs_json``
          ``$.source_id`` is in the set (the convention every pipeline job
          uses: harvest/admit/source_* all carry ``source_id`` refs).
          ``None`` applies no source filter (the drainer's non-ordinary
          lane scan reuses this helper); an empty or all-invalid set
          trivially matches nothing and returns ``[]``.
        * ``lane`` — a single lane name or an iterable of lane names
          (``lane IN (...)`` on schema v2). On a pre-v2 store the lane set
          degrades to the control-kind mapping exactly like :meth:`lease`:
          a lone ``'control'`` selects control kinds, any other lone lane
          selects non-control kinds, and a multi-lane set without
          ``'ordinary'`` maps to the control kinds — the only non-ordinary
          work a lane-less schema can express.

        This is deliberately NOT a second public scheduler: the only
        caller is ``Ingester.drain_report``'s bounded unblock-first pass.
        Returns ``[]`` when nothing due matches.
        """
        sid = None if scope_id is None else scope_key(scope_id)
        kind_values = [_kind_value(k) for k in kinds]
        if not kind_values:
            raise VerbatimError(ErrorCode.VALIDATION, "kinds must be non-empty")
        if limit < 1:
            raise VerbatimError(ErrorCode.VALIDATION, "limit must be >= 1")
        if lease_s <= 0:
            raise VerbatimError(ErrorCode.VALIDATION, "lease_s must be positive")
        require_id(owner, "owner")
        now = self._now()
        lease_until = now + int(lease_s * 1_000_000)

        if lane is None:
            lanes: Optional[tuple[str, ...]] = None
        elif isinstance(lane, str):
            lanes = (lane,)
        else:
            lanes = tuple(lane)
        if lanes is not None:
            bad = [l for l in lanes if l not in LANES]
            if bad:
                raise VerbatimError(
                    ErrorCode.VALIDATION, f"unknown job lane {bad[0]!r}"
                )
            if not lanes:
                return []

        clauses = [f"kind IN ({','.join('?' for _ in kind_values)})"]
        params_list: list[Any] = list(kind_values)
        if sid is not None:
            clauses.insert(0, "scope_id = ?")
            params_list.insert(0, sid)
        clauses.append("state IN ('queued','retry_wait')")
        clauses.append("not_before_us <= ?")
        params_list.append(now)
        if source_ids is not None:
            sids = [
                s
                for s in dict.fromkeys(source_ids)
                if isinstance(s, str) and s
            ]
            if not sids:
                return []
            clauses.append(
                "json_extract(input_refs_json, '$.source_id') IN"
                f" ({','.join('?' for _ in sids)})"
            )
            params_list.extend(sids)
        if lanes is not None:
            if self._has_lane:
                clauses.append(
                    f"lane IN ({','.join('?' for _ in lanes)})"
                )
                params_list.extend(lanes)
            else:
                ctrl = ",".join(f"'{k}'" for k in sorted(CONTROL_KINDS))
                if len(lanes) == 1:
                    # Exact parity with lease()'s single-lane fallback.
                    op = "IN" if lanes[0] == "control" else "NOT IN"
                    clauses.append(f"kind {op} ({ctrl})")
                elif "ordinary" in lanes:
                    # Ordinary + any non-ordinary lane covers every v1
                    # kind — the lane filter is a no-op.
                    pass
                else:
                    # A lane-less schema's only non-ordinary work is the
                    # control kind set.
                    clauses.append(f"kind IN ({ctrl})")
        if self._has_lane:
            lead = _LANE_ORDER_SQL
        else:
            ctrl = ",".join(f"'{k}'" for k in sorted(CONTROL_KINDS))
            lead = f"(kind IN ({ctrl})) DESC"
        order = (
            f"{lead}, {_KIND_ORDER_SQL}, (kind = 'purge') DESC,"
            " not_before_us ASC, rowid ASC"
        )
        params_list.append(limit)
        where = " AND ".join(clauses)
        with self._store.tx() as conn:
            rows = _rows(
                conn.execute(
                    f"SELECT job_id FROM jobs WHERE {where} ORDER BY {order} LIMIT ?",
                    params_list,
                )
            )
            ids = [r["job_id"] for r in rows]
            if not ids:
                return []
            id_ph = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE jobs SET state = 'leased', lease_owner = ?,"
                f" lease_until_us = ?, generation = generation + 1,"
                f" attempts = attempts + 1 WHERE job_id IN ({id_ph})",
                (owner, lease_until, *ids),
            )
            for jid in ids:
                self._append_event(conn, jid, JobState.LEASED.value, None)
            leased = _rows(
                conn.execute(
                    f"SELECT * FROM jobs WHERE job_id IN ({id_ph})"
                    f" ORDER BY {lead}, {_KIND_ORDER_SQL},"
                    " (kind = 'purge') DESC, rowid ASC",
                    ids,
                )
            )
        return [_row_to_job(r) for r in leased]

    # ------------------------------------------------------------------
    # dequeue-level coalescing (V8-13.05)
    # ------------------------------------------------------------------

    def _sibling_where(
        self, job: dict[str, Any], now: int
    ) -> tuple[str, list[Any]]:
        """The due-sibling predicate shared by :meth:`pending_siblings`
        and :meth:`claim_siblings`.

        A sibling is a *different* job row of the same kind in the same
        scope partition that the drain could lease right now — pending
        (``queued``/``retry_wait``), delay elapsed, deadline unexpired,
        and on schema v2 the same lane (a pinned-lane worker never
        crosses lane boundaries). The predicate mirrors :meth:`lease`
        minus the multi-kind scan so the coalesced batch is exactly the
        set the drain would have run serially.
        """
        clauses = [
            "scope_id = ?",
            "kind = ?",
            "job_id != ?",
            "state IN ('queued','retry_wait')",
            "not_before_us <= ?",
            "(deadline_us IS NULL OR deadline_us > ?)",
        ]
        params: list[Any] = [
            job["scope_id"], job["kind"], job["job_id"], now, now,
        ]
        if self._has_lane:
            clauses.append("lane = ?")
            params.append(job.get("lane") or "ordinary")
        return " AND ".join(clauses), params

    def pending_siblings(
        self,
        job: dict[str, Any],
        *,
        limit: int,
        now_us: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Read-snapshot peek at due same-kind+same-scope jobs (V8-13.05).

        Advisory only: the answer is what a same-moment lease scan would
        pick — the in-transaction :meth:`claim_siblings` re-verifies
        every condition before a sibling is leased, so a stale peek can
        only under-claim, never mis-claim. Rows come back in the queue's
        serial order (``not_before_us``, then ``rowid``).
        """
        if limit < 1:
            return []
        now = self._now() if now_us is None else now_us
        where, params = self._sibling_where(job, now)
        with self._store.read() as conn:
            rows = _rows(
                conn.execute(
                    f"SELECT * FROM jobs WHERE {where}"
                    " ORDER BY not_before_us ASC, rowid ASC LIMIT ?",
                    (*params, limit),
                )
            )
        return [_row_to_job(r) for r in rows]

    def claim_siblings(
        self,
        conn: sqlite3.Connection,
        job: dict[str, Any],
        *,
        owner: str,
        limit: int,
        job_ids: Optional[Iterable[str]] = None,
        lease_s: float = DEFAULT_LEASE_S,
        now_us: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Atomically claim due sibling jobs inside the caller's write
        transaction — the dequeue-level coalescing primitive (V8-13.05).

        Same fencing as :meth:`lease`: each claimed row flips to
        ``leased`` with the owner stamp and a ``generation``/``attempts``
        bump — the generation is the fencing token the worker presents at
        commit. Because the claim rides the caller's own ``store.tx()``,
        the one-writer rule is untouched and the claim is atomic with the
        batch's domain effects: a crash mid-batch rolls the claim back
        with everything else, so no sibling is ever lost, and a sibling
        already taken by another worker simply fails the conditional
        UPDATE — never double-claimed, never double-applied.

        ``job_ids`` restricts the claim to a caller-staged set (the
        handler derives only the jobs it has prepared); ``None`` claims
        the top ``limit`` due siblings in serial order. Ordering follows
        the lease order — ``not_before_us``, then ``rowid`` — so the
        batch is the same FIFO the drain would have run.
        """
        if limit < 1:
            return []
        require_id(owner, "owner")
        now = self._now() if now_us is None else now_us
        lease_until = now + int(lease_s * 1_000_000)
        where, params = self._sibling_where(job, now)
        if job_ids is not None:
            ids = [
                j for j in dict.fromkeys(job_ids) if isinstance(j, str) and j
            ]
            if not ids:
                return []
            where += f" AND job_id IN ({','.join('?' for _ in ids)})"
            params = [*params, *ids]
        rows = _rows(
            conn.execute(
                f"SELECT job_id FROM jobs WHERE {where}"
                " ORDER BY not_before_us ASC, rowid ASC LIMIT ?",
                (*params, limit),
            )
        )
        ids = [r["job_id"] for r in rows]
        if not ids:
            return []
        ph = ",".join("?" for _ in ids)
        # Conditional claim: the sibling predicate is re-checked inside
        # the UPDATE so a job reclaimed/cancelled between the SELECT and
        # this statement is never touched (both run under the same write
        # lock — the re-check is the durable contract, not a race hedge).
        cur = conn.execute(
            f"UPDATE jobs SET state = 'leased', lease_owner = ?,"
            f" lease_until_us = ?, generation = generation + 1,"
            f" attempts = attempts + 1 WHERE {where} AND job_id IN ({ph})",
            (owner, lease_until, *params, *ids),
        )
        if cur.rowcount != len(ids):
            # A staged id failed the predicate — re-read exactly which
            # claimed so the caller only ever runs jobs it owns.
            claimed = _rows(
                conn.execute(
                    f"SELECT job_id FROM jobs WHERE job_id IN ({ph})"
                    " AND state = 'leased' AND lease_owner = ?",
                    (*ids, owner),
                )
            )
            ids = [r["job_id"] for r in claimed]
            if not ids:
                return []
            ph = ",".join("?" for _ in ids)
        for jid in ids:
            self._append_event(conn, jid, JobState.LEASED.value, None)
        leased = _rows(
            conn.execute(
                f"SELECT * FROM jobs WHERE job_id IN ({ph})"
                " ORDER BY not_before_us ASC, rowid ASC",
                ids,
            )
        )
        return [_row_to_job(r) for r in leased]

    def release_siblings(
        self,
        conn: sqlite3.Connection,
        job_ids: Iterable[str],
        *,
        owner: str,
    ) -> int:
        """Return claimed siblings to ``queued`` inside the caller's
        write transaction — the coalescing budget-release (V8-13.05).

        A shared commit that exhausts its wall-clock budget hands back
        the claimed-but-unprocessed tail so the write lock is released
        for foreground writers; released rows re-lease on the next drain
        pass — never lost, never double-applied. The claim's
        ``attempts`` bump is undone (a released job was never attempted);
        ``generation`` still advances so a stale fencing token can never
        replay. Conditional on this owner still holding the lease — a
        row reclaimed/cancelled meanwhile is untouched.
        """
        ids = [
            j for j in dict.fromkeys(job_ids) if isinstance(j, str) and j
        ]
        if not ids:
            return 0
        require_id(owner, "owner")
        ph = ",".join("?" for _ in ids)
        cur = conn.execute(
            f"UPDATE jobs SET state = 'queued', lease_owner = NULL,"
            f" lease_until_us = NULL, generation = generation + 1,"
            f" attempts = MAX(0, attempts - 1)"
            f" WHERE job_id IN ({ph}) AND state = 'leased'"
            " AND lease_owner = ?",
            (*ids, owner),
        )
        for jid in ids:
            self._append_event(conn, jid, JobState.QUEUED.value, None)
        return cur.rowcount

    @property
    def supports_durability(self) -> bool:
        """Schema-v2 job columns (``operation_key`` + ``lane``) are present."""
        return self._has_opkey and self._has_lane

    def commit_if_current(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        owner: str,
        generation: int,
    ) -> bool:
        """True while the worker's lease generation still matches the row.

        Workers call this immediately before writing effects inside the same
        transaction; a False means the lease was reclaimed or cancelled and
        any staged result must be discarded (SPEC §22 generation fencing).
        """
        return self._is_current(self._fetch(conn, job_id), owner, generation)

    def assert_lease(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        owner: str,
        generation: int,
        now_us: Optional[int] = None,
    ) -> None:
        """Commit-gate fence: raise ``LEASE_LOST`` unless the lease is live.

        Call inside the transaction that commits a job's domain effects —
        the final check before ``complete``. Beyond ``commit_if_current``
        (state/owner/generation match) this also rejects a lease whose
        persisted ``lease_until_us`` has already passed, so a stalled worker
        that still *looks* current cannot commit behind its own expiry
        (V2-39.05/39.06). The error is retryable: the attempt is fenced
        out, but the job itself is intact and will be redelivered.
        """
        row = self._fetch(conn, job_id)
        if not self._is_current(row, owner, generation):
            raise VerbatimError(
                ErrorCode.LEASE_LOST,
                f"job {job_id} lease lost to another generation —"
                " staged effects discarded",
                retryable=True,
            )
        until = row["lease_until_us"]
        now = self._now() if now_us is None else now_us
        if until is not None and until <= now:
            raise VerbatimError(
                ErrorCode.LEASE_LOST,
                f"job {job_id} lease expired — staged effects discarded",
                retryable=True,
            )

    def complete(
        self,
        conn_or_job_id: Union[sqlite3.Connection, str],
        job_id: Optional[str] = None,
        owner: Optional[str] = None,
        generation: Optional[int] = None,
        result_summary: Optional[str] = None,
    ) -> bool:
        """Mark a leased job succeeded under generation fencing.

        Two call shapes are supported:
        ``complete(conn, job_id, owner, generation[, result_summary])`` —
        the transition joins the caller's transaction — and
        ``complete(job_id, owner, generation)`` — the queue opens its own
        ``store.tx()``. The conn form is required when the worker commits
        domain results in the same atomic unit.

        Returns False when the lease is stale — the caller must treat its
        staged effects as discarded. ``result_summary`` is accepted for API
        stability but intentionally not persisted: schema v1 carries no
        result column and job_events deliberately stores no payloads
        (SPEC §20); real results belong in domain tables written under the
        same fencing check.
        """
        if isinstance(conn_or_job_id, sqlite3.Connection):
            conn, jid, own, gen = conn_or_job_id, job_id, owner, generation
            return self._complete(conn, jid, own, gen, result_summary)
        # Snapshot precheck: generations are monotonic and 'succeeded' is
        # terminal, so a row that is not this lease's generation can never
        # satisfy the fence again — the False answer needs no write tx.
        with self._store.read() as conn:
            if not self._is_current(
                self._fetch(conn, conn_or_job_id), job_id, owner
            ):
                return False
        with self._store.tx() as conn:
            # shifted positionals: (job_id, owner, generation)
            return self._complete(conn, conn_or_job_id, job_id, owner, None)

    def _complete(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        owner: str,
        generation: int,
        result_summary: Optional[str],
    ) -> bool:
        row = self._fetch(conn, job_id)
        if not self._is_current(row, owner, generation):
            return False
        conn.execute(
            "UPDATE jobs SET state = 'succeeded', lease_owner = NULL,"
            " lease_until_us = NULL, error_code = NULL WHERE job_id = ?",
            (job_id,),
        )
        self._append_event(conn, job_id, JobState.SUCCEEDED.value, None)
        return True

    def fail(
        self,
        conn_or_job_id: Union[sqlite3.Connection, str],
        job_id: Optional[str] = None,
        owner: Optional[str] = None,
        generation: Optional[int] = None,
        error_code: Union[ErrorCode, str, None] = None,
        retryable: bool = False,
        max_attempts: int = 3,
    ) -> Optional[JobState]:
        """Record failure under fencing; returns the new state or None if stale.

        Two call shapes: ``fail(conn, job_id, owner, generation, error_code,
        retryable[, max_attempts])`` and the self-transacting
        ``fail(job_id, owner, generation, error_code, retryable)``.

        Retryable errors go to ``retry_wait`` with exponential backoff
        (base 2s, doubled per attempt, capped at 300s, ±50% jitter) and never
        earlier than a pre-existing ``not_before_us`` so server-provided
        delay hints are honored. Attempts at or beyond ``max_attempts``, and
        non-retryable errors, are terminal. Authentication/validation
        failures are expected to arrive with ``retryable=False`` — they are
        not retried until their inputs change (SPEC §22).
        """
        if isinstance(conn_or_job_id, sqlite3.Connection):
            return self._fail(
                conn_or_job_id, job_id, owner, generation, error_code, retryable, max_attempts
            )
        # Same stale-lease precheck as complete(): a non-current row is a
        # durable None — fencing is monotone, never re-enabled.
        with self._store.read() as conn:
            if not self._is_current(
                self._fetch(conn, conn_or_job_id), job_id, owner
            ):
                return None
        with self._store.tx() as conn:
            # shifted positionals: (job_id, owner, generation, error_code, retryable)
            return self._fail(
                conn, conn_or_job_id, job_id, owner, generation, error_code, max_attempts
            )

    def _fail(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        owner: str,
        generation: int,
        error_code: Union[ErrorCode, str],
        retryable: bool,
        max_attempts: int,
    ) -> Optional[JobState]:
        code = error_code.value if isinstance(error_code, ErrorCode) else str(error_code)
        row = self._fetch(conn, job_id)
        if not self._is_current(row, owner, generation):
            return None
        now = self._now()
        attempts = row["attempts"]
        if retryable and attempts < max_attempts:
            delay_s = min(RETRY_CAP_S, RETRY_BASE_S * (2 ** max(0, attempts - 1)))
            jittered = min(RETRY_CAP_S, delay_s * self._rand.uniform(0.5, 1.5))
            not_before = max(now + int(jittered * 1_000_000), row["not_before_us"] or 0)
            conn.execute(
                "UPDATE jobs SET state = 'retry_wait', not_before_us = ?,"
                " lease_owner = NULL, lease_until_us = NULL, error_code = ?"
                " WHERE job_id = ?",
                (not_before, code, job_id),
            )
            self._append_event(conn, job_id, JobState.RETRY_WAIT.value, code)
            return JobState.RETRY_WAIT
        conn.execute(
            "UPDATE jobs SET state = 'failed', lease_owner = NULL,"
            " lease_until_us = NULL, error_code = ? WHERE job_id = ?",
            (code, job_id),
        )
        self._append_event(conn, job_id, JobState.FAILED.value, code)
        return JobState.FAILED

    def cancel(
        self,
        conn_or_job_id: Union[sqlite3.Connection, str],
        job_id: Optional[str] = None,
    ) -> bool:
        """Cancel a non-terminal job; returns False if it was already terminal.

        ``generation`` is bumped so any worker still holding a lease on the
        job is fenced out and can no longer commit a result for it.
        """
        if isinstance(conn_or_job_id, sqlite3.Connection):
            return self._cancel(conn_or_job_id, job_id)
        # Terminal states are absorbing and a missing row never appears
        # (job_ids mint at insert) — both answers are durable without a
        # write tx.
        with self._store.read() as conn:
            row = self._fetch(conn, conn_or_job_id)
        if row is None:
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown job")
        if row["state"] in _TERMINAL:
            return False
        with self._store.tx() as conn:
            return self._cancel(conn, conn_or_job_id)

    def _cancel(self, conn: sqlite3.Connection, job_id: str) -> bool:
        row = self._fetch(conn, job_id)
        if row is None:
            raise VerbatimError(ErrorCode.NOT_FOUND_OR_FORBIDDEN, "unknown job")
        if row["state"] in _TERMINAL:
            return False
        conn.execute(
            "UPDATE jobs SET state = 'cancelled', lease_owner = NULL,"
            " lease_until_us = NULL, generation = generation + 1 WHERE job_id = ?",
            (job_id,),
        )
        self._append_event(conn, job_id, JobState.CANCELLED.value, None)
        return True

    def reclaim_expired(
        self,
        conn_or_now: Union[sqlite3.Connection, int, None] = None,
        now_us: Optional[int] = None,
    ) -> int:
        """Return expired leases to ``retry_wait``; returns the count reclaimed.

        ``reclaim_expired(conn[, now_us])`` joins the caller's transaction;
        ``reclaim_expired([now_us])`` self-transacts. The bumped generation
        is the fence: a worker that wakes up after its lease was reclaimed
        fails ``commit_if_current`` and cannot overwrite the replacement's
        result.
        """
        if isinstance(conn_or_now, sqlite3.Connection):
            conn = conn_or_now
            now = self._now() if now_us is None else now_us
            return self._reclaim(conn, now)
        now = self._now() if conn_or_now is None else conn_or_now
        if now_us is not None:
            now = now_us
        # Nothing to reclaim is a durable-enough answer for this tick: an
        # empty snapshot can only under-read a lease that expires inside
        # the precheck window, which the next reclaim pass picks up —
        # identical to the tx having run a hair earlier.
        with self._store.read() as conn:
            expired = conn.execute(
                "SELECT 1 FROM jobs WHERE state = 'leased'"
                " AND lease_until_us IS NOT NULL AND lease_until_us <= ?"
                " LIMIT 1",
                (now,),
            ).fetchone()
        if expired is None:
            return 0
        with self._store.tx() as conn:
            return self._reclaim(conn, now)

    def _reclaim(self, conn: sqlite3.Connection, now: int) -> int:
        rows = _rows(
            conn.execute(
                "SELECT job_id FROM jobs WHERE state = 'leased' AND lease_until_us IS NOT NULL"
                " AND lease_until_us <= ?",
                (now,),
            )
        )
        ids = [r["job_id"] for r in rows]
        if not ids:
            return 0
        ph = ",".join("?" for _ in ids)
        # A reclaimed job is already overdue — it becomes due at the earlier
        # of the observation horizon and the queue's own clock, so a scan
        # performed against a simulated-future horizon cannot push
        # redelivery into the future.
        due = now if now <= self._now() else self._now()
        conn.execute(
            f"UPDATE jobs SET state = 'retry_wait', not_before_us = ?,"
            f" lease_owner = NULL, lease_until_us = NULL,"
            f" generation = generation + 1 WHERE job_id IN ({ph})",
            (due, *ids),
        )
        for jid in ids:
            self._append_event(conn, jid, JobState.RETRY_WAIT.value, "lease_expired")
        return len(ids)

    # ------------------------------------------------------------------
    # inspection API
    # ------------------------------------------------------------------

    def list(
        self,
        scope_id: Union[Scope, str, None],
        state: Optional[Union[JobState, str]] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read-only listing for operators/dead-letter views (newest first).

        ``scope_id=None`` lists across scopes (operator view); a Scope or
        scope_id string filters to that partition.
        """
        sid = None if scope_id is None else scope_key(scope_id)
        if limit < 1:
            raise VerbatimError(ErrorCode.VALIDATION, "limit must be >= 1")
        scope_clause = "" if sid is None else "scope_id = ? AND "
        base_params: tuple = () if sid is None else (sid,)
        with self._store.read() as conn:
            if state is None:
                rows = _rows(
                    conn.execute(
                        f"SELECT * FROM jobs WHERE {scope_clause}1=1"
                        " ORDER BY rowid DESC LIMIT ?",
                        (*base_params, limit),
                    )
                )
            else:
                try:
                    sv = state.value if isinstance(state, JobState) else JobState(state).value
                except ValueError as exc:
                    raise VerbatimError(
                        ErrorCode.VALIDATION, f"unknown job state {state!r}"
                    ) from exc
                rows = _rows(
                    conn.execute(
                        f"SELECT * FROM jobs WHERE {scope_clause}state = ?"
                        " ORDER BY rowid DESC LIMIT ?",
                        (*base_params, sv, limit),
                    )
                )
        return [_row_to_job(r) for r in rows]

    def stats(self, scope_id: Union[Scope, str, None]) -> dict[str, Any]:
        """Queue health for one scope — or process-wide when None (SPEC §42).

        ``oldest_age_us`` uses the enqueue timestamp recorded inside
        input_refs at insert time; it is None when nothing is pending.
        """
        sid = None if scope_id is None else scope_key(scope_id)
        now = self._now()
        scope_clause = "" if sid is None else "scope_id = ? AND "
        p: tuple = () if sid is None else (sid,)
        with self._store.read() as conn:
            pending = _row(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM jobs WHERE {scope_clause}"
                    "state IN ('queued','retry_wait')",
                    p,
                )
            )
            leased = conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE {scope_clause}state = 'leased'",
                p,
            ).fetchone()[0]
            failed = conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE {scope_clause}state = 'failed'",
                p,
            ).fetchone()[0]
            oldest = _rows(
                conn.execute(
                    f"SELECT input_refs_json FROM jobs WHERE {scope_clause}"
                    "state IN ('queued','retry_wait') ORDER BY rowid ASC LIMIT 200",
                    p,
                )
            )
        oldest_age: Optional[int] = None
        min_enq: Optional[int] = None
        for r in oldest:
            try:
                refs = safe_json_loads(r["input_refs_json"] or "{}")
            except VerbatimError:
                continue
            enq = refs.get(_ENQUEUED_US_KEY) if isinstance(refs, dict) else None
            if isinstance(enq, int) and (min_enq is None or enq < min_enq):
                min_enq = enq
        if min_enq is not None:
            oldest_age = max(0, now - min_enq)
        return {
            "pending": pending["n"],
            "oldest_age_us": oldest_age,
            "leased": leased,
            "failed": failed,
        }
