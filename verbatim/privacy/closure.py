"""Resumable deletion-closure engine (SPEC_V4 §38, F4-07).

ONE engine for every deletion path (V4-05.05): ``handle_purge_derived`` and
the ordinary purge path (``purge._execute``) both run this machinery — the
512-child truncation is gone; closure is a durable, crash-safe state machine.

Lifecycle (V4-38 state table):

* ``begin(roots, scope_id)`` — one transaction that tombstones the roots
  (purge-registry rows + ``recorded_until`` markers), advances the erasure
  epoch, creates the ``closure_runs`` row, and queues the first frontier
  work. Async cleanup is acknowledged only *after* suppression is durable
  (V4-38.01). Descendants are excluded immediately by ancestry/epoch checks
  (``is_excluded``) even before enumeration reaches them (V4-38.02).
* ``step(run_id, budget)`` — a bounded batch of frontier work inside one
  transaction. Every unit of work is a ``closure_frontier`` row with a
  monotone cursor — the durable continuation (V4-38.03). Hitting the budget
  leaves ``pending`` rows and the run in ``cleaning``; it NEVER reports
  terminal success with unprocessed edges (V4-38.04).
* ``verify(run_id)`` — honest closure check (V3-36.04 carried into §38):
  purged objects gone per kind semantics, suppression/revalidation markers
  present, no surviving edge or member row into purged objects, no
  pending/failed frontier rows.
* ``resume(run_id)`` — re-queues failed units and re-opens enumeration so a
  crashed or partially-failed run converges without lost work (V4-38.11).

Frontier work items (``closure_frontier.action``):

* ``enumerate`` — one row per member; pages over the node's children across
  ``derivations``, ``dependency_edges``, and registered side references;
  keyset-paginated, resumable mid-node. Each newly discovered child gets its
  own ``enumerate`` row (member dedup is part of the run's durable state).
* ``settle`` — one row per *batch* of members (``detail.keys``); classifies
  every member over the *final* member set (all enumeration is complete
  before settling starts, so ancestry is fully known) and applies the
  marker-level suppression semantics: ``suppress`` members are withheld
  until recomputed (V4-38.06), ``revalidate`` members get the freshness
  flag, ``outside`` members (foreign scope / unresolvable kind) are
  disclosed untouched, ``erase`` members are queued into ``finalize``
  batches. Classification is a DFS over the member graph with memoized,
  row-persisted decisions — identical fixpoint semantics to
  ``privacy.deletion.plan_closure`` but incremental and bounded.
* ``finalize`` — one row per *batch* of erased members (``detail.keys``):
  suppression marker then physical delete per member, then every graph
  edge and secondary reference naming the purged objects, propagated-copy
  revocation+disclosure, the opaque erasure-ledger tombstone, and pending-
  job cancellation (V4-38.05). Finalize rows are always minted at cursors
  above every settle row, so edge stripping runs strictly after all
  classification consumed those edges — cursor monotonicity is the
  ordering guarantee, no separate coordination.

Crash semantics: every ``step`` commits or rolls back atomically; a kill
between or inside steps leaves a consistent frontier that ``resume`` (or the
next ``step``) picks up with no lost work. A failed unit is marked ``failed``
and the run lands ``failed_cleanup`` — suppression is retained, obligations
stay retryable, and visibility is never restored to improve availability
(V4-38.11).

Scale note: settle/finalize rows carry batches (``_SETTLE_BATCH`` /
``_FIN_BATCH`` members) and their physical effects are applied with grouped
``IN``-chunk statements; enumerate pages prefetch children in bulk. The
durable unit is still the frontier row — batching changes efficiency, not
semantics.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional

from ..core.lifecycle import PURGE_ACTOR, LifecycleMachine
from ..core.time import wall_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    require_id,
    safe_json_loads,
)
from ..core.types_v4 import ClosurePhase, ClosureRun
from ..purge import _apply_tombstones, _cancel_jobs, _has_table
from ..storage import repos_v4
from ..storage.repos import EventsRepo
from ..storage.repos_v2 import ErasureRepo
from . import deletion as _del

#: Step() processing bound: frontier row-visits per call. Each visit does
#: bounded work (one settle batch, one finalize batch, or one enumerate
#: page), so a step is bounded regardless of graph shape.
DEFAULT_BUDGET = 512

#: drain()'s per-step visit bound — larger than step()'s interactive
#: default so bulk closures commit in a small number of transactions.
_DRAIN_BUDGET = 8192

#: Children consumed per ``enumerate`` row-visit (keyset page size).
_ENUM_PAGE = 256

#: Pending ``enumerate`` rows fetched per poll inside a step.
_ENUM_ROWS = 256

#: Members planned per barrier visit — enumerate rows scanned per step
#: while the planner drains the enumerated member set.
_PLAN_PAGE = 4096

#: Members carried by one ``settle`` frontier row.
_SETTLE_BATCH = 512

#: Erased members carried by one ``finalize`` frontier row.
_FIN_BATCH = 512

#: SQL variable chunk for grouped IN (...) statements.
_IN_CHUNK = 400

#: Drain safety bound for in-process callers (purge/handler). Beyond it the
#: caller must surface CLOSURE_PENDING rather than claim completion.
_MAX_DRAIN_STEPS = 8192

_TERMINAL = (ClosurePhase.COMPLETED, ClosurePhase.FAILED)

#: Member classes (decided action, persisted on the settle row detail).
_C_ERASE = "erase"
_C_SUPPRESS = "suppress"
_C_REVALIDATE = "revalidate"
_C_OUTSIDE = "outside"
_C_UNAFFECTED = "unaffected"
_C_SURVIVOR = "survivor"          # non-member parent — never touched
_C_CYCLE = "cycle"                # in-progress member on a DFS path


def _ref_sort(ref: tuple[str, str, Optional[int]]) -> tuple[str, str, int]:
    """Deterministic member ordering; None revision sorts first."""
    return (ref[0], ref[1], -1 if ref[2] is None else ref[2])


#: Purge registry states that suppress reads (mirrors candidates.py).
_SUPPRESSING_STATES = ("suppressed", "purging", "completed")

#: Kinds whose erasure keeps a bitemporal skeleton row (V2-41.10): the
#: purge path's ``_erase_dated`` tombstones them in place, and the engine
#: honors that as already-gone. Other erased kinds delete their row.
_SKELETON_KINDS = frozenset({"episode", "procedure"})

#: Side walkers whose parent/child directions are bulk-prefetched inside a
#: step. Registered third-party walkers still resolve per member.
_BULK_WALKERS = frozenset({"claim_evidence", "observation_evidence"})

#: Kinds with a simple object table usable for batched existence checks —
#: mirrors ``deletion._object_exists`` for the single-table kinds. Kinds
#: absent here fall back to the per-key check (unchanged semantics).
_EXISTS_TABLES: dict[str, tuple[str, str]] = {
    "envelope": ("source_envelopes", "envelope_id"),
    "episode": ("episodes", "episode_id"),
    "transition": ("transitions", "transition_id"),
    "procedure": ("procedures", "procedure_id"),
    "observation": ("observations", "observation_id"),
    "plan": ("prospective_records", "record_id"),
    "social": ("social_memory", "record_id"),
    "trajectory": ("trajectories", "trajectory_id"),
    "trajectory_step": ("trajectory_steps", "step_id"),
    "state_anchor": ("state_anchors", "anchor_id"),
    "profile": ("profile_entries", "entry_id"),
}

#: Kinds whose ``_delete_object`` semantics are flat row deletes —
#: batched here as ``DELETE FROM t WHERE col IN (...)`` per chunk, in the
#: same table order the per-key path uses. Kinds absent here (claim, span,
#: source, source_revision, artifact, episode, procedure, trajectory —
#: lifecycle machines, byte scrubbing, or multi-level member cascades)
#: keep the per-key ``_del._delete_object`` path.
_BATCH_DELETES: dict[str, tuple[tuple[str, str], ...]] = {
    "observation": (
        ("observation_evidence", "observation_id"),
        ("observations", "observation_id"),
    ),
    "transition": (
        ("transition_anchors", "transition_id"),
        ("transitions", "transition_id"),
    ),
    "trajectory_step": (
        ("step_observations", "step_id"),
        ("trajectory_steps", "step_id"),
    ),
    "state_anchor": (
        ("transition_anchors", "anchor_id"),
        ("state_anchors", "anchor_id"),
    ),
    "plan": (("prospective_records", "record_id"),),
    "social": (("social_memory", "record_id"),),
    "envelope": (
        ("step_observations", "envelope_id"),
        ("source_envelopes", "envelope_id"),
    ),
}

#: Secondary member/link tables scrubbed for references to purged objects —
#: mirrors ``deletion._strip_references``/``_reference_orphans`` coverage.
_MEMBER_TABLES = (
    "observation_evidence",
    "episode_members",
    "family_members",
    "capsule_members",
    "decision_inputs",
    "artifact_links",
)


class _RowFailed(Exception):
    """Internal control flow: one frontier unit failed inside its
    savepoint; the step marks it failed and stops the batch."""

    def __init__(self, item: dict[str, Any], cause: BaseException) -> None:
        super().__init__(str(cause))
        self.item = item
        self.cause = cause


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur: sqlite3.Cursor) -> Optional[dict[str, Any]]:
    rs = _rows(cur)
    return rs[0] if rs else None


def _chunks(seq: list[Any], size: int = _IN_CHUNK) -> Iterable[list[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _placeholders(n: int) -> str:
    return ",".join("?" for _ in range(n))


class _Ctx:
    """Per-step scratch state — caches recomputable inside one transaction.

    Nothing here is authoritative: the durable state lives in
    ``closure_runs``/``closure_frontier`` and the object tables. A fresh
    ``_Ctx`` is built per step, so crash-resume correctness never depends
    on these caches.
    """

    __slots__ = (
        "tables",
        "by_id",
        "parents",
        "scopes",
        "children",
        "next_cursor",
        "seq",
        "jobs",
    )

    def __init__(self) -> None:
        self.tables: Optional[frozenset] = None
        self.by_id: Optional[dict] = None
        self.parents: Optional[dict] = None
        self.scopes: dict[str, dict[str, Optional[str]]] = {}
        self.children: dict[tuple, list] = {}
        self.next_cursor: Optional[int] = None
        self.seq: Optional[int] = None
        self.jobs: Optional[list[dict[str, Any]]] = None

    def has_table(self, conn: sqlite3.Connection, name: str) -> bool:
        if self.tables is None:
            self.tables = frozenset(
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master"
                    " WHERE type IN ('table','view')"
                ).fetchall()
            )
        return name in self.tables

    def alloc_seq(self, conn: sqlite3.Connection) -> int:
        """Block-allocated event-seq counter for marker columns."""
        if self.seq is None:
            self.seq = int(
                conn.execute(
                    "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM events"
                ).fetchone()[0]
            )
        else:
            self.seq += 1
        return self.seq

    def alloc_cursor(self, conn: sqlite3.Connection, run_id: str) -> int:
        if self.next_cursor is None:
            self.next_cursor = int(
                conn.execute(
                    "SELECT COALESCE(MAX(cursor), 0) FROM closure_frontier"
                    " WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
            )
        self.next_cursor += 1
        return self.next_cursor


# ----------------------------------------------------------------------
# registered side references (V4-38.03): extra edge sources the closure
# walks beyond derivations/dependency_edges. Each walker maps a node to
# derivation-shaped children (forward) and parents (reverse).
# ----------------------------------------------------------------------


def _claim_evidence_children(
    conn: sqlite3.Connection, ref: tuple[str, str, Optional[int]]
) -> list[tuple[str, str, Optional[int]]]:
    """span → claims citing it (a claim stands on its evidence spans)."""
    if ref[0] != "span" or not _has_table(conn, "claim_evidence"):
        return []
    return [
        ("claim", r[0], int(r[1]))
        for r in conn.execute(
            "SELECT claim_id, revision FROM claim_evidence WHERE span_id = ?"
            " ORDER BY claim_id, revision",
            (ref[1],),
        ).fetchall()
    ]


def _claim_evidence_parents(
    conn: sqlite3.Connection, ref: tuple[str, str, Optional[int]]
) -> list[tuple[str, str, Optional[int]]]:
    if ref[0] != "claim" or not _has_table(conn, "claim_evidence"):
        return []
    if ref[2] is None:
        rows = conn.execute(
            "SELECT DISTINCT span_id FROM claim_evidence WHERE claim_id = ?",
            (ref[1],),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT span_id FROM claim_evidence"
            " WHERE claim_id = ? AND revision = ?",
            (ref[1], ref[2]),
        ).fetchall()
    return [("span", r[0], None) for r in rows]


def _observation_evidence_children(
    conn: sqlite3.Connection, ref: tuple[str, str, Optional[int]]
) -> list[tuple[str, str, Optional[int]]]:
    """any object → observations citing it as evidence."""
    if not _has_table(conn, "observation_evidence"):
        return []
    sql = (
        "SELECT observation_id, revision FROM observation_evidence"
        " WHERE object_kind = ? AND object_id = ?"
    )
    params: list[Any] = [ref[0], ref[1]]
    if ref[2] is not None:
        sql += " AND object_revision = ?"
        params.append(ref[2])
    sql += " ORDER BY observation_id, revision"
    return [
        ("observation", r[0], int(r[1]))
        for r in conn.execute(sql, params).fetchall()
    ]


def _observation_evidence_parents(
    conn: sqlite3.Connection, ref: tuple[str, str, Optional[int]]
) -> list[tuple[str, str, Optional[int]]]:
    if ref[0] != "observation" or not _has_table(conn, "observation_evidence"):
        return []
    sql = (
        "SELECT object_kind, object_id, object_revision"
        " FROM observation_evidence WHERE observation_id = ?"
    )
    params: list[Any] = [ref[1]]
    if ref[2] is not None:
        sql += " AND revision = ?"
        params.append(ref[2])
    return [
        (r[0], r[1], int(r[2]) if r[2] is not None else None)
        for r in conn.execute(sql, params).fetchall()
    ]


#: The registry: name → (children, parents). Extensions register here, not
#: by editing the engine (V4-19.09-style declared coverage).
_SIDE_WALKERS: dict[str, dict[str, Any]] = {
    "claim_evidence": {
        "children": _claim_evidence_children,
        "parents": _claim_evidence_parents,
    },
    "observation_evidence": {
        "children": _observation_evidence_children,
        "parents": _observation_evidence_parents,
    },
}

#: Edge-table sources walked by enumerate/classify/sweep.
_EDGE_TABLES = ("derivations", "dependency_edges")


def register_side_walker(name: str, children: Any, parents: Any) -> None:
    """Register an additional edge source for closure walks.

    ``children(conn, ref)`` / ``parents(conn, ref)`` return iterable
    ``(kind, object_id, revision|None)`` triples. Registered sources are
    walked by every engine instance on this process.
    """
    require_id(name, "walker_name")
    _SIDE_WALKERS[name] = {"children": children, "parents": parents}


# ----------------------------------------------------------------------
# reference normalization (lax: roots may name already-erased objects and
# kinds the object tables don't know — they are tombstone intents, not reads)
# ----------------------------------------------------------------------


def _norm_ref(item: Any) -> tuple[str, str, Optional[int]]:
    kind: Any
    oid: Any
    rev: Any = None
    if isinstance(item, dict):
        kind = item.get("kind", item.get("object_kind"))
        oid = item.get("id", item.get("object_id"))
        rev = item.get("revision", item.get("object_revision"))
    else:
        try:
            parts = tuple(item)
        except TypeError:
            parts = ()
        if len(parts) == 2:
            kind, oid = parts
        elif len(parts) == 3:
            kind, oid, rev = parts
        else:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "closure refs must be (kind, id[, revision])",
            )
    if not isinstance(kind, str) or not kind:
        raise VerbatimError(ErrorCode.VALIDATION, "closure ref kind required")
    require_id(str(oid), "object_id")
    if rev is not None:
        if isinstance(rev, bool) or not isinstance(rev, int) or rev < 1:
            raise VerbatimError(
                ErrorCode.VALIDATION, "object revision must be an int >= 1"
            )
    return (kind, str(oid), rev)


def _target_oid(kind: str, oid: str, rev: Optional[int]) -> str:
    """object_id form used in purge_targets (v2 composite for revisions)."""
    if kind == "source_revision" and rev is not None and ":" not in oid:
        return f"{oid}:{rev}"
    return oid


# ----------------------------------------------------------------------
# the engine
# ----------------------------------------------------------------------


class ClosureEngine:
    """The one resumable deletion-closure engine (V4-38, F4-07)."""

    def __init__(self, store: Any) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # run state helpers
    # ------------------------------------------------------------------

    def _run_row(self, conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
        row = repos_v4.get(conn, "closure_runs", {"run_id": run_id})
        if row is None:
            raise VerbatimError(
                ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                f"closure run {run_id!r} not found",
            )
        return row

    @staticmethod
    def _to_run(row: dict[str, Any]) -> ClosureRun:
        roots_raw = safe_json_loads(row["roots_json"]) or []
        return ClosureRun(
            run_id=row["run_id"],
            scope_id=row["scope_id"],
            roots=tuple(
                (str(r[0]), str(r[1])) for r in roots_raw if len(r) >= 2
            ),
            erasure_epoch=int(row["erasure_epoch"]),
            phase=ClosurePhase(row["phase"]),
            boundary=safe_json_loads(row["boundary_json"]) or {},
            verification=(
                safe_json_loads(row["verification_json"])
                if row["verification_json"]
                else {}
            ),
            created_us=int(row["created_us"]),
            updated_us=int(row["updated_us"]),
            error=row["error"],
        )

    def status(self, run_id: str, *, conn: Any = None) -> ClosureRun:
        """Current durable state of a run."""
        require_id(run_id, "run_id")
        if conn is not None:
            return self._to_run(self._run_row(conn, run_id))
        with self._store.read() as rconn:
            return self._to_run(self._run_row(rconn, run_id))

    def frontier(
        self, run_id: str, *, conn: Any = None
    ) -> list[dict[str, Any]]:
        """All frontier rows for a run (plain dicts; diagnostic surface)."""
        require_id(run_id, "run_id")

        def _load(c: sqlite3.Connection) -> list[dict[str, Any]]:
            self._run_row(c, run_id)
            return repos_v4.query(
                c, "closure_frontier", {"run_id": run_id}, order="cursor"
            )

        if conn is not None:
            return _load(conn)
        with self._store.read() as rconn:
            return _load(rconn)

    # ------------------------------------------------------------------
    # begin — atomic suppression + durable run (V4-38.01/02/03)
    # ------------------------------------------------------------------

    def begin(
        self,
        roots: Iterable[Any],
        scope_id: str,
        *,
        conn: Optional[sqlite3.Connection] = None,
        purge_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> ClosureRun:
        """Tombstone ``roots``, advance the erasure epoch, queue the closure.

        Idempotent on ``run_id``: replaying a begin for a live run returns
        the run unchanged (a failed run is resumed so job redelivery heals).
        """
        require_id(scope_id, "scope_id")
        if conn is None:
            with self._store.tx() as owned:
                return self._begin_tx(owned, roots, scope_id, purge_id, run_id)
        return self._begin_tx(conn, roots, scope_id, purge_id, run_id)

    def _begin_tx(
        self,
        conn: sqlite3.Connection,
        roots: Iterable[Any],
        scope_id: str,
        purge_id: Optional[str],
        run_id: Optional[str],
    ) -> ClosureRun:
        norm: list[tuple[str, str, Optional[int]]] = []
        seen: set[tuple[str, str, Optional[int]]] = set()
        for item in roots or ():
            ref = _norm_ref(item)
            if ref not in seen:
                seen.add(ref)
                norm.append(ref)
        if not norm:
            raise VerbatimError(
                ErrorCode.VALIDATION, "closure needs at least one root"
            )
        run_id = run_id or f"crun-{new_id()}"
        existing = repos_v4.get(conn, "closure_runs", {"run_id": run_id})
        if existing is not None:
            prior = safe_json_loads(existing["roots_json"]) or []
            if [list(r) for r in norm] != [list(r) for r in prior]:
                raise VerbatimError(
                    ErrorCode.INTEGRITY,
                    f"closure run {run_id!r} exists with different roots",
                )
            if existing["phase"] == ClosurePhase.FAILED.value:
                # Redelivery on a failed run resumes it (V4-38.11).
                return self._resume_tx(conn, run_id)
            return self._to_run(existing)

        scopes: set[str] = set()
        for ref in norm:
            if not _del._object_exists(conn, ref, scope_id):
                # Absent objects are legal roots: purge_derived runs after
                # the parent was erased; the tombstone intent is what counts.
                continue
            owner = _del._object_scope(conn, ref)
            if owner is None or owner != scope_id:
                # Missing and foreign objects share one response (§10.05).
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
                    f"{ref[0]} {ref[1]!r} not found",
                )
            scopes.add(owner)
        if len(scopes) > 1:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "deletion closure is a single-scope operation",
            )

        # Suppression registry: reuse the v2 tombstone tables so every read
        # surface that honors purges observes the roots immediately.
        engine_owned = purge_id is None
        if purge_id is None:
            purge_id = f"purge-{run_id}"
            digest = self._store.hmac(
                json_dumps(
                    {"scope_id": scope_id,
                     "roots": sorted([k, i, r] for k, i, r in norm)}
                ).encode("utf-8")
            )
            conn.execute(
                "INSERT INTO purges"
                "(purge_id, selection_digest, scope_id, state, requested_us)"
                " VALUES (?, ?, ?, 'purging', ?)",
                (purge_id, digest, scope_id, wall_us()),
            )
        else:
            prow = _row(
                conn.execute(
                    "SELECT state, scope_id FROM purges WHERE purge_id = ?",
                    (purge_id,),
                )
            )
            if prow is None:
                raise VerbatimError(
                    ErrorCode.NOT_FOUND_OR_FORBIDDEN, "purge not found"
                )
            if prow["state"] == "previewed":
                conn.execute(
                    "UPDATE purges SET state = 'purging' WHERE purge_id = ?",
                    (purge_id,),
                )
            elif prow["state"] == "completed":
                raise VerbatimError(
                    ErrorCode.STALE_PROPOSAL,
                    "purge already completed; cannot open a closure run",
                )
        for k, i, r in norm:
            conn.execute(
                "INSERT OR IGNORE INTO purge_targets"
                "(purge_id, object_kind, object_id) VALUES (?, ?, ?)",
                (purge_id, k, _target_oid(k, i, r)),
            )
        # Kind-aware tombstones for roots (recorded_until/availability).
        _apply_tombstones(conn, [(k, i) for k, i, _r in norm])

        # V4-38.01: the erasure epoch advances *before* cleanup is
        # acknowledged — in-flight producers fence on it (V4-38.07 is the
        # producer side, coordinated by the apply path).
        epoch = self._bump_epoch(conn)

        now = wall_us()
        boundary = {
            "purge_id": purge_id,
            "engine_owned": engine_owned,
            "planned": False,
            "plan_cursor": 0,
            "roots": [list(r) for r in norm],
            "pending": len(norm),
        }
        repos_v4.insert(
            conn,
            "closure_runs",
            {
                "run_id": run_id,
                "scope_id": scope_id,
                "roots_json": json_dumps([list(r) for r in norm]),
                "erasure_epoch": epoch,
                "phase": ClosurePhase.CLEANING.value,
                "boundary_json": json_dumps(boundary),
                "created_us": now,
                "updated_us": now,
            },
        )
        cursor = 0
        for k, i, r in norm:
            cursor += 1
            repos_v4.insert(
                conn,
                "closure_frontier",
                {
                    "run_id": run_id,
                    "cursor": cursor,
                    "object_kind": k,
                    "object_id": i,
                    "revision": r,
                    "action": "enumerate",
                    "state": "pending",
                    "detail_json": json_dumps({"offsets": {}}),
                },
            )
        EventsRepo(self._store).append(
            conn,
            scope_id,
            "closure_began",
            PURGE_ACTOR,
            {"run_id": run_id, "purge_id": purge_id, "roots": len(norm),
             "erasure_epoch": epoch},
            "policy-1",
        )
        return self._to_run(self._run_row(conn, run_id))

    def _bump_epoch(self, conn: sqlite3.Connection) -> int:
        get = getattr(self._store, "_meta_get", None)
        set_ = getattr(self._store, "_meta_set", None)
        if not (callable(get) and callable(set_)):
            return 0
        current = get(conn, "erasure_epoch") or 0
        nxt = int(current) + 1
        set_(conn, "erasure_epoch", nxt)
        return nxt

    # ------------------------------------------------------------------
    # ancestry/epoch check — descendant exclusion before enumeration (V4-38.02)
    # ------------------------------------------------------------------

    def is_excluded(
        self,
        conn: sqlite3.Connection,
        ref: Any,
        *,
        max_depth: int = 64,
    ) -> bool:
        """True when ``ref`` is suppressed directly or descends from a
        tombstoned/suppressing object — the current ancestry check.

        Suppressed set = purge_targets under any suppressing purge plus
        every member of a non-completed closure run. Ancestors are walked
        through derivations + dependency_edges + registered side references
        (bounded, cycle-safe). Runs in ``completed`` phase contribute their
        erase members through the purge registry (still suppressing).
        """
        target = _norm_ref(ref)
        suppressed: set[tuple[str, str]] = {
            (r[0], r[1])
            for r in conn.execute(
                "SELECT pt.object_kind, pt.object_id FROM purge_targets pt"
                " JOIN purges p ON p.purge_id = pt.purge_id"
                " WHERE p.state IN ('suppressed','purging','completed')",
            ).fetchall()
        }
        active_members: set[tuple[str, str]] = set()
        if _has_table(conn, "closure_frontier"):
            for r in conn.execute(
                "SELECT DISTINCT f.object_kind, f.object_id"
                " FROM closure_frontier f"
                " JOIN closure_runs cr ON cr.run_id = f.run_id"
                " WHERE cr.phase IN ('cleaning','verifying','failed_cleanup')",
            ).fetchall():
                active_members.add((r[0], r[1]))

        def suppressed_or_active(ref3: tuple[str, str, Optional[int]]) -> bool:
            k, i, _r = ref3
            return (k, i) in suppressed or (k, i) in active_members

        if suppressed_or_active(target):
            return True
        seen: set[tuple[str, str, Optional[int]]] = {target}
        queue: list[tuple[tuple[str, str, Optional[int]], int]] = [(target, 0)]
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for parent in self._parents(conn, node):
                if parent in seen:
                    continue
                seen.add(parent)
                if suppressed_or_active(parent):
                    return True
                queue.append((parent, depth + 1))
        return False

    # ------------------------------------------------------------------
    # step — one bounded batch of frontier work (V4-38.03/04)
    # ------------------------------------------------------------------

    def step(
        self,
        run_id: str,
        budget: int = DEFAULT_BUDGET,
        *,
        conn: Optional[sqlite3.Connection] = None,
    ) -> ClosureRun:
        """Process up to ``budget`` frontier row-visits; return the run.

        A ``cleaning`` run that still has pending work stays ``cleaning`` —
        never a terminal phase with unprocessed edges (V4-38.04). On a
        ``verifying`` run the verification pass executes. ``failed_cleanup``
        requires ``resume()``; ``completed`` is a no-op.
        """
        require_id(run_id, "run_id")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise VerbatimError(
                ErrorCode.VALIDATION, "budget must be an int >= 1"
            )
        if conn is None:
            with self._store.tx() as owned:
                return self._to_run(self._step_tx(owned, run_id, budget))
        return self._to_run(self._step_tx(conn, run_id, budget))

    def _step_tx(
        self, conn: sqlite3.Connection, run_id: str, budget: int
    ) -> dict[str, Any]:
        row = self._run_row(conn, run_id)
        phase = row["phase"]
        if phase == ClosurePhase.COMPLETED.value:
            return row
        if phase == ClosurePhase.FAILED.value:
            raise VerbatimError(
                ErrorCode.INVALID_TRANSITION,
                f"closure run {run_id} is failed_cleanup; call resume()",
            )
        if phase == ClosurePhase.VERIFYING.value:
            self._verify_tx(conn, row)
            return self._run_row(conn, run_id)

        ctx = _Ctx()
        boundary = safe_json_loads(row["boundary_json"]) or {}
        members = self._load_members(conn, row)
        visits = 0
        failed_exc: Optional[BaseException] = None
        failed_item: Optional[dict[str, Any]] = None

        def _dispatch(item: dict[str, Any]) -> None:
            action = item["action"]
            conn.execute("SAVEPOINT closure_row")
            try:
                if action == "settle":
                    self._settle(conn, row, item, members, ctx)
                elif action == "finalize":
                    self._finalize(conn, row, item, members, ctx)
                else:  # pragma: no cover - defensive
                    raise VerbatimError(
                        ErrorCode.INTEGRITY,
                        f"unknown frontier action {action!r}",
                    )
                conn.execute("RELEASE SAVEPOINT closure_row")
            except Exception as exc:  # noqa: BLE001 - durability fence
                conn.execute("ROLLBACK TO SAVEPOINT closure_row")
                conn.execute("RELEASE SAVEPOINT closure_row")
                raise _RowFailed(item, exc) from exc

        while visits < budget:
            # Ordering invariant: enumerate drains before planning; planning
            # completes before any settle runs; settle mints finalize rows at
            # cursors above every settle, so edge stripping is always last.
            page = repos_v4.query(
                conn,
                "closure_frontier",
                {"run_id": run_id, "state": "pending", "action": "enumerate"},
                order="cursor",
                limit=min(_ENUM_ROWS, budget - visits),
            )
            if page:
                try:
                    self._enumerate_page(conn, row, page, members, ctx)
                except _RowFailed as exc:
                    failed_exc = exc.cause or exc
                    failed_item = exc.item
                    break
                visits += len(page)
                continue
            if not boundary.get("planned"):
                # Charge the budget for enumerate rows SCANNED, not rows
                # minted — scanning is the real per-member work and a
                # step's cost must stay proportional to ``budget``
                # (V4-38.04).
                scanned = self._plan_batch(
                    conn, row, members, boundary, ctx,
                    min(_PLAN_PAGE, budget - visits),
                )
                visits += max(1, scanned)
                continue
            pending = repos_v4.query(
                conn,
                "closure_frontier",
                {"run_id": run_id, "state": "pending"},
                order="cursor",
                limit=1,
            )
            if not pending:
                break
            item = pending[0]
            visits += 1
            try:
                _dispatch(item)
            except _RowFailed as exc:
                failed_exc = exc.cause or exc
                failed_item = exc.item
                break
        if failed_item is not None:
            # V4-38.11: the failure is durable; suppression committed in
            # earlier units is retained; the run stays resumable. The error
            # is merged into the row's existing detail — key lists and
            # pagination offsets must survive so resume re-runs the SAME
            # unit of work, not a truncated one.
            fdetail = safe_json_loads(failed_item["detail_json"] or "{}") or {}
            fdetail["error"] = (
                f"{type(failed_exc).__name__}: {failed_exc}"
            )
            repos_v4.update(
                conn,
                "closure_frontier",
                {"state": "failed", "detail_json": json_dumps(fdetail)},
                {"run_id": run_id, "cursor": failed_item["cursor"]},
            )
            boundary["pending"] = self._pending_count(conn, run_id)
            repos_v4.update(
                conn,
                "closure_runs",
                {
                    "phase": ClosurePhase.FAILED.value,
                    "boundary_json": json_dumps(boundary),
                    "error": (
                        f"frontier cursor {failed_item['cursor']} "
                        f"({failed_item['action']}) failed: "
                        f"{type(failed_exc).__name__}: {failed_exc}"
                    ),
                    "updated_us": wall_us(),
                },
                {"run_id": run_id},
            )
            return self._run_row(conn, run_id)

        pending_n = self._pending_count(conn, run_id)
        boundary["pending"] = pending_n
        if pending_n == 0:
            if not boundary.get("planned"):
                # Barrier: enumeration is complete; mint settle rows for
                # every member (paginated by enumerate cursor).
                self._plan_batch(conn, row, members, boundary, ctx, budget)
                pending_n = self._pending_count(conn, run_id)
                boundary["pending"] = pending_n
            if pending_n == 0 and boundary.get("planned"):
                # Frontier drained → durable phase transition, then verify.
                repos_v4.update(
                    conn,
                    "closure_runs",
                    {
                        "phase": ClosurePhase.VERIFYING.value,
                        "boundary_json": json_dumps(boundary),
                        "updated_us": wall_us(),
                    },
                    {"run_id": run_id},
                )
                self._verify_tx(conn, self._run_row(conn, run_id))
                return self._run_row(conn, run_id)
        repos_v4.update(
            conn,
            "closure_runs",
            {"boundary_json": json_dumps(boundary), "updated_us": wall_us()},
            {"run_id": run_id},
        )
        return self._run_row(conn, run_id)

    def _pending_count(self, conn: sqlite3.Connection, run_id: str) -> int:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM closure_frontier"
                " WHERE run_id = ? AND state = 'pending'",
                (run_id,),
            ).fetchone()[0]
        )

    # ------------------------------------------------------------------
    # members: the durable member set reconstructed from the frontier
    # ------------------------------------------------------------------

    @staticmethod
    def _member_default() -> dict[str, Any]:
        return {"class": None, "planned": False, "outside": None}

    def _load_members(
        self, conn: sqlite3.Connection, run_row: dict[str, Any]
    ) -> dict[tuple[str, str, Optional[int]], dict[str, Any]]:
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]] = {}
        for frow in conn.execute(
            "SELECT object_kind, object_id, revision, action, detail_json"
            " FROM closure_frontier WHERE run_id = ? ORDER BY cursor",
            (run_row["run_id"],),
        ).fetchall():
            action = frow[3]
            if action == "enumerate":
                members.setdefault(
                    (frow[0], frow[1], frow[2]), self._member_default()
                )
            elif action == "settle":
                detail = safe_json_loads(frow[4] or "{}") or {}
                keys = detail.get("keys") or [
                    (frow[0], frow[1], frow[2])
                ]
                classes = detail.get("classes") or []
                for idx, kk in enumerate(keys):
                    key = (kk[0], kk[1], kk[2])
                    m = members.setdefault(key, self._member_default())
                    m["planned"] = True
                    if idx < len(classes):
                        cls = classes[idx]
                        reason = None
                        if isinstance(cls, (list, tuple)):
                            cls, reason = cls[0], cls[1]
                        if cls:
                            m["class"] = cls
                            m["outside"] = reason
        # Roots are purged from the start (V4-38.01).
        for ref in safe_json_loads(run_row["roots_json"]) or []:
            key = (ref[0], ref[1], ref[2] if len(ref) > 2 else None)
            m = members.setdefault(key, self._member_default())
            m["class"] = _C_ERASE
            m["root"] = True
        return members

    @staticmethod
    def _by_id(
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]]
    ) -> dict[tuple[str, str], list[tuple[str, str, Optional[int]]]]:
        out: dict[tuple[str, str], list[tuple[str, str, Optional[int]]]] = {}
        for key in members:
            out.setdefault((key[0], key[1]), []).append(key)
        return out

    def _match(
        self,
        by_id: dict[tuple[str, str], list[tuple[str, str, Optional[int]]]],
        ref: tuple[str, str, Optional[int]],
    ) -> list[tuple[str, str, Optional[int]]]:
        """Member keys matching ``ref`` — None revision is the wildcard."""
        cands = by_id.get((ref[0], ref[1]), ())
        if ref[2] is None:
            return list(cands)
        return [c for c in cands if c[2] is None or c[2] == ref[2]]

    # ------------------------------------------------------------------
    # graph sources
    # ------------------------------------------------------------------

    def _edge_children(
        self,
        conn: sqlite3.Connection,
        ref: tuple[str, str, Optional[int]],
        table: str,
        last: Optional[list[Any]],
        limit: int,
    ) -> list[tuple[str, str, int]]:
        """Keyset page of child edges for ``ref`` from one edge table."""
        sql = (
            f"SELECT child_kind, child_id, child_revision FROM {table}"
            " WHERE parent_kind = ? AND parent_id = ?"
        )
        params: list[Any] = [ref[0], ref[1]]
        if ref[2] is not None:
            sql += " AND parent_revision = ?"
            params.append(ref[2])
        if last is not None:
            sql += (
                " AND (child_kind, child_id, child_revision) > (?, ?, ?)"
            )
            params.extend(last)
        sql += (
            " ORDER BY child_kind, child_id, child_revision LIMIT ?"
        )
        params.append(limit)
        return [
            (r[0], r[1], int(r[2]))
            for r in conn.execute(sql, params).fetchall()
        ]

    def _children(
        self, conn: sqlite3.Connection, ref: tuple[str, str, Optional[int]]
    ) -> dict[str, list[tuple[str, str, Optional[int]]]]:
        """All children of ``ref`` across edge tables + side walkers."""
        out: dict[str, list[tuple[str, str, Optional[int]]]] = {}
        for table in _EDGE_TABLES:
            if not _has_table(conn, table):
                out[table] = []
                continue
            sql = (
                f"SELECT child_kind, child_id, child_revision FROM {table}"
                " WHERE parent_kind = ? AND parent_id = ?"
            )
            params: list[Any] = [ref[0], ref[1]]
            if ref[2] is not None:
                sql += " AND parent_revision = ?"
                params.append(ref[2])
            sql += " ORDER BY child_kind, child_id, child_revision"
            out[table] = [
                (r[0], r[1], int(r[2]))
                for r in conn.execute(sql, params).fetchall()
            ]
        for name, walker in _SIDE_WALKERS.items():
            try:
                out[name] = list(walker["children"](conn, ref))
            except Exception:
                out[name] = []  # a broken walker degrades, never blocks
        return out

    def _parents(
        self, conn: sqlite3.Connection, ref: tuple[str, str, Optional[int]]
    ) -> list[tuple[str, str, Optional[int]]]:
        """All recorded parents of ``ref`` across every registered source."""
        out: set[tuple[str, str, Optional[int]]] = set()
        for table in _EDGE_TABLES:
            if not _has_table(conn, table):
                continue
            sql = (
                f"SELECT parent_kind, parent_id, parent_revision"
                f" FROM {table} WHERE child_kind = ? AND child_id = ?"
            )
            params: list[Any] = [ref[0], ref[1]]
            if ref[2] is not None:
                sql += " AND child_revision = ?"
                params.append(ref[2])
            for r in conn.execute(sql, params).fetchall():
                out.add((r[0], r[1], int(r[2])))
        for walker in _SIDE_WALKERS.values():
            try:
                for p in walker["parents"](conn, ref) or ():
                    out.add(tuple(p))
            except Exception:
                continue
        return sorted(out, key=_ref_sort)

    def _member_parents(
        self,
        conn: sqlite3.Connection,
        key: tuple[str, str, Optional[int]],
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        ctx: _Ctx,
    ) -> list[tuple[str, str, Optional[int]]]:
        """Parents of a member — bulk-prefetched edge tables + built-in
        walkers, plus per-member calls for third-party walkers."""
        if ctx.parents is None:
            ctx.parents = self._parents_bulk(conn, members, ctx)
        out: set[tuple[str, str, Optional[int]]] = set(
            ctx.parents.get(key, ())
        )
        for name, walker in _SIDE_WALKERS.items():
            if name in _BULK_WALKERS:
                continue
            try:
                for p in walker["parents"](conn, key) or ():
                    out.add(tuple(p))
            except Exception:
                continue
        return sorted(out, key=_ref_sort)

    def _parents_bulk(
        self,
        conn: sqlite3.Connection,
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        ctx: _Ctx,
    ) -> dict[tuple[str, str, Optional[int]],
             list[tuple[str, str, Optional[int]]]]:
        """Parents of every member in grouped IN-chunk queries.

        Semantics identical to per-member ``_parents`` — a ``None`` revision
        is the wildcard (no revision condition), built-in side walkers
        contribute their direction, custom walkers stay per-member.
        """
        out: dict[tuple[str, str, Optional[int]],
                  set[tuple[str, str, Optional[int]]]] = {}
        by_kind_rev: dict[tuple[str, Optional[int]], list[str]] = {}
        for k, i, r in members:
            by_kind_rev.setdefault((k, r), []).append(i)
        for table in _EDGE_TABLES:
            if not ctx.has_table(conn, table):
                continue
            for (kind, rev), ids in by_kind_rev.items():
                for chunk in _chunks(sorted(set(ids))):
                    sql = (
                        f"SELECT child_id, parent_kind, parent_id,"
                        f" parent_revision FROM {table}"
                        " WHERE child_kind = ?"
                        f" AND child_id IN ({_placeholders(len(chunk))})"
                    )
                    params: list[Any] = [kind, *chunk]
                    if rev is not None:
                        sql += " AND child_revision = ?"
                        params.append(rev)
                    for r in conn.execute(sql, params).fetchall():
                        out.setdefault((kind, r[0], rev), set()).add(
                            (r[1], r[2], int(r[3]))
                        )
        # Built-in walkers, bulked (claim_evidence: claim→span parents;
        # observation_evidence: observation→cited-object parents).
        if ctx.has_table(conn, "claim_evidence") and (
            "claim_evidence" in _SIDE_WALKERS
        ):
            claim_ids = sorted(
                {i for k, i, _r in members if k == "claim"}
            )
            by_cid: dict[str, list[tuple[int, str]]] = {}
            for chunk in _chunks(claim_ids):
                for r in conn.execute(
                    "SELECT claim_id, revision, span_id"
                    " FROM claim_evidence"
                    f" WHERE claim_id IN ({_placeholders(len(chunk))})",
                    chunk,
                ).fetchall():
                    by_cid.setdefault(r[0], []).append((int(r[1]), r[2]))
            for key in members:
                if key[0] != "claim":
                    continue
                for rev, sid in by_cid.get(key[1], ()):
                    if key[2] is not None and rev != key[2]:
                        continue
                    out.setdefault(key, set()).add(("span", sid, None))
        if ctx.has_table(conn, "observation_evidence") and (
            "observation_evidence" in _SIDE_WALKERS
        ):
            obs_ids = sorted(
                {i for k, i, _r in members if k == "observation"}
            )
            by_oid: dict[str, list[tuple[int, str, str, Any]]] = {}
            for chunk in _chunks(obs_ids):
                for r in conn.execute(
                    "SELECT observation_id, revision, object_kind,"
                    " object_id, object_revision FROM observation_evidence"
                    f" WHERE observation_id IN"
                    f" ({_placeholders(len(chunk))})",
                    chunk,
                ).fetchall():
                    by_oid.setdefault(r[0], []).append(
                        (int(r[1]), r[2], r[3], r[4])
                    )
            for key in members:
                if key[0] != "observation":
                    continue
                for rev, pk, pi, pr in by_oid.get(key[1], ()):
                    if key[2] is not None and rev != key[2]:
                        continue
                    out.setdefault(key, set()).add(
                        (pk, pi, int(pr) if pr is not None else None)
                    )
        return {k: sorted(v, key=_ref_sort) for k, v in out.items()}

    # ------------------------------------------------------------------
    # scope resolution — batched per kind, per-key fallback for phantoms
    # ------------------------------------------------------------------

    def _scope_bulk(
        self,
        conn: sqlite3.Connection,
        kind: str,
        ids: list[str],
        ctx: _Ctx,
    ) -> dict[str, Optional[str]]:
        """oid → scope for one kind, in IN-chunk queries.

        Mirrors ``deletion._object_scope``: the kind's simple table first,
        the special cases (source_revision→sources, span→join,
        environment→state table, working→sets/items), and nothing else —
        rows absent from the map fall back to the per-key resolver.
        """
        out: dict[str, Optional[str]] = {}
        spec = _del._SCOPE_SIMPLE.get(kind)
        if spec is not None and ctx.has_table(conn, spec[0]):
            table, col = spec
            for chunk in _chunks(ids):
                for r in conn.execute(
                    f"SELECT {col}, scope_id FROM {table}"
                    f" WHERE {col} IN ({_placeholders(len(chunk))})",
                    chunk,
                ).fetchall():
                    out[r[0]] = r[1]
            return out
        if kind == "source_revision" and ctx.has_table(conn, "sources"):
            for chunk in _chunks(ids):
                for r in conn.execute(
                    "SELECT source_id, scope_id FROM sources"
                    f" WHERE source_id IN ({_placeholders(len(chunk))})",
                    chunk,
                ).fetchall():
                    out[r[0]] = r[1]
        elif kind == "span" and ctx.has_table(conn, "spans"):
            for chunk in _chunks(ids):
                for r in conn.execute(
                    "SELECT sp.span_id, so.scope_id FROM spans sp"
                    " JOIN sources so ON so.source_id = sp.source_id"
                    f" WHERE sp.span_id IN ({_placeholders(len(chunk))})",
                    chunk,
                ).fetchall():
                    out[r[0]] = r[1]
        elif kind == "environment" and ctx.has_table(
            conn, "environment_state"
        ):
            for chunk in _chunks(ids):
                for r in conn.execute(
                    "SELECT key, scope_id FROM environment_state"
                    f" WHERE key IN ({_placeholders(len(chunk))})",
                    chunk,
                ).fetchall():
                    out.setdefault(r[0], r[1])
        elif kind == "working":
            if ctx.has_table(conn, "working_sets"):
                for chunk in _chunks(ids):
                    for r in conn.execute(
                        "SELECT set_id, scope_id FROM working_sets"
                        f" WHERE set_id IN ({_placeholders(len(chunk))})",
                        chunk,
                    ).fetchall():
                        out.setdefault(r[0], r[1])
            if ctx.has_table(conn, "working_set_items"):
                for chunk in _chunks(ids):
                    for r in conn.execute(
                        "SELECT wi.item_id, ws.scope_id"
                        " FROM working_set_items wi"
                        " JOIN working_sets ws ON ws.set_id = wi.set_id"
                        f" WHERE wi.item_id IN"
                        f" ({_placeholders(len(chunk))})",
                        chunk,
                    ).fetchall():
                        out.setdefault(r[0], r[1])
        return out

    def _scope_of(
        self,
        conn: sqlite3.Connection,
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        key: tuple[str, str, Optional[int]],
        ctx: _Ctx,
    ) -> Optional[str]:
        kind, oid = key[0], key[1]
        smap = ctx.scopes.get(kind)
        if smap is None:
            ids = sorted({i for k, i, _r in members if k == kind})
            smap = self._scope_bulk(conn, kind, ids, ctx)
            ctx.scopes[kind] = smap
        if oid in smap:
            return smap[oid]
        # Phantom/absent rows resolve through the graph itself.
        return _del._object_scope(conn, key)

    # ------------------------------------------------------------------
    # enumerate — bounded keyset-paginated expansion (V4-38.03/04)
    # ------------------------------------------------------------------

    def _enumerate_page(
        self,
        conn: sqlite3.Connection,
        run_row: dict[str, Any],
        page: list[dict[str, Any]],
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        ctx: _Ctx,
    ) -> None:
        """One savepoint around a page of ``enumerate`` visits.

        Bulk path: children of never-visited rows are prefetched in grouped
        IN-chunk queries, admits and state updates are flushed with
        ``executemany``. On failure the page rolls back and each row is
        retried under its own savepoint so the first genuinely bad row —
        and only that row — is marked failed.
        """
        run_id = run_row["run_id"]
        admitted: list[tuple[tuple[str, str, Optional[int]], tuple]] = []
        conn.execute("SAVEPOINT closure_page")
        try:
            self._children_prefetch(conn, page, ctx)
            updates: list[tuple[str, str, str, int]] = []
            for item in page:
                state, detail = self._enumerate_visit(
                    conn, run_row, item, members, ctx, admitted
                )
                updates.append(
                    (state, json_dumps(detail), run_id, item["cursor"])
                )
            conn.executemany(
                "UPDATE closure_frontier SET state = ?, detail_json = ?"
                " WHERE run_id = ? AND cursor = ?",
                updates,
            )
            if admitted:
                conn.executemany(
                    "INSERT INTO closure_frontier"
                    "(run_id, cursor, object_kind, object_id, revision,"
                    " action, state, detail_json)"
                    " VALUES (?, ?, ?, ?, ?, 'enumerate', 'pending', ?)",
                    [rowv for _key, rowv in admitted],
                )
            conn.execute("RELEASE SAVEPOINT closure_page")
            return
        except Exception:  # noqa: BLE001 - fall back to row isolation
            conn.execute("ROLLBACK TO SAVEPOINT closure_page")
            conn.execute("RELEASE SAVEPOINT closure_page")
        for key, _rowv in admitted:
            members.pop(key, None)
        for item in page:
            row_sink: list[tuple] = []
            conn.execute("SAVEPOINT closure_row")
            try:
                state, detail = self._enumerate_visit(
                    conn, run_row, item, members, ctx, row_sink
                )
                if row_sink:
                    conn.executemany(
                        "INSERT INTO closure_frontier"
                        "(run_id, cursor, object_kind, object_id, revision,"
                        " action, state, detail_json)"
                        " VALUES (?, ?, ?, ?, ?, 'enumerate', 'pending', ?)",
                        [rowv for _key, rowv in row_sink],
                    )
                conn.execute(
                    "UPDATE closure_frontier SET state = ?, detail_json = ?"
                    " WHERE run_id = ? AND cursor = ?",
                    (state, json_dumps(detail), run_id, item["cursor"]),
                )
                conn.execute("RELEASE SAVEPOINT closure_row")
            except Exception as exc:  # noqa: BLE001 - durability fence
                conn.execute("ROLLBACK TO SAVEPOINT closure_row")
                conn.execute("RELEASE SAVEPOINT closure_row")
                for key, _rowv in row_sink:
                    members.pop(key, None)
                raise _RowFailed(item, exc) from exc

    def _children_prefetch(
        self,
        conn: sqlite3.Connection,
        page: list[dict[str, Any]],
        ctx: _Ctx,
    ) -> None:
        """Bulk-load complete child lists for never-visited page rows.

        Only rows with empty durable offsets are prefetched — their
        ``state.last`` is None, so ``_enumerate_visit`` consumes the cached
        lists from position 0. Resumed rows keep the per-row keyset path.
        """
        fresh: list[tuple[str, str, Optional[int]]] = []
        for item in page:
            detail = safe_json_loads(item["detail_json"] or "{}") or {}
            if not (detail.get("offsets") or {}):
                fresh.append(
                    (item["object_kind"], item["object_id"], item["revision"])
                )
        if not fresh:
            return
        groups: dict[tuple[str, Optional[int]], list[str]] = {}
        for k, i, r in fresh:
            groups.setdefault((k, r), []).append(i)
        for table in _EDGE_TABLES:
            if not ctx.has_table(conn, table):
                continue
            for (kind, rev), ids in groups.items():
                for chunk in _chunks(sorted(set(ids))):
                    sql = (
                        f"SELECT parent_id, child_kind, child_id,"
                        f" child_revision FROM {table}"
                        " WHERE parent_kind = ?"
                        f" AND parent_id IN ({_placeholders(len(chunk))})"
                    )
                    params: list[Any] = [kind, *chunk]
                    if rev is not None:
                        sql += " AND parent_revision = ?"
                        params.append(rev)
                    sql += " ORDER BY child_kind, child_id, child_revision"
                    for r in conn.execute(sql, params).fetchall():
                        ctx.children.setdefault(
                            ((kind, r[0], rev), table), []
                        ).append((r[1], r[2], int(r[3])))
        if "observation_evidence" in _SIDE_WALKERS and ctx.has_table(
            conn, "observation_evidence"
        ):
            for (kind, rev), ids in groups.items():
                for chunk in _chunks(sorted(set(ids))):
                    sql = (
                        "SELECT object_id, observation_id, revision"
                        " FROM observation_evidence WHERE object_kind = ?"
                        f" AND object_id IN ({_placeholders(len(chunk))})"
                    )
                    params = [kind, *chunk]
                    if rev is not None:
                        sql += " AND object_revision = ?"
                        params.append(rev)
                    sql += " ORDER BY observation_id, revision"
                    for r in conn.execute(sql, params).fetchall():
                        ctx.children.setdefault(
                            ((kind, r[0], rev), "observation_evidence"), []
                        ).append(("observation", r[1], int(r[2])))
        if "claim_evidence" in _SIDE_WALKERS and ctx.has_table(
            conn, "claim_evidence"
        ):
            span_ids = sorted({i for k, i, _r in fresh if k == "span"})
            if span_ids:
                by_span: dict[str, list[tuple[str, str, int]]] = {}
                for chunk in _chunks(span_ids):
                    for r in conn.execute(
                        "SELECT span_id, claim_id, revision"
                        " FROM claim_evidence"
                        f" WHERE span_id IN ({_placeholders(len(chunk))})"
                        " ORDER BY claim_id, revision",
                        chunk,
                    ).fetchall():
                        by_span.setdefault(r[0], []).append(
                            ("claim", r[1], int(r[2]))
                        )
                for key in fresh:
                    if key[0] == "span" and key[1] in by_span:
                        ctx.children[(key, "claim_evidence")] = by_span[
                            key[1]
                        ]

    def _enumerate_visit(
        self,
        conn: sqlite3.Connection,
        run_row: dict[str, Any],
        item: dict[str, Any],
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        ctx: _Ctx,
        sink: Optional[list] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Page one ``enumerate`` row forward; returns (state, detail).

        The durable ``detail.offsets`` carry the per-source keyset
        position, so a visit bounded by ``_ENUM_PAGE`` resumes exactly
        where it stopped after a crash.
        """
        ref = (item["object_kind"], item["object_id"], item["revision"])
        detail = safe_json_loads(item["detail_json"] or "{}") or {}
        offsets: dict[str, dict[str, Any]] = detail.setdefault("offsets", {})
        remaining = _ENUM_PAGE
        done_all = True
        for source in (*_EDGE_TABLES, *_SIDE_WALKERS.keys()):
            state = offsets.setdefault(source, {"last": None, "done": False})
            if state["done"] or remaining <= 0:
                if not state["done"]:
                    done_all = False
                continue
            if source in _EDGE_TABLES:
                if not ctx.has_table(conn, source):
                    state["done"] = True
                    continue
                cached = (
                    ctx.children.get((ref, source))
                    if state["last"] is None
                    else None
                )
                if cached is not None:
                    page = cached[: remaining + 1]
                else:
                    page = self._edge_children(
                        conn, ref, source, state["last"], remaining + 1
                    )
            else:
                cached = ctx.children.get((ref, source))
                if cached is None or state["last"]:
                    try:
                        all_kids = (
                            _SIDE_WALKERS[source]["children"](conn, ref)
                            or []
                        )
                    except Exception:
                        all_kids = []
                else:
                    all_kids = cached
                start = int(state["last"][0]) if state["last"] else 0
                page = [
                    tuple(c)
                    for c in all_kids[start : start + remaining + 1]
                ]
            more = len(page) > remaining
            page = page[:remaining]
            if page:
                if source in _EDGE_TABLES:
                    state["last"] = list(page[-1])
                else:
                    state["last"] = [
                        (state["last"][0] if state["last"] else 0)
                        + len(page)
                    ]
                self._admit_children(
                    conn, run_row, page, members, ctx, sink
                )
                remaining -= len(page)
            if more:
                done_all = False
                break  # continue this source next visit
            state["done"] = True
        detail["offsets"] = offsets
        return ("done" if done_all else "pending"), detail

    def _admit_children(
        self,
        conn: sqlite3.Connection,
        run_row: dict[str, Any],
        children: list[tuple[str, str, Optional[int]]],
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        ctx: _Ctx,
        sink: Optional[list],
    ) -> None:
        """Persist newly discovered members — the durable frontier.

        With a ``sink`` the rows are appended for a page-level
        ``executemany``; without one they insert immediately (the row-level
        fallback path keeps its own savepoint semantics).
        """
        for child in children:
            key = (child[0], child[1], child[2])
            if key in members:
                continue
            members[key] = self._member_default()
            cursor = ctx.alloc_cursor(conn, run_row["run_id"])
            rowv = (
                run_row["run_id"],
                cursor,
                child[0],
                child[1],
                child[2],
                json_dumps({"offsets": {}}),
            )
            if sink is not None:
                sink.append((key, rowv))
            else:
                conn.execute(
                    "INSERT INTO closure_frontier"
                    "(run_id, cursor, object_kind, object_id, revision,"
                    " action, state, detail_json)"
                    " VALUES (?, ?, ?, ?, ?, 'enumerate', 'pending', ?)",
                    rowv,
                )

    # ------------------------------------------------------------------
    # planning barrier — settle batch rows for every member (V4-38.03)
    # ------------------------------------------------------------------

    def _plan_batch(
        self,
        conn: sqlite3.Connection,
        run_row: dict[str, Any],
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        boundary: dict[str, Any],
        ctx: _Ctx,
        limit: int,
    ) -> int:
        """Mint ``settle`` batch rows, paginated by enumerate position.

        Runs only once enumeration has fully drained, so classification sees
        the final member set — no provisional decisions, no revisits.
        Returns the number of enumerate rows scanned (>=0) so the caller's
        step budget reflects the real work performed.
        """
        cursor = int(boundary.get("plan_cursor") or 0)
        rows = conn.execute(
            "SELECT object_kind, object_id, revision FROM closure_frontier"
            " WHERE run_id = ? AND action = 'enumerate'"
            " ORDER BY cursor LIMIT ? OFFSET ?",
            (run_row["run_id"], limit, cursor),
        ).fetchall()
        minted = 0
        batch: list[tuple[str, str, Optional[int]]] = []

        def _flush() -> None:
            nonlocal minted
            if not batch:
                return
            cur = ctx.alloc_cursor(conn, run_row["run_id"])
            first = batch[0]
            conn.execute(
                "INSERT INTO closure_frontier"
                "(run_id, cursor, object_kind, object_id, revision,"
                " action, state, detail_json)"
                " VALUES (?, ?, ?, ?, ?, 'settle', 'pending', ?)",
                (
                    run_row["run_id"],
                    cur,
                    first[0],
                    first[1],
                    first[2],
                    json_dumps({"keys": [list(k) for k in batch]}),
                ),
            )
            for k in batch:
                members[k]["planned"] = True
            minted += 1
            batch.clear()

        for erow in rows:
            key = (erow[0], erow[1], erow[2])
            if members[key].get("planned"):
                continue
            batch.append(key)
            if len(batch) >= _SETTLE_BATCH:
                _flush()
        _flush()
        boundary["plan_cursor"] = cursor + len(rows)
        boundary["planned"] = len(rows) < limit
        return len(rows)

    # ------------------------------------------------------------------
    # settle — classify over the final member set, apply suppression (V4-38.06)
    # ------------------------------------------------------------------

    def _settle(
        self,
        conn: sqlite3.Connection,
        run_row: dict[str, Any],
        item: dict[str, Any],
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        ctx: _Ctx,
    ) -> None:
        sid = run_row["scope_id"]
        run_id = run_row["run_id"]
        purge_id = (safe_json_loads(run_row["boundary_json"]) or {}).get(
            "purge_id"
        )
        detail = safe_json_loads(item["detail_json"] or "{}") or {}
        keys = [
            (k[0], k[1], k[2])
            for k in (detail.get("keys") or [])
        ] or [(item["object_kind"], item["object_id"], item["revision"])]

        classes: list[list[Any]] = []
        suppress_keys: list[tuple[str, str, Optional[int]]] = []
        reval_keys: list[tuple[str, str, Optional[int]]] = []
        erase_keys: list[tuple[str, str, Optional[int]]] = []
        for key in keys:
            info = members[key]
            cls = info.get("class")
            if cls is None:
                cls = self._resolve(conn, run_row, members, key, ctx)
                info = members[key]
            classes.append([cls, info.get("outside")])
            if cls == _C_ERASE:
                erase_keys.append(key)
            elif cls == _C_SUPPRESS:
                suppress_keys.append(key)
            elif cls == _C_REVALIDATE:
                reval_keys.append(key)

        if suppress_keys or reval_keys:
            seq = ctx.alloc_seq(conn)
            if suppress_keys:
                self._suppress_batch(
                    conn, suppress_keys, sid, seq, purge_id, ctx
                )
            if reval_keys:
                self._revalidate_batch(conn, reval_keys, sid, seq, ctx)

        # Erased members queue their physical cleanup at cursors above
        # every settle row — edges stay readable until classification ends.
        for i in range(0, len(erase_keys), _FIN_BATCH):
            self._queue_finalize(
                conn, run_id, erase_keys[i : i + _FIN_BATCH], ctx
            )

        detail["classes"] = classes
        repos_v4.update(
            conn,
            "closure_frontier",
            {"state": "done", "detail_json": json_dumps(detail)},
            {"run_id": run_id, "cursor": item["cursor"]},
        )

    def _queue_finalize(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        keys: list[tuple[str, str, Optional[int]]],
        ctx: _Ctx,
    ) -> int:
        """Mint one ``finalize`` batch row; returns its cursor.

        Always at a cursor above every extant settle row — finalize work
        (edge stripping) runs strictly after classification consumed the
        edges it removes.
        """
        if not keys:
            return -1
        cursor = ctx.alloc_cursor(conn, run_id)
        conn.execute(
            "INSERT INTO closure_frontier"
            "(run_id, cursor, object_kind, object_id, revision,"
            " action, state, detail_json)"
            " VALUES (?, ?, ?, ?, ?, 'finalize', 'pending', ?)",
            (
                run_id,
                cursor,
                keys[0][0],
                keys[0][1],
                keys[0][2],
                json_dumps({"keys": [list(k) for k in keys]}),
            ),
        )
        return cursor

    def _suppress_batch(
        self,
        conn: sqlite3.Connection,
        keys: list[tuple[str, str, Optional[int]]],
        sid: str,
        seq: int,
        purge_id: Optional[str],
        ctx: _Ctx,
    ) -> None:
        """Suppression markers for a key batch — same marker semantics as
        ``deletion._apply_suppression``: the kind's recorded-time column is
        authoritative when it exists and the row is present; otherwise the
        quarantine registry carries the suppression state (§34.03)."""
        if purge_id:
            conn.executemany(
                "INSERT OR IGNORE INTO purge_targets"
                "(purge_id, object_kind, object_id) VALUES (?, ?, ?)",
                [
                    (purge_id, k, _target_oid(k, i, r))
                    for k, i, r in keys
                ],
            )
        claims: list[tuple[str, str, Optional[int]]] = []
        quarantine_keys: list[tuple[str, str, Optional[int]]] = []
        by_kind: dict[str, list[str]] = {}
        for key in keys:
            if key[0] == "claim":
                claims.append(key)
                continue
            by_kind.setdefault(key[0], []).append(key[1])
        for key in claims:
            _del._apply_suppression(conn, key, sid, seq)
        for kind, ids in by_kind.items():
            dated = _del._DATED_TABLES.get(kind)
            if dated is not None and ctx.has_table(conn, dated[0]):
                present: set[str] = set()
                for chunk in _chunks(sorted(set(ids))):
                    for r in conn.execute(
                        f"SELECT {dated[1]} FROM {dated[0]}"
                        f" WHERE {dated[1]} IN"
                        f" ({_placeholders(len(chunk))})",
                        chunk,
                    ).fetchall():
                        present.add(r[0])
                for chunk in _chunks(sorted(set(ids))):
                    conn.execute(
                        f"UPDATE {dated[0]} SET recorded_until = ?"
                        f" WHERE {dated[1]} IN"
                        f" ({_placeholders(len(chunk))})"
                        " AND recorded_until IS NULL",
                        [seq, *chunk],
                    )
                quarantine_keys.extend(
                    (kind, i, None) for i in ids if i not in present
                )
            elif kind == "artifact" and ctx.has_table(conn, "artifacts"):
                present = set()
                for chunk in _chunks(sorted(set(ids))):
                    for r in conn.execute(
                        "SELECT artifact_id FROM artifacts"
                        f" WHERE artifact_id IN"
                        f" ({_placeholders(len(chunk))})",
                        chunk,
                    ).fetchall():
                        present.add(r[0])
                for chunk in _chunks(sorted(set(ids))):
                    conn.execute(
                        "UPDATE artifacts SET availability = 'suppressed'"
                        f" WHERE artifact_id IN"
                        f" ({_placeholders(len(chunk))})"
                        " AND availability = 'available'",
                        chunk,
                    )
                quarantine_keys.extend(
                    (kind, i, None) for i in ids if i not in present
                )
            elif (
                kind in _del._OBJECTS_DISPOSITION_KINDS
                and ctx.has_table(conn, "objects")
            ):
                # Registry-carried kinds (branch, derived_view): the
                # objects row's disposition is the suppression marker —
                # same effect as deletion._apply_suppression's clause.
                present = set()
                for chunk in _chunks(sorted(set(ids))):
                    for r in conn.execute(
                        "SELECT object_id FROM objects"
                        f" WHERE kind = ? AND object_id IN"
                        f" ({_placeholders(len(chunk))})",
                        [kind, *chunk],
                    ).fetchall():
                        present.add(r[0])
                for chunk in _chunks(sorted(set(ids))):
                    conn.execute(
                        "UPDATE objects SET disposition = 'held'"
                        f" WHERE kind = ? AND object_id IN"
                        f" ({_placeholders(len(chunk))})"
                        " AND disposition = 'active'",
                        [kind, *chunk],
                    )
                quarantine_keys.extend(
                    (kind, i, None) for i in ids if i not in present
                )
            else:
                quarantine_keys.extend(
                    (k, i, r) for k, i, r in keys if k == kind
                )
        if quarantine_keys and ctx.has_table(conn, "quarantine"):
            conn.executemany(
                "INSERT OR REPLACE INTO quarantine"
                "(object_kind, object_id, revision, scope_id,"
                " reason_codes_json, findings_json, state, opened_event,"
                " decision_json)"
                " VALUES (?, ?, ?, ?, '[]', '[]', 'suppressed', ?, '{}')",
                [
                    (k, i, r if r is not None else 1, sid, seq)
                    for k, i, r in quarantine_keys
                ],
            )

    def _revalidate_batch(
        self,
        conn: sqlite3.Connection,
        keys: list[tuple[str, str, Optional[int]]],
        sid: str,
        seq: int,
        ctx: _Ctx,
    ) -> None:
        """Freshness 'revalidate_after' flags — mirrors
        ``deletion._apply_revalidation``."""
        if ctx.has_table(conn, "freshness"):
            conn.executemany(
                "INSERT OR REPLACE INTO freshness"
                "(scope_id, object_kind, object_id, revision, class)"
                " VALUES (?, ?, ?, ?, 'revalidate_after')",
                [
                    (sid, k, i, r if r is not None else 1)
                    for k, i, r in keys
                ],
            )
        obs_ids = sorted({i for k, i, _r in keys if k == "observation"})
        for chunk in _chunks(obs_ids):
            conn.execute(
                "UPDATE observations"
                " SET stale_since_seq = COALESCE(stale_since_seq, ?)"
                f" WHERE observation_id IN ({_placeholders(len(chunk))})",
                [seq, *chunk],
            )

    # ------------------------------------------------------------------
    # finalize — edges, secondary refs, propagations, ledger, jobs (V4-38.05)
    # ------------------------------------------------------------------

    def _finalize(
        self,
        conn: sqlite3.Connection,
        run_row: dict[str, Any],
        item: dict[str, Any],
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        ctx: _Ctx,
    ) -> None:
        sid = run_row["scope_id"]
        boundary = safe_json_loads(run_row["boundary_json"]) or {}
        purge_id = boundary.get("purge_id")
        epoch = int(run_row["erasure_epoch"])
        run_id = run_row["run_id"]
        detail = safe_json_loads(item["detail_json"] or "{}") or {}
        keys = [
            (k[0], k[1], k[2]) for k in (detail.get("keys") or [])
        ] or [(item["object_kind"], item["object_id"], item["revision"])]

        seq = ctx.alloc_seq(conn)
        gone = self._gone_set(conn, keys, sid, ctx)

        # Suppression marker FIRST (failure can never restore visibility),
        # then the kind's physical erasure for members not already gone —
        # a member tombstoned by an *earlier* operation (a purge-path
        # skeleton) is left as-is while a live member is deleted.
        self._suppress_batch(conn, keys, sid, seq, purge_id, ctx)
        live = [key for key in keys if key not in gone]
        self._delete_batch(conn, live, sid, purge_id or run_id, ctx)

        # Edges naming the purged objects on either side, in every edge
        # table — the pointer into erased material is what goes (V3-36.02).
        edges = self._edges_out_batch(conn, keys, ctx)
        # Secondary references on surviving rows (member tables, FTS-side
        # pointers, influence rows) — same coverage as v3 closure.
        self._strip_batch(conn, keys, ctx)

        # Propagated copies: revoke locally, disclose remotely (V3-36.05).
        propagated = self._propagate_batch(conn, keys, seq, ctx)

        # Opaque erasure tombstones — restore fencing (V3-36.03).
        ledger_n = self._ledger_batch(
            conn, keys, sid, purge_id, epoch, ctx
        )

        jobs_cancelled = self._cancel_jobs_batch(conn, sid, keys, ctx)
        detail.update(
            {
                "propagated": propagated,
                "ledger": ledger_n,
                "jobs_cancelled": jobs_cancelled,
                "edges_removed": edges,
            }
        )
        repos_v4.update(
            conn,
            "closure_frontier",
            {"state": "done", "detail_json": json_dumps(detail)},
            {"run_id": run_id, "cursor": item["cursor"]},
        )

    def _gone_set(
        self,
        conn: sqlite3.Connection,
        keys: list[tuple[str, str, Optional[int]]],
        sid: str,
        ctx: _Ctx,
    ) -> set[tuple[str, str, Optional[int]]]:
        """Members already gone per their kind's erasure semantics —
        batched twin of ``_member_gone``: row absent per the kind's object
        table, or (skeleton kinds only) ``recorded_until`` already set.
        Kinds with richer gone semantics (claim/span/source*/artifact)
        resolve per key through ``_member_gone`` unchanged."""
        gone: set[tuple[str, str, Optional[int]]] = set()
        by_kind: dict[str, list[tuple[str, str, Optional[int]]]] = {}
        for key in keys:
            by_kind.setdefault(key[0], []).append(key)
        for kind, group in by_kind.items():
            spec = _EXISTS_TABLES.get(kind)
            if kind == "environment":
                if ctx.has_table(conn, "environment_state"):
                    present = set()
                    ids = sorted({i for _k, i, _r in group})
                    for chunk in _chunks(ids):
                        for r in conn.execute(
                            "SELECT key FROM environment_state"
                            " WHERE scope_id = ?"
                            f" AND key IN ({_placeholders(len(chunk))})",
                            [sid, *chunk],
                        ).fetchall():
                            present.add(r[0])
                    gone.update(k for k in group if k[1] not in present)
                else:
                    gone.update(group)
            elif kind == "working":
                present = set()
                ids = sorted({i for _k, i, _r in group})
                if ctx.has_table(conn, "working_sets"):
                    for chunk in _chunks(ids):
                        for r in conn.execute(
                            "SELECT set_id FROM working_sets"
                            f" WHERE set_id IN"
                            f" ({_placeholders(len(chunk))})",
                            chunk,
                        ).fetchall():
                            present.add(r[0])
                if ctx.has_table(conn, "working_set_items"):
                    for chunk in _chunks(ids):
                        for r in conn.execute(
                            "SELECT item_id FROM working_set_items"
                            f" WHERE item_id IN"
                            f" ({_placeholders(len(chunk))})",
                            chunk,
                        ).fetchall():
                            present.add(r[0])
                gone.update(k for k in group if k[1] not in present)
            elif spec is not None and ctx.has_table(conn, spec[0]):
                table, col = spec
                present = set()
                ids = sorted({i for _k, i, _r in group})
                for chunk in _chunks(ids):
                    for r in conn.execute(
                        f"SELECT {col} FROM {table}"
                        f" WHERE {col} IN ({_placeholders(len(chunk))})",
                        chunk,
                    ).fetchall():
                        present.add(r[0])
                if kind in _SKELETON_KINDS:
                    dated = _del._DATED_TABLES.get(kind)
                    tombstoned: set[str] = set()
                    if dated is not None:
                        for chunk in _chunks(ids):
                            for r in conn.execute(
                                f"SELECT {dated[1]} FROM {dated[0]}"
                                f" WHERE {dated[1]} IN"
                                f" ({_placeholders(len(chunk))})"
                                " AND recorded_until IS NOT NULL",
                                chunk,
                            ).fetchall():
                                tombstoned.add(r[0])
                    gone.update(
                        k for k in group
                        if k[1] not in present or k[1] in tombstoned
                    )
                else:
                    gone.update(k for k in group if k[1] not in present)
            else:
                for key in group:
                    if self._member_gone(conn, key, sid):
                        gone.add(key)
        return gone

    def _member_gone(
        self,
        conn: sqlite3.Connection,
        ref: tuple[str, str, Optional[int]],
        sid: str,
    ) -> bool:
        """True when the object is already erased *or* already tombstoned
        by the outer purge — only kinds whose erasure semantics keep a
        bitemporal skeleton (episode/procedure) count a closed
        ``recorded_until`` as gone; every other erased kind deletes its
        row, so verify never passes on a suppressed-but-live member."""
        if _del._is_gone(conn, ref, sid):
            return True
        kind, oid, _r = ref
        if kind not in _SKELETON_KINDS:
            return False
        dated = _del._DATED_TABLES.get(kind)
        if dated is not None and _has_table(conn, dated[0]):
            row = conn.execute(
                f"SELECT recorded_until FROM {dated[0]} WHERE {dated[1]} = ?",
                (oid,),
            ).fetchone()
            if row is not None and row[0] is not None:
                return True
        return False

    def _delete_batch(
        self,
        conn: sqlite3.Connection,
        keys: list[tuple[str, str, Optional[int]]],
        sid: str,
        purge_id: str,
        ctx: _Ctx,
    ) -> None:
        """Physical erasure per kind — flat-table kinds delete in grouped
        IN-chunks; kinds with richer semantics (byte scrubbing, lifecycle
        machine, member cascades) keep ``_del._delete_object`` per key."""
        machine: Optional[LifecycleMachine] = None
        by_kind: dict[str, list[tuple[str, str, Optional[int]]]] = {}
        for key in keys:
            by_kind.setdefault(key[0], []).append(key)
        for kind, group in by_kind.items():
            spec = _BATCH_DELETES.get(kind)
            if spec is not None:
                ids = sorted({i for _k, i, _r in group})
                for table, col in spec:
                    if not ctx.has_table(conn, table):
                        continue
                    for chunk in _chunks(ids):
                        conn.execute(
                            f"DELETE FROM {table}"
                            f" WHERE {col} IN ({_placeholders(len(chunk))})",
                            chunk,
                        )
            elif kind == "environment":
                if ctx.has_table(conn, "environment_state"):
                    ids = sorted({i for _k, i, _r in group})
                    for chunk in _chunks(ids):
                        conn.execute(
                            "DELETE FROM environment_state"
                            " WHERE scope_id = ?"
                            f" AND key IN ({_placeholders(len(chunk))})",
                            [sid, *chunk],
                        )
            elif kind == "working":
                ids = sorted({i for _k, i, _r in group})
                if ctx.has_table(conn, "working_set_items"):
                    for chunk in _chunks(ids):
                        conn.execute(
                            "DELETE FROM working_set_items"
                            f" WHERE set_id IN"
                            f" ({_placeholders(len(chunk))})"
                            f" OR item_id IN ({_placeholders(len(chunk))})",
                            [*chunk, *chunk],
                        )
                if ctx.has_table(conn, "working_sets"):
                    for chunk in _chunks(ids):
                        conn.execute(
                            "DELETE FROM working_sets"
                            f" WHERE set_id IN"
                            f" ({_placeholders(len(chunk))})",
                            chunk,
                        )
            else:
                if machine is None:
                    machine = LifecycleMachine(self._store)
                for key in group:
                    _del._delete_object(
                        conn, self._store, machine, key, purge_id, sid
                    )

    def _edges_out_batch(
        self,
        conn: sqlite3.Connection,
        keys: list[tuple[str, str, Optional[int]]],
        ctx: _Ctx,
    ) -> int:
        """Delete every edge naming a purged object on either side —
        grouped by (kind, revision); a None revision keeps the original
        wildcard (all revisions of the object id)."""
        edges = 0
        groups: dict[tuple[str, Optional[int]], set[str]] = {}
        by_kind: dict[str, set[str]] = {}
        for k, i, r in keys:
            groups.setdefault((k, r), set()).add(i)
            by_kind.setdefault(k, set()).add(i)
        for table in _EDGE_TABLES:
            if not ctx.has_table(conn, table):
                continue
            for (kind, rev), ids in groups.items():
                for chunk in _chunks(sorted(ids)):
                    ph = _placeholders(len(chunk))
                    if rev is None:
                        cur = conn.execute(
                            f"DELETE FROM {table}"
                            f" WHERE (child_kind = ? AND child_id IN ({ph}))"
                            f"    OR (parent_kind = ?"
                            f"        AND parent_id IN ({ph}))",
                            [kind, *chunk, kind, *chunk],
                        )
                    else:
                        cur = conn.execute(
                            f"DELETE FROM {table} WHERE"
                            f" (child_kind = ? AND child_id IN ({ph})"
                            f"  AND child_revision = ?)"
                            f" OR (parent_kind = ? AND parent_id IN ({ph})"
                            f"  AND parent_revision = ?)",
                            [kind, *chunk, rev, kind, *chunk, rev],
                        )
                    edges += cur.rowcount
        # Legacy v2 dependency rows are scrubbed at object granularity.
        if ctx.has_table(conn, "dependency_refs"):
            for kind, ids in by_kind.items():
                for chunk in _chunks(sorted(ids)):
                    ph = _placeholders(len(chunk))
                    conn.execute(
                        "DELETE FROM dependency_refs"
                        f" WHERE (derived_kind = ? AND derived_id IN ({ph}))"
                        f"    OR (input_kind = ? AND input_id IN ({ph}))",
                        [kind, *chunk, kind, *chunk],
                    )
        return edges

    def _strip_batch(
        self,
        conn: sqlite3.Connection,
        keys: list[tuple[str, str, Optional[int]]],
        ctx: _Ctx,
    ) -> None:
        """Secondary references on surviving rows — batched twin of
        ``deletion._strip_references``: member tables at (kind, id)
        granularity, observation_evidence and influence revision-aware,
        plus the kind-specific pointer columns."""
        groups: dict[tuple[str, Optional[int]], set[str]] = {}
        by_kind: dict[str, set[str]] = {}
        for k, i, r in keys:
            groups.setdefault((k, r), set()).add(i)
            by_kind.setdefault(k, set()).add(i)

        if ctx.has_table(conn, "observation_evidence"):
            for (kind, rev), ids in groups.items():
                for chunk in _chunks(sorted(ids)):
                    ph = _placeholders(len(chunk))
                    if rev is None:
                        conn.execute(
                            "DELETE FROM observation_evidence"
                            f" WHERE object_kind = ? AND object_id IN ({ph})",
                            [kind, *chunk],
                        )
                    else:
                        conn.execute(
                            "DELETE FROM observation_evidence"
                            f" WHERE object_kind = ? AND object_id IN ({ph})"
                            " AND object_revision = ?",
                            [kind, *chunk, rev],
                        )
        for table in ("episode_members", "family_members",
                      "capsule_members", "decision_inputs",
                      "artifact_links"):
            if not ctx.has_table(conn, table):
                continue
            for kind, ids in by_kind.items():
                for chunk in _chunks(sorted(ids)):
                    conn.execute(
                        f"DELETE FROM {table}"
                        f" WHERE object_kind = ?"
                        f" AND object_id IN ({_placeholders(len(chunk))})",
                        [kind, *chunk],
                    )
        span_ids = sorted(by_kind.get("span") or ())
        if span_ids and ctx.has_table(conn, "claim_evidence"):
            for chunk in _chunks(span_ids):
                conn.execute(
                    f"DELETE FROM claim_evidence WHERE span_id IN"
                    f" ({_placeholders(len(chunk))})",
                    chunk,
                )
            if ctx.has_table(conn, "procedure_steps"):
                for chunk in _chunks(span_ids):
                    conn.execute(
                        "UPDATE procedure_steps SET span_id = NULL"
                        f" WHERE span_id IN ({_placeholders(len(chunk))})",
                        chunk,
                    )
        anchor_ids = sorted(by_kind.get("state_anchor") or ())
        if anchor_ids and ctx.has_table(conn, "transition_anchors"):
            for chunk in _chunks(anchor_ids):
                conn.execute(
                    "DELETE FROM transition_anchors"
                    f" WHERE anchor_id IN ({_placeholders(len(chunk))})",
                    chunk,
                )
        envelope_ids = sorted(by_kind.get("envelope") or ())
        if envelope_ids and ctx.has_table(conn, "step_observations"):
            for chunk in _chunks(envelope_ids):
                conn.execute(
                    "DELETE FROM step_observations"
                    f" WHERE envelope_id IN ({_placeholders(len(chunk))})",
                    chunk,
                )
        if ctx.has_table(conn, "profile_entry_support"):
            # Profile support pointers to a purged object go with it —
            # the entry fails closed at read until refresh() tombstones.
            for (kind, rev), ids in groups.items():
                for chunk in _chunks(sorted(ids)):
                    ph = _placeholders(len(chunk))
                    if rev is None:
                        conn.execute(
                            "DELETE FROM profile_entry_support"
                            " WHERE support_kind = ?"
                            f" AND support_id IN ({ph})",
                            [kind, *chunk],
                        )
                    else:
                        conn.execute(
                            "DELETE FROM profile_entry_support"
                            " WHERE support_kind = ?"
                            f" AND support_id IN ({ph})"
                            " AND support_revision = ?",
                            [kind, *chunk, rev],
                        )
        if ctx.has_table(conn, "influence"):
            for (kind, rev), ids in groups.items():
                for chunk in _chunks(sorted(ids)):
                    ph = _placeholders(len(chunk))
                    if rev is None:
                        conn.execute(
                            "UPDATE influence SET redacted = 1"
                            f" WHERE object_kind = ? AND object_id IN ({ph})",
                            [kind, *chunk],
                        )
                    else:
                        conn.execute(
                            "UPDATE influence SET redacted = 1"
                            f" WHERE object_kind = ? AND object_id IN ({ph})"
                            " AND revision = ?",
                            [kind, *chunk, rev],
                        )

    def _propagate_batch(
        self,
        conn: sqlite3.Connection,
        keys: list[tuple[str, str, Optional[int]]],
        seq: int,
        ctx: _Ctx,
    ) -> list[dict[str, Any]]:
        """Revoke propagations for purged objects and disclose the remote
        copies — batched twin of the per-member propagation sweep."""
        if not ctx.has_table(conn, "propagations"):
            return []
        propagated: list[dict[str, Any]] = []
        by_kind: dict[str, set[str]] = {}
        rev_of: dict[tuple[str, str], Optional[int]] = {}
        for k, i, r in keys:
            by_kind.setdefault(k, set()).add(i)
            rev_of.setdefault((k, i), r)
        for kind, ids in by_kind.items():
            for chunk in _chunks(sorted(ids)):
                ph = _placeholders(len(chunk))
                for prow in _rows(
                    conn.execute(
                        "SELECT propagation_id, recipient_id, object_id"
                        " FROM propagations"
                        " WHERE object_kind = ?"
                        f" AND object_id IN ({ph})"
                        " AND revoked_seq IS NULL",
                        [kind, *chunk],
                    )
                ):
                    propagated.append(
                        {
                            "ref": [
                                kind,
                                prow["object_id"],
                                rev_of.get((kind, prow["object_id"])),
                            ],
                            "propagation_id": prow["propagation_id"],
                            "recipient_id": prow["recipient_id"],
                        }
                    )
                conn.execute(
                    "UPDATE propagations SET revoked_seq = ?"
                    " WHERE object_kind = ?"
                    f" AND object_id IN ({ph})"
                    " AND revoked_seq IS NULL",
                    [seq, kind, *chunk],
                )
        return propagated

    def _ledger_batch(
        self,
        conn: sqlite3.Connection,
        keys: list[tuple[str, str, Optional[int]]],
        sid: str,
        purge_id: Optional[str],
        epoch: int,
        ctx: _Ctx,
    ) -> int:
        """Erasure-ledger tombstones — same row shape as
        ``ErasureRepo.record``; existing digests are skipped so a replayed
        batch stays idempotent."""
        if not ctx.has_table(conn, "erasure_ledger"):
            return 0
        erasure = ErasureRepo(self._store)
        by_kind: dict[str, dict[str, list[tuple[str, str, Optional[int]]]]] = {}
        for k, i, r in keys:
            lid = (
                f"{i}:{r}"
                if (k == "source_revision" and r is not None)
                else i
            )
            by_kind.setdefault(k, {}).setdefault(lid, []).append((k, i, r))
        written = 0
        for kind, lids in by_kind.items():
            digests = {
                lid: erasure.digest_for(kind, lid) for lid in lids
            }
            for chunk in _chunks(sorted(lids)):
                have = {
                    r[0]
                    for r in conn.execute(
                        "SELECT object_digest FROM erasure_ledger"
                        " WHERE scope_id = ? AND object_kind = ?"
                        f" AND object_digest IN"
                        f" ({_placeholders(len(chunk))})",
                        [sid, kind, *[digests[lid] for lid in chunk]],
                    ).fetchall()
                }
                rows = [
                    (
                        new_id(),
                        sid,
                        kind,
                        digests[lid],
                        purge_id,
                        ctx.alloc_seq(conn),
                        epoch,
                    )
                    for lid in chunk
                    if digests[lid] not in have
                ]
                if rows:
                    conn.executemany(
                        "INSERT INTO erasure_ledger"
                        "(erasure_id, scope_id, object_kind, object_digest,"
                        " purge_id, erased_event, erasure_epoch)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
                    written += len(rows)
        return written

    def _cancel_jobs_batch(
        self,
        conn: sqlite3.Connection,
        sid: str,
        keys: list[tuple[str, str, Optional[int]]],
        ctx: _Ctx,
    ) -> int:
        """Cancel pending/leased jobs referencing purged objects —
        batched twin of ``purge._cancel_jobs``: one scan of the scope's
        cancellable jobs, matched by ``input_refs_json`` substring, with
        the erasure-lane exemption (``purge``/``purge_derived``/
        ``purge_vault`` — a closure must never cancel the job driving it)."""
        if not ctx.has_table(conn, "jobs"):
            return 0
        if ctx.jobs is None:
            ctx.jobs = _rows(
                conn.execute(
                    "SELECT job_id, input_refs_json FROM jobs"
                    " WHERE state IN ('queued','retry_wait','leased')"
                    "   AND kind NOT IN"
                    "       ('purge','purge_derived','purge_vault')"
                    "   AND scope_id = ?",
                    (sid,),
                )
            )
        ids = {i for _k, i, _r in keys}
        hits: list[str] = []
        keep: list[dict[str, Any]] = []
        for j in ctx.jobs:
            payload = j["input_refs_json"] or ""
            if any(oid in payload for oid in ids):
                hits.append(j["job_id"])
            else:
                keep.append(j)
        ctx.jobs = keep
        for chunk in _chunks(hits):
            conn.execute(
                "UPDATE jobs SET state = 'cancelled'"
                f" WHERE job_id IN ({_placeholders(len(chunk))})",
                chunk,
            )
        return len(hits)

    # ------------------------------------------------------------------
    # classification — the fixpoint, evaluated lazily per member
    # ------------------------------------------------------------------

    def _resolve(
        self,
        conn: sqlite3.Connection,
        run_row: dict[str, Any],
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        start: tuple[str, str, Optional[int]],
        ctx: _Ctx,
    ) -> str:
        """DFS-resolve ``start``'s class; memoized + persisted on settle rows.

        Equivalent to ``plan_closure``: erase iff every recorded parent is
        purged; suppress on any purged (or cycle-tainted) parent; revalidate
        on invalidated ancestry; outside for foreign scope / unresolvable
        kind. Decisions write back into ``members`` so peers never recompute.
        """
        sid = run_row["scope_id"]
        if ctx.by_id is None:
            ctx.by_id = self._by_id(members)
        by_id = ctx.by_id
        memo: dict[tuple[str, str, Optional[int]], str] = {}
        gray: set[tuple[str, str, Optional[int]]] = set()
        stack: list[tuple[tuple[str, str, Optional[int]], bool]] = [
            (start, False)
        ]
        while stack:
            key, expanded = stack.pop()
            if not expanded:
                if key in memo:
                    continue
                known = members[key].get("class")
                if known is not None:
                    memo[key] = known
                    continue
                outside = self._outside_reason(conn, sid, members, key, ctx)
                if outside is not None:
                    members[key]["class"] = _C_OUTSIDE
                    members[key]["outside"] = outside
                    memo[key] = _C_OUTSIDE
                    continue
                gray.add(key)
                stack.append((key, True))
                for p in self._member_parents(conn, key, members, ctx):
                    for m in self._match(by_id, p):
                        if m in memo or m in gray:
                            continue
                        if members[m].get("class") is not None:
                            memo[m] = members[m]["class"]
                            continue
                        stack.append((m, False))
            else:
                gray.discard(key)
                cls = self._combine(
                    conn, sid, key, members, by_id, memo, gray, ctx
                )
                memo[key] = cls
                members[key]["class"] = cls
        return memo[start]

    def _combine(
        self,
        conn: sqlite3.Connection,
        sid: str,
        key: tuple[str, str, Optional[int]],
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        by_id: dict[tuple[str, str], list[tuple[str, str, Optional[int]]]],
        memo: dict[tuple[str, str, Optional[int]], str],
        gray: set[tuple[str, str, Optional[int]]],
        ctx: _Ctx,
    ) -> str:
        parents = self._member_parents(conn, key, members, ctx)
        if not parents:
            return _C_UNAFFECTED if not members[key].get("root") else _C_ERASE
        if members[key].get("root"):
            return _C_ERASE
        purged = tainted = invalidated = 0
        for p in parents:
            matches = self._match(by_id, p)
            if not matches:
                continue  # non-member parent → surviving ancestry
            classes = set()
            for m in matches:
                if m in memo:
                    classes.add(memo[m])
                elif m in gray:
                    classes.add(_C_CYCLE)
                else:
                    classes.add(members[m].get("class") or _C_CYCLE)
            if classes == {_C_ERASE}:
                purged += 1
            elif classes & {_C_ERASE, _C_CYCLE}:
                tainted += 1
            elif classes & {_C_SUPPRESS, _C_REVALIDATE}:
                invalidated += 1
            # outside/unaffected member-parents count as survivors.
        if purged == len(parents):
            return _C_ERASE
        if purged or tainted:
            return _C_SUPPRESS
        if invalidated:
            return _C_REVALIDATE
        return _C_UNAFFECTED

    def _outside_reason(
        self,
        conn: sqlite3.Connection,
        sid: str,
        members: dict[tuple[str, str, Optional[int]], dict[str, Any]],
        ref: tuple[str, str, Optional[int]],
        ctx: _Ctx,
    ) -> Optional[str]:
        if ref[0] not in _del._RESOLVABLE_KINDS:
            return "unresolvable_kind"
        scope = self._scope_of(conn, members, ref, ctx)
        if scope is not None and scope != sid:
            return "foreign_scope"
        return None

    # ------------------------------------------------------------------
    # verify — the honest closure check (V4-38.03/04/08)
    # ------------------------------------------------------------------

    def verify(
        self, run_id: str, *, conn: Optional[sqlite3.Connection] = None
    ) -> dict[str, Any]:
        """Run the closure check. On a verifying run this is the terminal
        transition; otherwise it reports current posture without mutating."""
        require_id(run_id, "run_id")
        if conn is None:
            with self._store.tx() as owned:
                row = self._run_row(owned, run_id)
                if row["phase"] == ClosurePhase.VERIFYING.value:
                    self._verify_tx(owned, row)
                    row = self._run_row(owned, run_id)
                    return self._to_run(row).verification
                return self._verify_tx(owned, row, readonly=True)
        row = self._run_row(conn, run_id)
        if row["phase"] == ClosurePhase.VERIFYING.value:
            self._verify_tx(conn, row)
            return self._to_run(self._run_row(conn, run_id)).verification
        return self._verify_tx(conn, row, readonly=True)

    def _verify_tx(
        self,
        conn: sqlite3.Connection,
        run_row: dict[str, Any],
        readonly: bool = False,
    ) -> dict[str, Any]:
        sid = run_row["scope_id"]
        run_id = run_row["run_id"]
        ctx = _Ctx()
        members = self._load_members(conn, run_row)
        orphans: list[dict[str, Any]] = []

        unfinished = conn.execute(
            "SELECT COUNT(*) FROM closure_frontier"
            " WHERE run_id = ? AND state IN ('pending','failed')",
            (run_id,),
        ).fetchone()[0]
        if unfinished:
            orphans.append(
                {"reason": "frontier_unfinished", "count": int(unfinished)}
            )

        purged = {
            k for k, m in members.items() if m.get("class") == _C_ERASE
        }
        suppressed = {
            k for k, m in members.items() if m.get("class") == _C_SUPPRESS
        }
        revalidate = {
            k for k, m in members.items() if m.get("class") == _C_REVALIDATE
        }
        outside = {
            k for k, m in members.items() if m.get("class") == _C_OUTSIDE
        }

        # 1. Every purged object is gone per its kind's erasure semantics.
        gone = self._gone_set(conn, sorted(purged, key=_ref_sort), sid, ctx)
        for ref in sorted(purged - gone, key=_ref_sort):
            orphans.append({"ref": list(ref), "reason": "row_remains"})

        # 2. No edge anywhere still names a purged object on either side.
        orphans.extend(self._edge_orphans(conn, purged, ctx))

        # 3. Secondary references on surviving rows must be stripped.
        orphans.extend(self._ref_orphans(conn, purged, ctx))

        # 4. Planned markers must exist.
        for ref in sorted(suppressed, key=_ref_sort):
            if not self._is_suppressed(conn, run_row, ref, sid):
                orphans.append(
                    {"ref": list(ref), "reason": "suppression_missing"}
                )
        for ref in sorted(revalidate, key=_ref_sort):
            if not _del._is_marked_revalidate(conn, ref, sid):
                orphans.append(
                    {"ref": list(ref), "reason": "revalidation_unmarked"}
                )

        verification = {
            "closed": not orphans,
            "orphans": orphans[:500],
            "counts": {
                "purged": len(purged),
                "suppressed": len(suppressed),
                "revalidate": len(revalidate),
                "outside": len(outside),
                "members": len(members),
            },
        }
        if readonly:
            return verification

        boundary = safe_json_loads(run_row["boundary_json"]) or {}
        boundary.update(self._receipt_counts(conn, run_row, members))
        boundary["pending"] = 0
        if orphans:
            phase = ClosurePhase.FAILED.value
            error = f"closure verification failed: {orphans[:5]!r}"
        else:
            phase = ClosurePhase.COMPLETED.value
            error = None
            if boundary.get("engine_owned") and boundary.get("purge_id"):
                conn.execute(
                    "UPDATE purges SET state = 'completed', completed_us = ?"
                    " WHERE purge_id = ? AND state IN ('suppressed','purging')",
                    (wall_us(), boundary["purge_id"]),
                )
            bump = getattr(self._store, "bump_generation", None)
            if callable(bump):
                boundary["projection_generation"] = int(bump(conn))
            EventsRepo(self._store).append(
                conn,
                sid,
                "closure_completed",
                PURGE_ACTOR,
                {"run_id": run_id, "purge_id": boundary.get("purge_id"),
                 "members": len(members), "purged": len(purged)},
                "policy-1",
            )
        repos_v4.update(
            conn,
            "closure_runs",
            {
                "phase": phase,
                "boundary_json": json_dumps(boundary),
                "verification_json": json_dumps(verification),
                "error": error,
                "updated_us": wall_us(),
            },
            {"run_id": run_id},
        )
        if phase == ClosurePhase.FAILED.value:
            EventsRepo(self._store).append(
                conn,
                sid,
                "closure_failed",
                PURGE_ACTOR,
                {"run_id": run_id, "orphans": orphans[:20]},
                "policy-1",
            )
        return verification

    def _edge_orphans(
        self,
        conn: sqlite3.Connection,
        purged: set[tuple[str, str, Optional[int]]],
        ctx: _Ctx,
    ) -> list[dict[str, Any]]:
        """Surviving edges naming a purged object — grouped by (kind, rev)."""
        if not purged:
            return []
        orphans: list[dict[str, Any]] = []
        groups: dict[tuple[str, Optional[int]], set[str]] = {}
        for k, i, r in purged:
            groups.setdefault((k, r), set()).add(i)

        def _in_purged(k: str, i: str, r: Optional[int]) -> bool:
            return (k, i, r) in purged or (k, i, None) in purged

        counts: dict[tuple[str, tuple[str, str, Optional[int]]], int] = {}
        for table in _EDGE_TABLES:
            if not ctx.has_table(conn, table):
                continue
            for (kind, rev), ids in groups.items():
                for chunk in _chunks(sorted(ids)):
                    ph = _placeholders(len(chunk))
                    if rev is None:
                        sql = (
                            f"SELECT child_kind, child_id, child_revision,"
                            f" parent_kind, parent_id, parent_revision"
                            f" FROM {table}"
                            f" WHERE (child_kind = ? AND child_id IN ({ph}))"
                            f"    OR (parent_kind = ?"
                            f"        AND parent_id IN ({ph}))"
                        )
                        params: list[Any] = [kind, *chunk, kind, *chunk]
                    else:
                        sql = (
                            f"SELECT child_kind, child_id, child_revision,"
                            f" parent_kind, parent_id, parent_revision"
                            f" FROM {table} WHERE"
                            f" (child_kind = ? AND child_id IN ({ph})"
                            f"  AND child_revision = ?)"
                            f" OR (parent_kind = ? AND parent_id IN ({ph})"
                            f"  AND parent_revision = ?)"
                        )
                        params = [
                            kind, *chunk, rev, kind, *chunk, rev,
                        ]
                    for r in conn.execute(sql, params).fetchall():
                        child = (r[0], r[1], int(r[2]))
                        parent = (r[3], r[4], int(r[5]))
                        if _in_purged(*parent):
                            key = (table, parent)
                            counts[key] = counts.get(key, 0) + 1
                        if _in_purged(*child):
                            key = (table, child)
                            counts[key] = counts.get(key, 0) + 1
        for (table, ref), n in sorted(
            counts.items(), key=lambda kv: (kv[0][0], _ref_sort(kv[0][1]))
        ):
            orphans.append(
                {
                    "ref": list(ref),
                    "reason": "edge_remains",
                    "table": table,
                    "count": n,
                }
            )
        return orphans

    def _ref_orphans(
        self,
        conn: sqlite3.Connection,
        purged: set[tuple[str, str, Optional[int]]],
        ctx: _Ctx,
    ) -> list[dict[str, Any]]:
        """Surviving rows in member/link tables that still name a purged
        object — batched twin of ``deletion._reference_orphans``."""
        orphans: list[dict[str, Any]] = []
        purged_ids: dict[str, set[str]] = {}
        for k, i, _r in purged:
            purged_ids.setdefault(k, set()).add(i)
        for table in ("observation_evidence", "episode_members",
                      "family_members", "capsule_members",
                      "decision_inputs", "artifact_links"):
            if not ctx.has_table(conn, table):
                continue
            for kind, ids in purged_ids.items():
                for chunk in _chunks(sorted(ids)):
                    for r in _rows(
                        conn.execute(
                            f"SELECT * FROM {table}"
                            " WHERE object_kind = ?"
                            f" AND object_id IN"
                            f" ({_placeholders(len(chunk))})",
                            [kind, *chunk],
                        )
                    ):
                        rev = r.get("object_revision", r.get("revision"))
                        orphans.append(
                            {
                                "ref": [kind, r["object_id"], rev],
                                "reason": "reference_remains",
                                "table": table,
                            }
                        )
        span_ids = sorted(purged_ids.get("span") or ())
        if span_ids and ctx.has_table(conn, "claim_evidence"):
            for chunk in _chunks(span_ids):
                for r in _rows(
                    conn.execute(
                        "SELECT claim_id, revision, span_id"
                        " FROM claim_evidence"
                        f" WHERE span_id IN ({_placeholders(len(chunk))})",
                        chunk,
                    )
                ):
                    orphans.append(
                        {
                            "ref": ["span", r["span_id"], None],
                            "reason": "reference_remains",
                            "table": "claim_evidence",
                            "detail": {
                                "claim_id": r["claim_id"],
                                "revision": r["revision"],
                            },
                        }
                    )
        if span_ids and ctx.has_table(conn, "procedure_steps"):
            for chunk in _chunks(span_ids):
                for r in conn.execute(
                    "SELECT DISTINCT span_id FROM procedure_steps"
                    f" WHERE span_id IN ({_placeholders(len(chunk))})",
                    chunk,
                ).fetchall():
                    orphans.append(
                        {
                            "ref": ["span", r[0], None],
                            "reason": "reference_remains",
                            "table": "procedure_steps",
                        }
                    )
        anchor_ids = sorted(purged_ids.get("state_anchor") or ())
        if anchor_ids and ctx.has_table(conn, "transition_anchors"):
            for chunk in _chunks(anchor_ids):
                for r in conn.execute(
                    "SELECT DISTINCT anchor_id FROM transition_anchors"
                    f" WHERE anchor_id IN ({_placeholders(len(chunk))})",
                    chunk,
                ).fetchall():
                    orphans.append(
                        {
                            "ref": ["state_anchor", r[0], None],
                            "reason": "reference_remains",
                            "table": "transition_anchors",
                        }
                    )
        if ctx.has_table(conn, "profile_entry_support"):
            # Support pointers name (support_kind, support_id) —
            # (kind, id) granularity, like the member tables.
            for kind, ids in purged_ids.items():
                for chunk in _chunks(sorted(ids)):
                    for r in conn.execute(
                        "SELECT DISTINCT support_id FROM"
                        " profile_entry_support"
                        " WHERE support_kind = ?"
                        f" AND support_id IN ({_placeholders(len(chunk))})",
                        [kind, *chunk],
                    ).fetchall():
                        orphans.append(
                            {
                                "ref": [kind, r[0], None],
                                "reason": "reference_remains",
                                "table": "profile_entry_support",
                            }
                        )
        orphans.sort(key=lambda o: _ref_sort(tuple(o["ref"])))
        return orphans

    def _is_suppressed(
        self,
        conn: sqlite3.Connection,
        run_row: dict[str, Any],
        ref: tuple[str, str, Optional[int]],
        sid: str,
    ) -> bool:
        if _del._is_suppressed(conn, ref, sid):
            return True
        boundary = safe_json_loads(run_row["boundary_json"]) or {}
        purge_id = boundary.get("purge_id")
        if not purge_id:
            return False
        return conn.execute(
            "SELECT 1 FROM purge_targets pt JOIN purges p"
            " ON p.purge_id = pt.purge_id"
            " WHERE pt.object_kind = ? AND pt.object_id = ?"
            " AND p.state IN ('suppressed','purging','completed')",
            (ref[0], _target_oid(ref[0], ref[1], ref[2])),
        ).fetchone() is not None

    # ------------------------------------------------------------------
    # resume — crash recovery / retry (V4-38.11)
    # ------------------------------------------------------------------

    def resume(
        self, run_id: str, *, conn: Optional[sqlite3.Connection] = None
    ) -> ClosureRun:
        """Re-open a failed or interrupted run.

        ``failed`` frontier rows re-pend and the run returns to
        ``cleaning``; verify-failed runs additionally re-pend every
        ``enumerate`` row so late-arriving edges are re-scanned (idempotent
        — member dedup makes re-enumeration a no-op for known children).
        Suppression is never lifted (V4-38.11).
        """
        require_id(run_id, "run_id")
        if conn is None:
            with self._store.tx() as owned:
                return self._resume_tx(owned, run_id)
        return self._resume_tx(conn, run_id)

    def _resume_tx(
        self, conn: sqlite3.Connection, run_id: str
    ) -> ClosureRun:
        row = self._run_row(conn, run_id)
        phase = row["phase"]
        if phase == ClosurePhase.COMPLETED.value:
            return self._to_run(row)
        if phase == ClosurePhase.FAILED.value:
            failed = conn.execute(
                "SELECT COUNT(*) FROM closure_frontier"
                " WHERE run_id = ? AND state = 'failed'",
                (run_id,),
            ).fetchone()[0]
            conn.execute(
                "UPDATE closure_frontier SET state = 'pending'"
                " WHERE run_id = ? AND state = 'failed'",
                (run_id,),
            )
            if not failed:
                # Verify-level failure (surviving orphan): re-run the whole
                # pass — enumeration restarts from empty offsets so edges
                # missed earlier (or written late) are re-discovered, and
                # finalize work is re-minted. Classification itself is
                # DURABLE: settle rows keep their persisted ``classes``
                # because re-deciding on the post-cleanup graph would see
                # members whose derivation edges were already stripped as
                # parentless — a false ``unaffected`` that would let the
                # retried verify pass vacuously while rows remain.
                # Newly discovered members get fresh settle rows through
                # the planning barrier; all physical effects are idempotent
                # and suppression is never lifted.
                conn.execute(
                    "UPDATE closure_frontier"
                    " SET state = 'pending', detail_json = '{\"offsets\": {}}'"
                    " WHERE run_id = ? AND action = 'enumerate'",
                    (run_id,),
                )
                conn.execute(
                    "UPDATE closure_frontier SET state = 'pending'"
                    " WHERE run_id = ? AND action = 'settle'",
                    (run_id,),
                )
                conn.execute(
                    "DELETE FROM closure_frontier"
                    " WHERE run_id = ? AND action = 'finalize'",
                    (run_id,),
                )
                boundary = safe_json_loads(row["boundary_json"]) or {}
                boundary["planned"] = False
                boundary["plan_cursor"] = 0
                repos_v4.update(
                    conn,
                    "closure_runs",
                    {"boundary_json": json_dumps(boundary)},
                    {"run_id": run_id},
                )
            repos_v4.update(
                conn,
                "closure_runs",
                {
                    "phase": ClosurePhase.CLEANING.value,
                    "error": None,
                    "updated_us": wall_us(),
                },
                {"run_id": run_id},
            )
        return self._to_run(self._run_row(conn, run_id))

    # ------------------------------------------------------------------
    # drain + receipts
    # ------------------------------------------------------------------

    def drain(
        self,
        run_id: str,
        *,
        budget: int = _DRAIN_BUDGET,
        max_steps: int = _MAX_DRAIN_STEPS,
        conn: Optional[sqlite3.Connection] = None,
    ) -> ClosureRun:
        """Drive a run to a terminal phase. Never lies: a run that cannot
        finish inside ``max_steps`` raises CLOSURE_PENDING (retryable) —
        the frontier is durable, so a later drain continues."""
        steps = 0
        while True:
            run = self.step(run_id, budget=budget, conn=conn)
            if run.phase in _TERMINAL:
                return run
            steps += 1
            if steps >= max_steps:
                raise VerbatimError(
                    ErrorCode.CLOSURE_PENDING,
                    f"closure run {run_id} still {run.phase.value} after "
                    f"{max_steps} steps",
                    retryable=True,
                )

    def _receipt_counts(
        self,
        conn: sqlite3.Connection,
        run_row: dict[str, Any],
        members: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Surface tallies for the deletion receipt (V4-38.08)."""
        if members is None:
            members = self._load_members(conn, run_row)
        counts: dict[str, int] = {}
        external: list[dict[str, Any]] = []
        for key, info in members.items():
            cls = info.get("class")
            if cls == _C_ERASE:
                counts[f"derived_{key[0]}"] = (
                    counts.get(f"derived_{key[0]}", 0) + 1
                )
            elif cls == _C_SUPPRESS:
                counts["suppressed"] = counts.get("suppressed", 0) + 1
            elif cls == _C_REVALIDATE:
                counts["revalidate"] = counts.get("revalidate", 0) + 1
        for frow in conn.execute(
            "SELECT detail_json FROM closure_frontier"
            " WHERE run_id = ? AND action = 'finalize' AND state = 'done'",
            (run_row["run_id"],),
        ).fetchall():
            detail = safe_json_loads(frow[0] or "{}") or {}
            counts["derivation_edges"] = counts.get(
                "derivation_edges", 0
            ) + int(detail.get("edges_removed") or 0)
            counts["jobs_cancelled"] = counts.get(
                "jobs_cancelled", 0
            ) + int(detail.get("jobs_cancelled") or 0)
            counts["ledger_rows"] = counts.get(
                "ledger_rows", 0
            ) + int(detail.get("ledger") or 0)
            for p in detail.get("propagated") or []:
                external.append(
                    {
                        "ref": p.get("ref"),
                        "reason": "propagated_copy",
                        "detail": (
                            f"recipient {p['recipient_id']} holds a copy"
                        ),
                        "propagation_id": p["propagation_id"],
                        "recipient_id": p["recipient_id"],
                    }
                )
        outside = [
            {"ref": [k[0], k[1], k[2]],
             "reason": info.get("outside") or "outside_boundary"}
            for k, info in members.items() if info.get("class") == _C_OUTSIDE
        ]
        unaffected = [
            {"ref": [k[0], k[1], k[2]], "reason": "unaffected_derived"}
            for k, info in members.items()
            if info.get("class") == _C_UNAFFECTED
        ]
        return {
            "counts": counts,
            "outside_boundary": outside,
            "unaffected": unaffected,
            "external": external,
        }

    def receipt(
        self, run_id: str, *, conn: Optional[sqlite3.Connection] = None
    ) -> dict[str, Any]:
        """Deletion receipt (V4-38.08): verified surfaces, pending work,
        external copies, and cryptographic status — honest by construction."""
        require_id(run_id, "run_id")

        def _build(c: sqlite3.Connection) -> dict[str, Any]:
            row = self._run_row(c, run_id)
            members = self._load_members(c, row)
            boundary = safe_json_loads(row["boundary_json"]) or {}
            extra = self._receipt_counts(c, row, members)
            pending = {
                "pending": self._pending_count(c, run_id),
                "failed": int(
                    c.execute(
                        "SELECT COUNT(*) FROM closure_frontier"
                        " WHERE run_id = ? AND state = 'failed'",
                        (run_id,),
                    ).fetchone()[0]
                ),
            }
            return {
                "run_id": run_id,
                "scope_id": row["scope_id"],
                "phase": row["phase"],
                "purge_id": boundary.get("purge_id"),
                "erasure_epoch": int(row["erasure_epoch"]),
                "roots": safe_json_loads(row["roots_json"]) or [],
                "members": len(members),
                "erased": sorted(
                    (list(k) for k, m in members.items()
                     if m.get("class") == _C_ERASE),
                    key=_ref_sort,
                ),
                "suppressed": sorted(
                    (list(k) for k, m in members.items()
                     if m.get("class") == _C_SUPPRESS),
                    key=_ref_sort,
                ),
                "revalidate": sorted(
                    (list(k) for k, m in members.items()
                     if m.get("class") == _C_REVALIDATE),
                    key=_ref_sort,
                ),
                "outside_boundary": extra["outside_boundary"],
                "unaffected": extra["unaffected"],
                "surfaces": extra["counts"],
                "pending_work": pending,
                "external_copies": extra["external"],
                "verification": (safe_json_loads(row["verification_json"]) if row["verification_json"] else {}),
                "cryptographic_erasure": (
                    "unproven — logical erasure; backup/WAL copies may "
                    "remain while wrap keys exist (§35.05)"
                ),
                "backup_obligation": (
                    "backups taken before this erasure may still contain "
                    "the removed content; restore fencing uses the erasure "
                    "ledger digests"
                ),
                "error": row["error"],
            }

        if conn is not None:
            return _build(conn)
        with self._store.read() as rconn:
            return _build(rconn)


__all__ = [
    "ClosureEngine",
    "DEFAULT_BUDGET",
    "register_side_walker",
]
