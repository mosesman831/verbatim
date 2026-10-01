"""Ordered, transactional schema migrations (SPEC §21, SPEC_V4 §41).

Each migration runs inside a ``BEGIN EXCLUSIVE`` transaction — the
exclusive-begin guard is also the cross-process migration lock (C64,
V4-40.12): no reader or writer in any process can hold the database
while the schema is rewritten, and an interrupted migration rolls back
transactionally rather than resuming from an undefined half-state.
Ownership is rechecked after the lock is acquired (V4-41.07): a peer
process that completed the migration while this one waited is observed
through the recorded ``meta.schema_version``, and the step is skipped.

Every applied version is recorded in ``migration_history`` with a digest
of the exact statements. Before any mutation is permitted, every recorded
history row is verified against the immutable migration/creation
definitions (V4-41.03, C63) — a changed historical checksum blocks the
mutation with a diagnostic instead of building on drifted state.

Each migration phase is recorded in the ``schema_operations`` ledger
(SPEC_V4 §41) with declared → running → applied transitions; the public
``meta.schema_version`` advances inside the same commit, only after all
phases of the migration succeed (V4-41.05). Migration steps are
non-resumable single-transaction phases: a crash rolls the ledger rows
back with the schema, and the next ``apply`` restarts the step from zero
honestly (V4-41.06). The deferred rebuild phases that must run outside
the version commit (CHECK-widening table swaps, FTS creation, index
rebuilds) are resumable by schema introspection — their durable state IS
the schema shape — and are recorded in the ledger after their own
transactions commit.

Newer-than-supported schemas are never downgraded: ``apply`` raises
SCHEMA_UNSUPPORTED, matching ``Store.open``'s refusal (V4-41.08).
Migration SQL is stored as explicit statement tuples — not scripts —
because ``executescript`` would implicitly commit and break atomicity.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

from ..core.time import now_us
from ..core.types import ErrorCode, VerbatimError, safe_json_loads
from .schema import (
    DDL_FTS5,
    DDL_V1,
    DDL_V2,
    DDL_V2_ALTER,
    DDL_V2_JOBS_REBUILD,
    FTS_TRIGGERS,
    SCHEMA_VERSION,
)
from .schema_v3 import (
    DDL_V3,
    DDL_V3_ALTER,
    DDL_V3_JOBS_REBUILD,
    DDL_V3_PROCEDURES_REBUILD,
)
from .schema_v4 import SCHEMA_OPERATIONS_DDL, v4_statements
from .schema_v5 import (
    ADDITIVE_INDEXES,
    ADDITIVE_TABLES,
    DDL_V5_FTS5,
    DDL_V5_JOBS_REBUILD,
    V5_FTS_TRIGGERS,
    v5_additive_statements,
    v5_statements,
)
from .store import Store, _creation_ddl, _creation_ddl_v5


#: BEGIN retry cadence for the manual migration transactions — the same
#: mechanism ``Store._write_tx`` uses: the stock busy handler sleeps up
#: to ~25 ms per round while a chained writer leaves only sub-ms free
#: windows, so ``Store.open`` racing a hot drainer starved until the
#: 250 ms cap expired. ~1 ms retries land in the next window under the
#: SAME total busy allowance (``_BUSY_TIMEOUT_MS`` — V4-40.03 unchanged).
_BEGIN_RETRY_SLEEP_S = 0.001


def _begin_immediate(writer: sqlite3.Connection, store: Store) -> None:
    """``BEGIN IMMEDIATE`` under a bounded ~1 ms retry cadence.

    ``_BUSY_TIMEOUT_MS`` supplies the total cap — the same allowance the
    stock busy handler enforced (V4-40.03 unchanged); each stall sleeps
    at most that cap's remainder and marks the writer-wait registry so
    drain passes open a yield window for this opener
    (``writers_waiting``).
    """
    from . import store as _store_mod

    deadline = time.monotonic() + _store_mod._BUSY_TIMEOUT_MS / 1000.0
    while True:
        try:
            writer.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.Error as exc:
            if "locked" not in str(exc).lower():
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            try:
                _store_mod.note_writer_wait(store.path)
            except Exception:
                pass
            time.sleep(min(_BEGIN_RETRY_SLEEP_S, remaining))


@dataclass(frozen=True)
class Migration:
    """One irreversible step; statements execute in order inside one tx."""

    version: int
    name: str
    statements: tuple[str, ...]

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.name.encode("utf-8"))
        for stmt in self.statements:
            h.update(b"\0")
            h.update(stmt.encode("utf-8"))
        return h.hexdigest()


def _v2_statements() -> tuple[str, ...]:
    """DDL_V2 new tables followed by the ALTER column additions.

    executescript-style splitting: each statement ends at a semicolon;
    ``CREATE TABLE IF NOT EXISTS`` makes replay after a partial migration
    safe within the same transaction boundary.
    """
    stmts: list[str] = []
    buf: list[str] = []
    for line in DDL_V2.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        buf.append(line)
        if stripped.endswith(";"):
            stmts.append("\n".join(buf).rstrip().rstrip(";"))
            buf = []
    for stmt in DDL_V2_ALTER.split(";"):
        s = stmt.strip()
        if s:
            stmts.append(s)
    return tuple(stmts)


def _v3_statements() -> tuple[str, ...]:
    """DDL_V3 new tables + ALTER column additions (SPEC_V3 §39).

    The two CHECK-widening parent swaps (``procedures``, ``jobs``) are NOT
    here — like the v2 jobs rebuild they must run outside the migration
    transaction with ``PRAGMA foreign_keys`` OFF; ``_rebuild_v3_tables``
    handles them after this migration commits.
    """
    stmts: list[str] = []
    buf: list[str] = []
    for line in DDL_V3.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or stripped.startswith("PRAGMA") or not stripped:
            continue
        buf.append(line)
        if stripped.endswith(";"):
            stmts.append("\n".join(buf).rstrip().rstrip(";"))
            buf = []
    for stmt in DDL_V3_ALTER.split(";"):
        s = stmt.strip()
        if s:
            stmts.append(s)
    return tuple(stmts)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=2,
        name="v2-evidence-semantics",
        statements=_v2_statements(),
    ),
    Migration(
        version=3,
        name="v3-learning-substrate",
        statements=_v3_statements(),
    ),
    Migration(
        version=4,
        name="v4-kernel-contracts",
        statements=v4_statements(),
    ),
    Migration(
        # v5 source-projection plane (docs/v5_contracts.md §3). The jobs
        # CHECK widening (V5-08.16) and the source FTS5 shadow are NOT
        # here — like the earlier CHECK rebuilds and claims FTS they run
        # as fenced post-commit phases: the parent swap cannot run inside
        # a foreign-keys-on transaction, and FTS5 stays capability-gated.
        version=5,
        name="v5-source-projections",
        statements=v5_statements(),
    ),
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Every era's ``Store.create`` hashed the full DDL material of its day into
# ``migration_history``; a migrated store instead records
# ``Migration.digest()``. Both are immutable, reproducible definitions — a
# recorded digest that matches NEITHER is drift or corruption and must
# block mutation (V4-41.03). The v2 era shipped two create formulas (the
# jobs CHECK rebuild landed mid-era), so version 2 accepts both.
_CREATION_DDL_BY_VERSION: dict[int, tuple[str, ...]] = {
    1: (DDL_V1 + DDL_FTS5 + FTS_TRIGGERS,),
    2: (
        DDL_V1 + DDL_V2 + DDL_V2_ALTER + DDL_FTS5 + FTS_TRIGGERS,
        DDL_V1 + DDL_V2 + DDL_V2_ALTER + DDL_V2_JOBS_REBUILD
        + DDL_FTS5 + FTS_TRIGGERS,
    ),
    3: (
        DDL_V1 + DDL_V2 + DDL_V2_ALTER + DDL_V2_JOBS_REBUILD
        + DDL_V3 + DDL_V3_ALTER + DDL_V3_PROCEDURES_REBUILD
        + DDL_V3_JOBS_REBUILD + DDL_FTS5 + FTS_TRIGGERS,
    ),
    4: (_creation_ddl(),),
    5: (_creation_ddl_v5(),),
}


def acceptable_history_digests(version: int) -> frozenset[str]:
    """Legitimate ``migration_history`` digests for a recorded version:
    the immutable ``Migration.digest()`` plus each era's fresh-create
    digest. Empty for versions this binary has never produced.
    """
    accepted = {_sha(m) for m in _CREATION_DDL_BY_VERSION.get(version, ())}
    for migration in MIGRATIONS:
        if migration.version == version:
            accepted.add(migration.digest())
    return frozenset(accepted)


def _verify_history(conn: sqlite3.Connection) -> None:
    """V4-41.03 / C63: verify every recorded ``migration_history`` digest
    against the immutable migration definitions BEFORE any mutation is
    permitted. A mismatched or unknown-version row blocks mutation with a
    diagnostic — building on unverifiable history is prohibited.
    """
    try:
        rows = conn.execute(
            "SELECT version, migration_digest FROM migration_history"
        ).fetchall()
    except sqlite3.Error as exc:
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT, f"migration_history unreadable: {exc}"
        ) from exc
    for version, digest in rows:
        try:
            v = int(version)
        except (TypeError, ValueError) as exc:
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"migration_history has unreadable version {version!r}; "
                "refusing to mutate under unverifiable history",
            ) from exc
        accepted = acceptable_history_digests(v)
        if not accepted:
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"migration_history records unknown schema version "
                f"{v}; refusing to mutate under unverifiable history",
            )
        if digest not in accepted:
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT,
                f"migration_history checksum mismatch for schema version "
                f"{version}: recorded digest {str(digest)[:16]}… matches no "
                "immutable migration/creation definition (V4-41.03); "
                "refusing to mutate",
            )


def _recorded_version(conn: sqlite3.Connection) -> int:
    """``meta.schema_version`` as durably recorded — the cross-process
    ownership truth, rechecked after the exclusive lock is held (V4-41.07)."""
    row = conn.execute(
        "SELECT value_json FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT, "meta.schema_version is absent"
        )
    try:
        return int(safe_json_loads(row[0]))
    except (VerbatimError, ValueError) as exc:
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT, f"unreadable schema_version: {row[0]!r}"
        ) from exc


def _record_phase(
    conn: sqlite3.Connection,
    operation_id: str,
    version_from: int,
    version_to: int,
    phase: str,
    checksum: str,
    cursor: Optional[str],
    state: str,
    owner_lease: str,
) -> None:
    """Upsert one ``schema_operations`` row through the
    declared → running → applied transition sequence (V4-41.05)."""
    now = now_us()
    conn.execute(
        "INSERT INTO schema_operations("
        " operation_id, version_from, version_to, phase, checksum, cursor,"
        " state, owner_lease, created_us, updated_us)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(operation_id) DO UPDATE SET"
        "   state = excluded.state,"
        "   cursor = excluded.cursor,"
        "   owner_lease = excluded.owner_lease,"
        "   updated_us = excluded.updated_us",
        (
            operation_id,
            version_from,
            version_to,
            phase,
            checksum,
            cursor,
            state,
            owner_lease,
            now,
            now,
        ),
    )


def _record_deferred_phase(
    store: Store,
    phase: str,
    version: int,
    checksum: str,
    owner_lease: str,
) -> None:
    """Ledger row for a resumable post-commit phase that did real work.

    These phases persist no cursor — the schema shape itself is the resume
    state, probed by the sniffer on every ``apply`` (``cursor`` records
    that mechanism honestly). The row is written only after the phase's
    own transaction commits, so a crash mid-phase leaves no false
    'applied' marker (V4-41.06). No-ops when the ledger table is absent
    (a store whose history predates the ledger never reaches here with
    work to do, but the probe keeps the write honest).
    """
    writer = store._writer
    if writer is None:
        return
    # Manual tx under the write lock, same discipline as the rebuilds —
    # take the lock BEFORE the schema probe so it cannot observe another
    # thread's in-flight write-tx schema state.
    with store._write_lock:
        has_ledger = writer.execute(
            "SELECT 1 FROM sqlite_master"
            " WHERE type='table' AND name='schema_operations'"
        ).fetchone()
        if has_ledger is None:
            return
        _begin_immediate(writer, store)
        try:
            _record_phase(
                writer,
                f"schema-phase:{phase}",
                version,
                version,
                phase,
                checksum,
                "schema-sniff",
                "applied",
                owner_lease,
            )
            writer.execute("COMMIT")
        except sqlite3.Error:
            if writer.in_transaction:
                writer.execute("ROLLBACK")
            raise


def apply(store: Store) -> int:
    """Bring ``store`` to ``SCHEMA_VERSION``; returns the resulting version.

    No-op when current. Raises SCHEMA_UNSUPPORTED when the database is
    newer (no downgrade, V4-41.08) or when no ordered path reaches
    SCHEMA_VERSION, and STORE_CORRUPT when recorded migration history
    fails checksum verification (V4-41.03).
    """
    current = store.schema_version
    if current > SCHEMA_VERSION:
        raise VerbatimError(
            ErrorCode.SCHEMA_UNSUPPORTED,
            f"schema {current} is newer than supported {SCHEMA_VERSION}; "
            "refusing to downgrade",
        )

    # C63: verify recorded history before any mutation is permitted. The
    # read runs under the write lock so it cannot observe another thread's
    # in-flight write-tx state on the shared writer connection.
    if store._writer is not None:
        with store._write_lock:
            _verify_history(store._writer)
    else:
        _verify_history(store._reader())

    # Owner lease records which process/thread performed the phases —
    # meaningful across processes sharing the file (C64).
    owner_lease = f"pid:{os.getpid()} tid:{threading.get_ident()}"

    if current < SCHEMA_VERSION:
        pending = sorted(
            (m for m in MIGRATIONS if m.version > current),
            key=lambda m: m.version,
        )
        if not pending or pending[-1].version != SCHEMA_VERSION:
            raise VerbatimError(
                ErrorCode.SCHEMA_UNSUPPORTED,
                f"no migration path from schema {current} to {SCHEMA_VERSION}",
            )

        for migration in pending:
            with store._write_tx(exclusive=True) as conn:
                # V4-41.07 / C64: ownership recheck AFTER acquiring the
                # cross-process exclusive lock — a peer may have applied
                # this migration while we waited on the database lock.
                recorded = _recorded_version(conn)
                if recorded >= migration.version:
                    # Peer process already applied this step; adopt the
                    # durably recorded version as our own.
                    store._schema_version = recorded
                    continue
                # The ledger is itself migration machinery: creating it
                # early lets pre-v4 pending migrations record phases too.
                conn.execute(SCHEMA_OPERATIONS_DDL)
                op_id = f"schema-migration:{migration.name}"
                _record_phase(
                    conn, op_id, recorded, migration.version, "ddl",
                    migration.digest(), None, "declared", owner_lease,
                )
                _record_phase(
                    conn, op_id, recorded, migration.version, "ddl",
                    migration.digest(), None, "running", owner_lease,
                )
                for statement in migration.statements:
                    conn.execute(statement)
                _record_phase(
                    conn, op_id, recorded, migration.version, "ddl",
                    migration.digest(), None, "applied", owner_lease,
                )
                # V4-41.05: the public version advances in the same commit
                # as the applied phase record — only after all phases of
                # this migration succeeded.
                store._meta_set(conn, "schema_version", migration.version)
                conn.execute(
                    "INSERT INTO migration_history(version, applied_us, migration_digest)"
                    " VALUES (?, ?, ?)",
                    (migration.version, now_us(), migration.digest()),
                )
            store._schema_version = migration.version

    # Idempotent follow-ups run on EVERY apply(), not just after a migration
    # commits: the version + migration_history are committed inside the
    # migration transaction while the CHECK-widening rebuilds must run after
    # it, so a crash in between leaves schema_version recorded with pre-v3
    # constraints — the sniffers resume that half-migrated state here (and
    # are no-ops when the CHECKs are already current). Each resumable phase
    # that did real work is recorded in the schema_operations ledger.
    if _rebuild_jobs_check(store):
        _record_deferred_phase(
            store, "jobs-check-rebuild", 2, _sha(DDL_V2_JOBS_REBUILD), owner_lease
        )
    if _rebuild_v3_tables(store):
        _record_deferred_phase(
            store,
            "v3-table-rebuild",
            3,
            _sha(DDL_V3_PROCEDURES_REBUILD + DDL_V3_JOBS_REBUILD),
            owner_lease,
        )
    if _rebuild_v5_jobs(store):
        _record_deferred_phase(
            store, "v5-jobs-rebuild", 5,
            _sha(DDL_V5_JOBS_REBUILD), owner_lease,
        )
    if _ensure_fts(store):
        _record_deferred_phase(
            store, "fts-ensure", SCHEMA_VERSION,
            _sha(DDL_FTS5 + FTS_TRIGGERS), owner_lease,
        )
    if _rebuild_sources_dedup_index(store):
        _record_deferred_phase(
            store, "sources-dedup-reindex", 3,
            _sha("idx_sources_origin_ext:scope-partitioned"), owner_lease,
        )
    if _ensure_source_fts(store):
        _record_deferred_phase(
            store, "source-fts-ensure", SCHEMA_VERSION,
            _sha(DDL_V5_FTS5 + V5_FTS_TRIGGERS), owner_lease,
        )
    if _ensure_v6_additive(store):
        _record_deferred_phase(
            store, "v6-additive-ensure", SCHEMA_VERSION,
            _sha("".join(v5_additive_statements())), owner_lease,
        )
    return store.schema_version


def _rebuild_sources_dedup_index(store: Store) -> bool:
    """Partition the ``sources`` dedup index by scope (idempotent).

    The pre-partition index was ``UNIQUE(origin, external_id)`` globally:
    a host key captured into two scopes collided across the partition
    boundary — the second capture grafted onto the first scope's source,
    and purging one scope destroyed the other's evidence. The scoped
    index ``UNIQUE(scope_id, origin, external_id)`` is strictly weaker
    than the global one it replaces, so the rebuild can never collide
    with existing rows. No-ops once the index SQL mentions ``scope_id``.

    Returns True when the rebuild actually ran (for ledger recording).
    """
    writer = store._writer
    if writer is None:
        return False
    with store._write_lock:
        row = writer.execute(
            "SELECT sql FROM sqlite_master"
            " WHERE type='index' AND name='idx_sources_origin_ext'"
        ).fetchone()
        if row is None or "scope_id" in (row[0] or ""):
            return False
        # Manual tx — _write_tx would re-acquire the lock we already hold.
        _begin_immediate(writer, store)
        try:
            writer.execute("DROP INDEX idx_sources_origin_ext")
            writer.execute(
                "CREATE UNIQUE INDEX idx_sources_origin_ext"
                " ON sources(scope_id, origin, external_id)"
                " WHERE external_id IS NOT NULL"
            )
            writer.execute("COMMIT")
        except sqlite3.Error:
            if writer.in_transaction:
                writer.execute("ROLLBACK")
            raise
    return True


def _rebuild_jobs_check(store: Store) -> bool:
    """Widen ``jobs.kind`` CHECK to all JobKind values (idempotent).

    SQLite cannot ALTER a CHECK constraint, and the parent swap cannot run
    inside a foreign-keys-on transaction: ``DROP TABLE``'s implicit delete
    increments the deferred-FK violation counter, and the later
    ``RENAME`` does not decrement it — the commit then fails despite the
    final state being consistent. So the rebuild runs in its own
    transaction with ``PRAGMA foreign_keys`` OFF (the pragma is a no-op
    once a transaction is open, which is why this lives outside
    ``_write_tx``), followed by an explicit ``foreign_key_check`` — the
    sanctioned procedure for table swaps with child references.

    Runs only after the migration commits so the source table already
    carries ``operation_key``/``lane``. Skips when the CHECK already lists
    ``replay`` — fresh stores rebuild inline during ``create()``.

    Returns True when the rebuild actually ran (for ledger recording).
    """
    writer = store._writer
    if writer is None:
        return False
    # The manual transaction bypasses _write_tx's lock — take it BEFORE the
    # sqlite_master/table_info probes so they cannot observe another
    # thread's in-flight write-tx schema state.
    with store._write_lock:
        sql_row = writer.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        if sql_row is None or "replay" in (sql_row[0] or ""):
            return False
        cols = {r[1] for r in writer.execute("PRAGMA table_info(jobs)")}
        if not {"operation_key", "lane"} <= cols:
            return False  # pre-ALTER state; migration will run this after it commits

        writer.execute("PRAGMA foreign_keys = OFF")
        try:
            _begin_immediate(writer, store)
            try:
                for stmt in DDL_V2_JOBS_REBUILD.split(";"):
                    s = stmt.strip()
                    if s:
                        writer.execute(s)
                _restore_control_lane(writer)
                writer.execute("COMMIT")
            except sqlite3.Error:
                if writer.in_transaction:
                    writer.execute("ROLLBACK")
                raise
        finally:
            writer.execute("PRAGMA foreign_keys = ON")
    violations = writer.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT,
            f"jobs rebuild left {len(violations)} foreign-key violations",
        )
    return True


def _restore_control_lane(writer: sqlite3.Connection) -> None:
    """Re-tag in-flight control jobs that inherited the migration default.

    A v1 database has no ``jobs.lane`` column — the v2 ALTER fills every
    existing row with 'ordinary', which strips the reserved lane from
    in-flight purge/reindex/review_apply jobs (the kinds ``enqueue`` maps
    to 'control' via ``CONTROL_KINDS`` in jobs/queue.py). Restoring it for
    rows that are still in flight keeps deletion/correctness work ahead of
    enrichment; terminal rows are left as recorded history.
    """
    writer.execute(
        "UPDATE jobs SET lane = 'control'"
        " WHERE lane = 'ordinary'"
        " AND kind IN ('purge','reindex','review_apply')"
        " AND state IN ('queued','leased','retry_wait')"
    )


def _rebuild_v3_tables(store: Store) -> bool:
    """Apply the v3 CHECK-widening parent swaps (idempotent, FK-off fenced).

    Same constraint as ``_rebuild_jobs_check``: ``DROP TABLE`` inside a
    foreign-keys-on transaction cannot commit, so the ``procedures``
    (promotion-ladder CHECK, §21) and ``jobs`` (v3 kinds + four lanes, §40)
    rebuilds run here under one fenced FK-off transaction followed by an
    explicit ``foreign_key_check``. Skips tables already carrying the v3
    CHECK — fresh stores rebuilt inline during ``create()``.

    Returns True when the rebuild actually ran (for ledger recording).
    """
    writer = store._writer
    if writer is None:
        return False
    # Take the write lock BEFORE the sqlite_master/table_info probes so they
    # cannot observe another thread's in-flight write-tx schema state.
    with store._write_lock:
        proc_row = writer.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='procedures'"
        ).fetchone()
        jobs_row = writer.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        need_proc = proc_row is not None and "candidate" not in (proc_row[0] or "")
        need_jobs = (
            jobs_row is not None and "procedure_compile" not in (jobs_row[0] or "")
        )
        if not need_proc and not need_jobs:
            return False

        stmts: list[str] = []
        if need_proc:
            proc_cols = {r[1] for r in writer.execute("PRAGMA table_info(procedures)")}
            if "intent_signature_json" in proc_cols:
                stmts.extend(
                    s for s in DDL_V3_PROCEDURES_REBUILD.split(";") if s.strip()
                )
        did_jobs = False
        if need_jobs:
            job_cols = {r[1] for r in writer.execute("PRAGMA table_info(jobs)")}
            if "operation_key" in job_cols and "lane" in job_cols:
                stmts.extend(
                    s for s in DDL_V3_JOBS_REBUILD.split(";") if s.strip()
                )
                did_jobs = True
        if not stmts:
            return False

        writer.execute("PRAGMA foreign_keys = OFF")
        try:
            _begin_immediate(writer, store)
            try:
                for stmt in stmts:
                    writer.execute(stmt)
                if did_jobs:
                    _restore_control_lane(writer)
                writer.execute("COMMIT")
            except sqlite3.Error:
                if writer.in_transaction:
                    writer.execute("ROLLBACK")
                raise
        finally:
            writer.execute("PRAGMA foreign_keys = ON")
    violations = writer.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT,
            f"v3 table rebuild left {len(violations)} foreign-key violations",
        )
    return True


def _ensure_fts(store: Store) -> bool:
    """Create the FTS5 index + triggers if absent (idempotent, capability-gated).

    ``DDL_FTS5``/``FTS_TRIGGERS`` only run inside ``Store.create`` — a v1/v2
    database upgraded through ``apply()`` would otherwise keep
    ``fts_enabled=False`` forever. The same optional-capability probe as
    ``create()`` applies: when this SQLite build lacks FTS5 the store
    simply reports lexical search unavailable instead of failing open.

    Returns True when the index was actually created (for ledger recording).
    """
    writer = store._writer
    if writer is None or store._fts_enabled:
        return False
    with store._write_lock:
        try:
            writer.executescript(DDL_FTS5)
            writer.executescript(FTS_TRIGGERS)
        except sqlite3.Error:
            return False
        store._fts_enabled = True
    return True


def _rebuild_v5_jobs(store: Store) -> bool:
    """Widen ``jobs.kind`` CHECK to the v5 source-pipeline kinds (idempotent).

    V5-08.16: ``source_project``/``source_embed``/``source_backfill`` are
    persisted job kinds — the JobKind enum carries them but the v3-era
    CHECK rejects them, so the same fenced parent-swap as the v2/v3
    rebuilds applies: outside the migration transaction, ``PRAGMA
    foreign_keys = OFF``, then an explicit ``foreign_key_check``. Lane
    defaults are unchanged — v5 kinds enqueue onto existing lanes.

    Runs after ``_rebuild_v3_tables`` so a store migrating 1→5 or 3→5
    always rebuilds from the v3-shaped ``jobs`` (same column set). The
    schema shape is the resume state: a crash between the version commit
    and this phase is re-sniffed on the next ``apply``.

    Returns True when the rebuild actually ran (for ledger recording).
    """
    writer = store._writer
    if writer is None:
        return False
    # Take the write lock BEFORE the sqlite_master/table_info probes so they
    # cannot observe another thread's in-flight write-tx schema state.
    with store._write_lock:
        sql_row = writer.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        if sql_row is None or "source_project" in (sql_row[0] or ""):
            return False
        cols = {r[1] for r in writer.execute("PRAGMA table_info(jobs)")}
        if not {"operation_key", "lane"} <= cols:
            return False  # pre-v2 shape; earlier rebuilds fix it first

        writer.execute("PRAGMA foreign_keys = OFF")
        try:
            _begin_immediate(writer, store)
            try:
                for stmt in DDL_V5_JOBS_REBUILD.split(";"):
                    s = stmt.strip()
                    if s:
                        writer.execute(s)
                writer.execute("COMMIT")
            except sqlite3.Error:
                if writer.in_transaction:
                    writer.execute("ROLLBACK")
                raise
        finally:
            writer.execute("PRAGMA foreign_keys = ON")
    violations = writer.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise VerbatimError(
            ErrorCode.STORE_CORRUPT,
            f"jobs v5 rebuild left {len(violations)} foreign-key violations",
        )
    return True


def _ensure_source_fts(store: Store) -> bool:
    """Create the v5 source-projection FTS5 shadow + triggers if absent
    (idempotent, capability-gated).

    Same contract as ``_ensure_fts``: a v4 database upgraded through
    ``apply()`` gains ``source_fts_idx`` here, and a SQLite build without
    FTS5 keeps the store openable with the source lexical lane reported
    unavailable instead of failing the migration. The durable probe is the
    ``source_fts_idx`` row in sqlite_master — the schema shape itself is
    the resume state.

    Returns True when the index was actually created (for ledger
    recording).
    """
    writer = store._writer
    if writer is None or store._source_fts_enabled:
        return False
    with store._write_lock:
        try:
            writer.executescript(DDL_V5_FTS5)
            writer.executescript(V5_FTS_TRIGGERS)
        except sqlite3.Error:
            return False
        store._source_fts_enabled = True
    return True


def _ensure_v6_additive(store: Store) -> bool:
    """Create the V6 additive tables if absent (idempotent).

    ``source_exposure`` (V6-03.14) and ``policy_artifact_attestations``
    (V6-03.15) are additive to the frozen v5 set: they cannot ride
    ``v5_statements()`` — the migration-5 and v5-creation digests hash
    that exact text and ``_verify_history`` would then refuse every
    previously-recorded v5 store. Same ensure-phase family as
    ``_ensure_fts``/``_ensure_source_fts``/``_rebuild_sources_dedup_index``:
    the sqlite_master shape is the resume state, the phase runs on every
    ``apply`` (so a crashed or pre-V6 store self-heals on the next
    writable open), and it no-ops once the tables exist. ``Store.create``
    never calls ``apply`` — fresh stores get the tables lazily inside
    the first writer's own transaction via
    ``schema_v5.ensure_additive_tables``.

    Runs the statements inside one BEGIN IMMEDIATE transaction (not
    executescript — an explicit commit boundary keeps the phase
    transactional and matches the dedup-reindex precedent).

    Returns True when the phase actually created anything (for ledger
    recording).
    """
    writer = store._writer
    if writer is None:
        return False
    # Take the write lock BEFORE the sqlite_master probe so it cannot
    # observe another thread's in-flight write-tx schema state.
    with store._write_lock:
        missing = [
            name for name in ADDITIVE_TABLES
            if writer.execute(
                "SELECT 1 FROM sqlite_master"
                " WHERE type='table' AND name=?",
                (name,),
            ).fetchone() is None
        ]
        missing += [
            name for name in ADDITIVE_INDEXES
            if writer.execute(
                "SELECT 1 FROM sqlite_master"
                " WHERE type='index' AND name=?",
                (name,),
            ).fetchone() is None
        ]
        if not missing:
            return False
        # Manual tx under the write lock, same discipline as
        # _rebuild_sources_dedup_index — the CREATEs are transactional
        # and commit or roll back as one phase.
        _begin_immediate(writer, store)
        try:
            for stmt in v5_additive_statements():
                writer.execute(stmt)
            writer.execute("COMMIT")
        except sqlite3.Error:
            if writer.in_transaction:
                writer.execute("ROLLBACK")
            raise
    return True
