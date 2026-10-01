"""SQLite store: connections, transactions, durability, and profile keying.

SQLite is the source of truth; every projection is rebuildable derived state
(SPEC §19-21). A single process-local writer connection serializes all
mutations behind a lock, while readers use thread-local WAL snapshot
connections so a read never blocks on — or is polluted by — an uncommitted
write. Content dedup fingerprints are keyed with a profile-local 256-bit HMAC
key so persisted hashes cannot be used to confirm low-entropy personal values
by dictionary attack (SPEC §8). The key is never stored in the database,
never logged, and never regenerated silently: opening a database whose key is
missing fails clearly rather than corrupting every dedup invariant.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac as _hmac_mod
import os
import secrets
import sqlite3
import stat
import threading
import time
from typing import Any, Iterator, Optional
from urllib.parse import quote

from ..core.time import now_us
from ..core.types import (
    ErrorCode,
    VerbatimError,
    json_dumps,
    new_id,
    safe_json_loads,
)
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
from .schema_v4 import v4_statements
from .schema_v5 import (
    DDL_V5_ADDITIVE,
    DDL_V5_FTS5,
    DDL_V5_JOBS_REBUILD,
    V5_FTS_TRIGGERS,
    v5_statements,
)
from . import commit_notify

_BUSY_TIMEOUT_MS = 250
_HMAC_KEY_BYTES = 32
_FTS_TABLE = "facts_fts_idx"
# V5 source-projection FTS5 shadow (docs/v5_contracts.md §3) — the same
# optional-capability probe as _FTS_TABLE; absence degrades the source
# lexical lane to unavailable, never a failed open.
_SOURCE_FTS_TABLE = "source_fts_idx"
# V4-40.04: ordinary write batches target at most 25 ms of lock hold;
# holds past the target are counted in diagnostics, not silently absorbed.
_WRITE_HOLD_TARGET_MS = 25
# Cadence for the in-store BEGIN retry loop (see ``_write_tx``): short
# enough to land in the sub-ms windows a chained writer leaves between
# transactions, long enough to stay off a pure spin. The *total* busy
# allowance is unchanged (``_BUSY_TIMEOUT_MS`` — V4-40.03).
_BEGIN_RETRY_SLEEP_S = 0.001

# Advisory per-path marks refreshed by ``_write_tx`` while its BEGIN is
# stalled on the WAL write lock; ``writers_waiting`` lets a drainer open
# a real inter-transaction window instead of chaining back-to-back write
# transactions until the contender's busy cap expires (the starvation
# mode that surfaced as "begin transaction: database is locked" on the
# consumer facade under a draining managed worker). Plain dict get/set
# only — GIL-atomic, no mutex a forked child could inherit held, and a
# stale or torn entry costs at most one bounded drain yield. Advisory,
# never ordering.
_WRITER_WAIT_MARKS: dict = {}

#: A mark counts as "a writer is stalled now" for this long after its
#: last refresh — covers the whole busy cap plus scheduling slack; a
#: stale mark only buys one extra bounded yield, so false positives are
#: cheap and false negatives (a just-expired mark on a still-waiting
#: writer) self-correct on that writer's next refresh.
_WRITER_WAIT_HORIZON_S = 0.4


def note_writer_wait(path: str) -> None:
    """Refresh ``path``'s stalled-writer mark (called from BEGIN retries)."""
    _WRITER_WAIT_MARKS[os.path.abspath(path)] = time.monotonic()


def writers_waiting(
    path: str, horizon_s: float = _WRITER_WAIT_HORIZON_S
) -> bool:
    """True when a writer stalled on ``path`` within ``horizon_s``."""
    mark = _WRITER_WAIT_MARKS.get(os.path.abspath(path))
    return mark is not None and (time.monotonic() - mark) <= max(
        horizon_s, 0.0
    )


#: Interleave window a ``_yield_to_blocked_writers`` store opens BEFORE
#: each of its write transactions while a peer writer is stalled on the
#: same file. ``Ingester.drain_report`` already yields this much between
#: jobs (its own ``_WRITER_YIELD_S``); doing it in ``_write_tx`` covers
#: the multi-transaction bodies one job commits — the gap a between-job
#: check cannot reach. ~1 ms BEGIN retries land in a 5 ms free window.
_WRITER_YIELD_S = 0.005


def _split_alters(ddl: str) -> list:
    """Split an ALTER-script string into individual statements."""
    return [s.strip() for s in ddl.split(";") if s.strip()]


def _creation_ddl() -> str:
    """The canonical DDL material whose digest a fresh store records in
    ``migration_history`` (SPEC_V4 V4-41.03).

    ``Store.create`` and ``migrations.acceptable_history_digests`` share
    this single definition so a creation row is verifiable against the
    immutable schema definition, exactly like a migration row.
    """
    return (
        DDL_V1 + DDL_V2 + DDL_V2_ALTER + DDL_V2_JOBS_REBUILD
        + DDL_V3 + DDL_V3_ALTER + DDL_V3_PROCEDURES_REBUILD
        + DDL_V3_JOBS_REBUILD + "".join(v4_statements())
        + DDL_FTS5 + FTS_TRIGGERS
    )


def _creation_ddl_v5() -> str:
    """The v5-era creation material whose digest a fresh v5 store records
    in ``migration_history``.

    ``_creation_ddl`` is intentionally left byte-identical (it remains the
    immutable v4-era definition historical digests verify against,
    V5-03.07); the v5 string inserts the v5 jobs-CHECK rebuild where
    ``Store.create`` executes it — after the v3 rebuild, before the v4
    statements — and appends the v5 tables plus the capability-gated
    source FTS shadow.
    """
    return (
        DDL_V1 + DDL_V2 + DDL_V2_ALTER + DDL_V2_JOBS_REBUILD
        + DDL_V3 + DDL_V3_ALTER + DDL_V3_PROCEDURES_REBUILD
        + DDL_V3_JOBS_REBUILD + DDL_V5_JOBS_REBUILD
        + "".join(v4_statements()) + "".join(v5_statements())
        + DDL_FTS5 + FTS_TRIGGERS + DDL_V5_FTS5 + V5_FTS_TRIGGERS
    )

# Fixed allowlist of canonical tables reported by check_integrity. These are
# literal identifiers, never caller-supplied text (SPEC §19: text payloads
# never become SQL; structural choices come from fixed allowlists).
_COUNT_TABLES = (
    "scopes",
    "sources",
    "source_revisions",
    "spans",
    "entities",
    "claims",
    "claim_revisions",
    "valid_intervals",
    "claim_evidence",
    "edges",
    "events",
    "decisions",
    "reviews",
    "jobs",
    "fts_rows",
    "feedback",
    "consents",
    "purges",
    # v5 source-projection plane (docs/v5_contracts.md §3) — derived
    # projections of retained source bytes; absent on pre-v5 stores, where
    # check_integrity reports None per table.
    "source_state",
    "source_lexical_projection",
    "source_fts_rows",
    "source_fts",
    "source_vectors",
    "entity_postings",
    "duplicate_links",
    "enrichment",
    "update_candidates",
    "backfill_cursor",
    # v7 unit-projection plane (schema_v7.V7_TABLES + V7_INTERNAL_TABLES —
    # physical names incl. the events/entity_aliases collision renames and
    # the FTS5 MATCH targets, which count as doc counts). Derived
    # projections of retained source bytes; absent on pre-v7 stores, where
    # check_integrity reports None per table.
    "units",
    "unit_fts_rows",
    "unit_fts_content",
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


def _db_error(exc: sqlite3.Error, operation: str) -> VerbatimError:
    """Translate sqlite failures into the stable error taxonomy (SPEC §43).

    Lock contention is retryable STORE_BUSY; integrity violations are caller
    VALIDATION errors; remaining OperationalErrors are durability failures;
    anything else under DatabaseError is treated as possible corruption so
    writes fail closed instead of compounding damage.
    """
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, sqlite3.IntegrityError):
        return VerbatimError(
            ErrorCode.VALIDATION, f"{operation}: constraint violation: {msg}"
        )
    if isinstance(exc, sqlite3.OperationalError):
        if "locked" in low:
            return VerbatimError(
                ErrorCode.STORE_BUSY,
                f"{operation}: database is locked",
                retryable=True,
            )
        return VerbatimError(ErrorCode.STORE_WRITE_FAILED, f"{operation}: {msg}")
    if isinstance(exc, sqlite3.DatabaseError):
        return VerbatimError(ErrorCode.STORE_CORRUPT, f"{operation}: {msg}")
    return VerbatimError(ErrorCode.STORE_WRITE_FAILED, f"{operation}: {msg}")


def _require_file(path: str, what: str) -> str:
    """Resolve a caller path and reject symlink escapes (SPEC §21, §41)."""
    abspath = os.path.abspath(path)
    if os.path.islink(abspath):
        raise VerbatimError(ErrorCode.VALIDATION, f"{what} path is a symlink: {abspath}")
    if os.path.isdir(abspath):
        raise VerbatimError(ErrorCode.VALIDATION, f"{what} path is a directory: {abspath}")
    return abspath


def _chmod(path: str, mode: int, *, strict: bool) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        if strict:
            raise


def _chmod_sidecars(db_path: str) -> None:
    """Restrict WAL/SHM companions where the filesystem supports it."""
    for suffix in ("-wal", "-shm"):
        _chmod(db_path + suffix, 0o600, strict=False)


class Store:
    """One profile's SQLite database: the only writer plus snapshot readers."""

    def __init__(
        self,
        path: str,
        key_path: str,
        hmac_key: bytes,
        writer: Optional[sqlite3.Connection],
        *,
        readonly: bool,
        schema_version: int,
        fts_enabled: bool,
        source_fts_enabled: bool = False,
    ) -> None:
        self._path = path
        self._key_path = key_path
        self._hmac_key = hmac_key
        # Pre-padded HMAC template: ``hmac()`` copies it and updates with
        # the message, so the two key-pad SHA-256 compressions run once
        # per Store instead of once per call — identical digests (HMAC
        # is keyed-hash, copying the pristine object is exactly what
        # ``hmac.copy()`` is specified for).
        self._hmac_template = _hmac_mod.new(
            self._hmac_key, b"", hashlib.sha256
        )
        self._writer = writer
        self._readonly = readonly
        self._schema_version = schema_version
        self._fts_enabled = fts_enabled
        self._source_fts_enabled = source_fts_enabled
        self._write_lock = threading.Lock()
        self._tx_holder: Optional[int] = None
        # Foreground-writer interleave (V6 fairness fix): when a drain-
        # owning store sets this (``ManagedWorker`` does on its private
        # store), ``_write_tx`` opens a real unlocked window BEFORE each
        # write transaction while a peer writer is stalled in BEGIN on
        # the same file — covering lease/execute/settle chains inside a
        # single job, which the between-job yield cannot reach. Opt-in,
        # advisory, bounded; ordinary stores default off.
        self._yield_to_blocked_writers = False
        # Monotonic commit counter — bumped after every successful COMMIT
        # on this Store's writer. Snapshot readers combine it with
        # ``PRAGMA data_version`` (which covers commits from any OTHER
        # connection) to key advisory content memos; the pair versions
        # "the committed database as this process can see it".
        self._write_epoch = 0
        # Commit notification — signaled after every successful COMMIT so
        # readiness barriers can wake on the settle instead of sleeping a
        # fixed poll interval (advisory: a missed signal costs at most one
        # poll delay, never correctness — waiters always re-verify).
        self._commit_cond = threading.Condition()
        self._clock_lock = threading.Lock()
        self._tls = threading.local()
        # reader conn -> creation time (us); the age of the oldest reader
        # bounds WAL retention and is reported in diagnostics (V4-40.07).
        self._readers: dict[sqlite3.Connection, int] = {}
        self._readers_lock = threading.Lock()
        self._closed = False
        self._last_event_us = 0
        # V4-40.07: measured writer admission + hold metrics.
        self._metrics_lock = threading.Lock()
        self._write_metrics: dict[str, int] = {
            "tx_count": 0,
            "wait_us_total": 0,
            "wait_us_max": 0,
            "wait_us_last": 0,
            "hold_us_total": 0,
            "hold_us_max": 0,
            "hold_us_last": 0,
            "hold_over_target": 0,
            "admission_timeouts": 0,
            "busy_errors": 0,
        }

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, path: str, *, hmac_key_path: Optional[str] = None) -> "Store":
        """Create a new profile database with schema v1 and a fresh HMAC key.

        The directory is created 0o700, the database 0o600, and the key file
        0o600. If a key file already exists it is loaded (never overwritten),
        which keeps dedup HMACs stable across recreated metadata. An existing
        non-empty database is never clobbered.
        """
        abspath = _require_file(path, "database")
        parent = os.path.dirname(abspath) or "."
        if not os.path.exists(parent):
            try:
                os.makedirs(parent, mode=0o700)
            except OSError as exc:
                raise VerbatimError(
                    ErrorCode.STORE_WRITE_FAILED,
                    f"cannot create data directory {parent}: {exc}",
                ) from exc
            _chmod(parent, 0o700, strict=False)
        elif not os.path.isdir(parent):
            raise VerbatimError(ErrorCode.VALIDATION, f"not a directory: {parent}")
        if os.path.exists(abspath) and os.path.getsize(abspath) > 0:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"database already exists; use Store.open: {abspath}",
            )

        key_path = (
            _require_file(hmac_key_path, "hmac key")
            if hmac_key_path
            else abspath + ".key"
        )
        key = cls._load_or_create_key(key_path, create=True)

        try:
            conn = sqlite3.connect(
                abspath, isolation_level=None, check_same_thread=False
            )
        except sqlite3.Error as exc:
            raise _db_error(exc, "create database") from exc
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(DDL_V1)
            conn.executescript(DDL_V2)
            for stmt in _split_alters(DDL_V2_ALTER):
                conn.execute(stmt)
            # jobs kind-CHECK rebuild runs last: it needs the operation_key/
            # lane columns the ALTERs just added. Fresh store — tables are
            # empty, so the parent swap is safe under foreign_keys=ON.
            for stmt in _split_alters(DDL_V2_JOBS_REBUILD):
                conn.execute(stmt)
            # v3 (SPEC_V3 §39): column additions first so the procedures
            # rebuild below can copy them, then new tables, then the two
            # CHECK-widening parent swaps (procedures, jobs) — all safe
            # under foreign_keys=ON because every table is still empty.
            conn.executescript(DDL_V3)
            for stmt in _split_alters(DDL_V3_ALTER):
                conn.execute(stmt)
            for stmt in _split_alters(DDL_V3_PROCEDURES_REBUILD):
                conn.execute(stmt)
            for stmt in _split_alters(DDL_V3_JOBS_REBUILD):
                conn.execute(stmt)
            # v5 jobs CHECK widening (V5-08.16): the source pipeline kinds
            # ride the same empty-table parent swap as the earlier rebuilds.
            for stmt in _split_alters(DDL_V5_JOBS_REBUILD):
                conn.execute(stmt)
            # v4 kernel tables (SPEC_V4 §41) — tables + purpose_tag ALTER +
            # the legacy-purpose tagging UPDATE, all on an empty store.
            for stmt in v4_statements():
                conn.execute(stmt)
            # v5 source-projection plane (docs/v5_contracts.md §3) —
            # control artifact, lexical/vector projections, postings,
            # dedup links, enrichment, update candidates, backfill cursor.
            for stmt in v5_statements():
                conn.execute(stmt)
            fts = True
            try:
                conn.executescript(DDL_FTS5)
                conn.executescript(FTS_TRIGGERS)
            except sqlite3.Error:
                # FTS5 is an optional capability: the store still works with
                # lexical search reported as unavailable (SPEC §20, §42).
                fts = False
            source_fts = True
            try:
                conn.executescript(DDL_V5_FTS5)
                conn.executescript(V5_FTS_TRIGGERS)
            except sqlite3.Error:
                # Same optional-capability rule for the v5 source shadow:
                # the lexical source lane reports unavailable instead of
                # failing creation.
                source_fts = False
            now = now_us()
            digest = hashlib.sha256(
                _creation_ddl_v5().encode("utf-8")
            ).hexdigest()
            conn.execute("BEGIN IMMEDIATE")
            try:
                for key_, value in (
                    ("schema_version", SCHEMA_VERSION),
                    ("db_id", new_id()),
                    ("created_us", now),
                    ("projection_generation", 1),
                    ("policy_epoch", 0),
                ):
                    conn.execute(
                        "INSERT INTO meta(key, value_json) VALUES (?, ?)",
                        (key_, json_dumps(value)),
                    )
                conn.execute(
                    "INSERT INTO migration_history(version, applied_us, migration_digest)"
                    " VALUES (?, ?, ?)",
                    (SCHEMA_VERSION, now, digest),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        except BaseException:
            conn.close()
            for suffix in ("", "-wal", "-shm", "-journal"):
                try:
                    os.remove(abspath + suffix)
                except OSError:
                    pass
            raise

        _chmod(abspath, 0o600, strict=True)
        _chmod_sidecars(abspath)
        return cls(
            abspath,
            key_path,
            key,
            conn,
            readonly=False,
            schema_version=SCHEMA_VERSION,
            fts_enabled=fts,
            source_fts_enabled=source_fts,
        )

    @classmethod
    def open(
        cls,
        path: str,
        *,
        hmac_key_path: Optional[str] = None,
        readonly: bool = False,
    ) -> "Store":
        """Open an existing database.

        A newer schema_version raises SCHEMA_UNSUPPORTED — never a downgrade
        attempt (SPEC §21). An older version is upgraded in place by
        ``migrations.apply`` before this returns (the migration is
        transactional and idempotent), so a writable open always yields a
        current-schema store; a read-only open skips migration and stays
        read-only. A missing or malformed HMAC key fails clearly; a
        different key is never generated silently.
        """
        abspath = _require_file(path, "database")
        if not os.path.exists(abspath):
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, f"database does not exist: {abspath}"
            )
        if os.path.getsize(abspath) == 0:
            raise VerbatimError(
                ErrorCode.STORE_CORRUPT, f"database file is empty: {abspath}"
            )
        key_path = (
            _require_file(hmac_key_path, "hmac key")
            if hmac_key_path
            else abspath + ".key"
        )

        if readonly:
            uri = "file:" + quote(abspath) + "?mode=ro"
            probe = sqlite3.connect(
                uri, uri=True, isolation_level=None, check_same_thread=False
            )
            writer = None
        else:
            writer = sqlite3.connect(
                abspath, isolation_level=None, check_same_thread=False
            )
            probe = writer

        try:
            probe.execute("PRAGMA foreign_keys = ON")
            probe.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            try:
                row = probe.execute(
                    "SELECT value_json FROM meta WHERE key = 'schema_version'"
                ).fetchone()
            except sqlite3.Error as exc:
                raise VerbatimError(
                    ErrorCode.STORE_CORRUPT,
                    f"missing meta table — not a verbatim database: {exc}",
                ) from exc
            if row is None:
                raise VerbatimError(
                    ErrorCode.STORE_CORRUPT, "meta.schema_version is absent"
                )
            try:
                version = int(safe_json_loads(row[0]))
            except (VerbatimError, ValueError) as exc:
                raise VerbatimError(
                    ErrorCode.STORE_CORRUPT, f"unreadable schema_version: {row[0]!r}"
                ) from exc
            if version > SCHEMA_VERSION:
                raise VerbatimError(
                    ErrorCode.SCHEMA_UNSUPPORTED,
                    f"schema {version} is newer than supported {SCHEMA_VERSION}; "
                    "refusing to open for mutation or downgrade",
                )

            # Only after the file proves to be a verbatim database do we
            # require its key — a missing key must fail clearly, never
            # silently generate a different one (SPEC §8).
            key = cls._load_or_create_key(key_path, create=False)

            if writer is not None:
                # journal_mode=WAL can return SQLITE_BUSY immediately (not
                # through the busy handler) while a peer holds the exclusive
                # migration lock — retry briefly so concurrent opens serialize
                # rather than crash (V4-41.07 / C64).
                wal_deadline = time.monotonic() + max(
                    _BUSY_TIMEOUT_MS / 1000.0, 5.0
                )
                while True:
                    try:
                        writer.execute("PRAGMA journal_mode = WAL")
                        break
                    except sqlite3.OperationalError as exc:
                        if (
                            "locked" not in str(exc).lower()
                            or time.monotonic() >= wal_deadline
                        ):
                            raise
                        time.sleep(0.05)
                writer.execute("PRAGMA synchronous = FULL")
                _chmod(abspath, 0o600, strict=False)
                _chmod_sidecars(abspath)

            fts = (
                probe.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (_FTS_TABLE,),
                ).fetchone()
                is not None
            )
            source_fts = (
                probe.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (_SOURCE_FTS_TABLE,),
                ).fetchone()
                is not None
            )
        except BaseException:
            probe.close()
            if writer is not None:
                writer.close()
            raise

        store = cls(
            abspath,
            key_path,
            key,
            writer,
            readonly=readonly,
            schema_version=version,
            fts_enabled=fts,
            source_fts_enabled=source_fts,
        )
        if readonly:
            # Seed this thread's reader with the already-open ro connection.
            probe.execute("PRAGMA query_only = ON")
            store._tls.conn = probe
            store._readers[probe] = now_us()
        try:
            store._last_event_us = probe.execute(
                "SELECT COALESCE(MAX(recorded_us), 0) FROM events"
            ).fetchone()[0]
        except sqlite3.Error:
            store._last_event_us = 0
        if writer is not None:
            # Migrate older schemas in place (idempotent) and resume any
            # crash-interrupted CHECK rebuilds/FTS creation — apply()'s
            # sniffers are no-ops on a fully current store, so this also
            # recovers a database whose schema_version committed before its
            # rebuilds finished. Deferred import: migrations imports Store.
            from . import migrations

            try:
                migrations.apply(store)
            except BaseException:
                store.close()
                raise
        return store

    @staticmethod
    def _read_key_file(key_path: str) -> bytes:
        """Read and validate an existing profile key; any defect fails LOCKED.

        A missing, unreadable, truncated, or malformed key is a locked-key
        condition (V4-39.02): without the exact persisted key every dedup
        HMAC would fork, so the failure must surface — never silently
        regenerate and never fall back to plaintext keying (C60).
        """
        try:
            with open(key_path, "rb") as fh:
                data = fh.read()
        except FileNotFoundError:
            raise VerbatimError(
                ErrorCode.LOCKED, f"hmac key file missing: {key_path}"
            )
        except OSError as exc:
            raise VerbatimError(
                ErrorCode.LOCKED, f"hmac key unreadable: {key_path}: {exc}"
            ) from exc
        if len(data) != _HMAC_KEY_BYTES:
            raise VerbatimError(
                ErrorCode.LOCKED,
                f"hmac key file must be exactly {_HMAC_KEY_BYTES} bytes "
                f"(got {len(data)}): {key_path}",
            )
        _chmod(key_path, 0o600, strict=False)
        return data

    @staticmethod
    def _read_key_file_wait(key_path: str) -> bytes:
        """Bounded wait for a peer's in-flight exclusive creation.

        The winner fsyncs file and directory before returning, so a
        0-byte or short read here is transient; poll briefly for the
        complete durable key. A file that stays malformed past the wait
        fails LOCKED — never overwritten, never regenerated (V4-39.02).
        """
        last_exc: Optional[VerbatimError] = None
        for _ in range(20):
            try:
                return Store._read_key_file(key_path)
            except VerbatimError as exc:
                last_exc = exc
                time.sleep(0.005)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _load_or_create_key(key_path: str, *, create: bool) -> bytes:
        """Load the 256-bit profile key; generate only when ``create``.

        Durable key protocol (V4-39.01): exclusive ``O_CREAT|O_EXCL``
        creation, a write-all loop with length verification, 0o600
        permissions, an fsync of the key file, and an fsync of the parent
        directory — all before the key is returned for dependent commits.
        The loser of a creation race polls briefly for the winner's
        durable key so both sides converge on one key; a missing or
        malformed existing key fails LOCKED, never regenerates
        (V4-39.02).
        """
        if os.path.islink(key_path):
            raise VerbatimError(
                ErrorCode.VALIDATION, f"hmac key path is a symlink: {key_path}"
            )
        try:
            return Store._read_key_file(key_path)
        except VerbatimError:
            if not create:
                raise  # missing/malformed on an open — locked key
            if os.path.exists(key_path):
                # Exists but unreadable/malformed: either a genuinely bad
                # key (fails LOCKED after the wait) or a peer's in-flight
                # O_EXCL creation (converges on the winner's key).
                return Store._read_key_file_wait(key_path)
            # create=True and the file is absent: fall through to the
            # exclusive-create path.

        key = secrets.token_bytes(_HMAC_KEY_BYTES)
        try:
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # Lost the O_EXCL race: converge on the winner's durable key.
            return Store._read_key_file_wait(key_path)
        except OSError as exc:
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID,
                f"cannot create hmac key file {key_path}: {exc}",
            ) from exc
        try:
            view = memoryview(key)
            while len(view):
                written = os.write(fd, view)
                view = view[written:]
            if os.fstat(fd).st_size != _HMAC_KEY_BYTES:
                raise VerbatimError(
                    ErrorCode.STORE_WRITE_FAILED,
                    f"hmac key write incomplete: {key_path}",
                )
            try:
                os.fchmod(fd, 0o600)  # mode change rides the same fsync
            except (AttributeError, OSError):
                pass
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            # O_EXCL guarantees this file is ours: remove the torn key so
            # a failed write never wedges every later open behind a
            # malformed-file LOCKED.
            try:
                os.remove(key_path)
            except OSError:
                pass
            raise
        os.close(fd)
        _chmod(key_path, 0o600, strict=True)
        # Parent-directory fsync: the directory entry itself must be
        # durable before the key may back dependent commits (V4-39.01).
        # Filesystems that cannot fsync a directory still have the file
        # fsync above; tolerate only that platform gap.
        try:
            dfd = os.open(
                os.path.dirname(key_path) or ".",
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
        return key

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def key_path(self) -> str:
        return self._key_path

    @property
    def readonly(self) -> bool:
        return self._readonly

    @property
    def schema_version(self) -> int:
        return self._schema_version

    @property
    def fts_enabled(self) -> bool:
        return self._fts_enabled

    @property
    def source_fts_enabled(self) -> bool:
        """The v5 source-projection FTS5 shadow exists (contracts §3)."""
        return self._source_fts_enabled

    # ------------------------------------------------------------------
    # connections
    # ------------------------------------------------------------------

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            uri = "file:" + quote(self._path) + "?mode=ro"
            conn = sqlite3.connect(
                uri, uri=True, isolation_level=None, check_same_thread=False
            )
        else:
            conn = sqlite3.connect(
                self._path, isolation_level=None, check_same_thread=False
            )
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        return conn

    def _reader(self) -> sqlite3.Connection:
        """This thread's snapshot connection; query_only so reads can't mutate."""
        if self._closed:
            raise VerbatimError(ErrorCode.VALIDATION, "store is closed")
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            try:
                # a conn closed under us (e.g. close() on another thread)
                # must not be handed back
                _ = conn.in_transaction
            except sqlite3.ProgrammingError:
                conn = None
                self._tls.conn = None
        if conn is None:
            conn = self._connect(readonly=self._readonly)
            conn.execute("PRAGMA query_only = ON")
            self._tls.conn = conn
            with self._readers_lock:
                self._readers[conn] = now_us()
        return conn

    # ------------------------------------------------------------------
    # transactions
    # ------------------------------------------------------------------

    def tx(
        self,
        *,
        deadline_us: Optional[int] = None,
        budget_ms: Optional[float] = None,
    ) -> "contextlib.AbstractContextManager[sqlite3.Connection]":
        """Write transaction: BEGIN IMMEDIATE, COMMIT or ROLLBACK (SPEC §21).

        The process-local write lock guarantees one writer per Store; the
        busy_timeout plus retryable STORE_BUSY covers cross-process
        contention, which a process lock cannot see.

        ``deadline_us``/``budget_ms`` enable deadline-aware writer
        admission (SPEC_V4 V4-40.02/03): the in-process lock wait never
        exceeds the request's remaining budget (timeout → retryable
        BACKPRESSURE), and the connection's SQLite busy_timeout is
        clamped to the smaller of the configured contention allowance
        and the remaining operation budget for the duration of the
        transaction.
        """
        if self._schema_version != SCHEMA_VERSION:
            raise VerbatimError(
                ErrorCode.SCHEMA_UNSUPPORTED,
                f"schema {self._schema_version} requires migration to "
                f"{SCHEMA_VERSION} before writes",
            )
        return self._write_tx(
            exclusive=False, deadline_us=deadline_us, budget_ms=budget_ms
        )

    def _record_write(
        self,
        *,
        wait_us: Optional[int] = None,
        hold_us: Optional[int] = None,
        timed_out: bool = False,
        busy: bool = False,
    ) -> None:
        """Accumulate writer admission/hold metrics (V4-40.07)."""
        with self._metrics_lock:
            m = self._write_metrics
            if wait_us is not None:
                m["wait_us_total"] += wait_us
                m["wait_us_last"] = wait_us
                if wait_us > m["wait_us_max"]:
                    m["wait_us_max"] = wait_us
            if hold_us is not None:
                m["tx_count"] += 1
                m["hold_us_total"] += hold_us
                m["hold_us_last"] = hold_us
                if hold_us > m["hold_us_max"]:
                    m["hold_us_max"] = hold_us
                if hold_us > _WRITE_HOLD_TARGET_MS * 1000:
                    m["hold_over_target"] += 1
            if timed_out:
                m["admission_timeouts"] += 1
            if busy:
                m["busy_errors"] += 1

    def _tx_error(self, exc: sqlite3.Error, operation: str) -> VerbatimError:
        """_db_error plus a busy counter so contention is measurable."""
        err = _db_error(exc, operation)
        if err.code is ErrorCode.STORE_BUSY:
            self._record_write(busy=True)
        return err

    @contextlib.contextmanager
    def _write_tx(
        self,
        *,
        exclusive: bool,
        deadline_us: Optional[int] = None,
        budget_ms: Optional[float] = None,
    ) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise VerbatimError(ErrorCode.VALIDATION, "store is closed")
        if self._readonly:
            raise VerbatimError(
                ErrorCode.STORE_WRITE_FAILED, "store was opened read-only"
            )
        if self._writer is None:
            raise VerbatimError(ErrorCode.STORE_WRITE_FAILED, "no writer connection")
        # A non-reentrant Lock would deadlock on a nested tx; detect the
        # reentry before blocking so the mistake fails fast instead.
        if self._tx_holder == threading.get_ident():
            raise VerbatimError(
                ErrorCode.VALIDATION, "nested write transaction on one Store"
            )

        # V4-40.02: effective admission deadline — the earlier of the
        # caller's absolute deadline and now + budget. Without either,
        # admission waits are unbounded (migrations and unsized callers).
        deadline_eff_us: Optional[int] = None
        if deadline_us is not None:
            deadline_eff_us = int(deadline_us)
        if budget_ms is not None:
            if budget_ms < 0:
                raise VerbatimError(
                    ErrorCode.VALIDATION, "budget_ms must be >= 0"
                )
            rel = now_us() + int(budget_ms * 1000)
            deadline_eff_us = (
                rel if deadline_eff_us is None else min(deadline_eff_us, rel)
            )

        wait_start = time.monotonic_ns()
        if deadline_eff_us is None:
            acquired = self._write_lock.acquire()
        else:
            remaining_us = deadline_eff_us - now_us()
            acquired = remaining_us > 0 and self._write_lock.acquire(
                timeout=remaining_us / 1e6
            )
        wait_us = int((time.monotonic_ns() - wait_start) // 1000)
        if not acquired:
            # Saturation returns explicit backpressure (V4-40.10): the
            # writer queue would out-wait the request's remaining budget.
            self._record_write(wait_us=wait_us, timed_out=True)
            raise VerbatimError(
                ErrorCode.BACKPRESSURE,
                f"write admission exceeded remaining budget "
                f"(waited {wait_us}us)",
                retryable=True,
            )
        self._tx_holder = threading.get_ident()
        conn = self._writer
        hold_start = time.monotonic_ns()
        # V4-40.03: bound the SQLite busy-wait by the smaller of the
        # configured contention allowance and the remaining budget.
        busy_ms = _BUSY_TIMEOUT_MS
        if deadline_eff_us is not None:
            remaining_ms = max(0, (deadline_eff_us - now_us()) // 1000)
            busy_ms = min(_BUSY_TIMEOUT_MS, remaining_ms)
        try:
            # Foreground-writer interleave (opt-in): a drain-owning
            # store (the managed worker's private Store) opens a real
            # unlocked window BEFORE this transaction while a peer
            # writer is stalled in BEGIN on the same file, so its ~1 ms
            # retry lands instead of expiring the busy cap behind our
            # back-to-back transactions. ``drain_report`` already yields
            # between jobs; doing it here covers the lease/execute/
            # settle chain inside a single job — the gap that produced
            # "begin transaction: database is locked" on the consumer
            # facade under drain load. Advisory + bounded: at most one
            # ``_WRITER_YIELD_S`` sleep per tx, only while a fresh
            # writer-wait mark exists, and it precedes the busy budget
            # so no caller's allowance is consumed.
            if self._yield_to_blocked_writers:
                try:
                    if writers_waiting(self._path):
                        time.sleep(_WRITER_YIELD_S)
                except Exception:
                    pass
            # BEGIN under our own bounded retry rather than the driver's
            # busy handler. The stock handler sleeps up to ~25 ms per
            # round; a writer chaining back-to-back transactions leaves
            # only sub-ms windows, so a contender routinely misses every
            # window until its cap expires ("begin transaction: database
            # is locked"). A ~1 ms retry cadence lands in the next free
            # window almost surely, under the *same* total busy cap
            # (busy_ms — V4-40.03 is unchanged; only the sleep
            # granularity differs). While stalled we mark the per-path
            # writer-wait registry so drain passes open a deliberate
            # yield window (``Ingester.drain_report``).
            begin_deadline = time.monotonic() + busy_ms / 1000.0
            begin_sql = "BEGIN EXCLUSIVE" if exclusive else "BEGIN IMMEDIATE"
            conn.execute("PRAGMA busy_timeout = 0")
            while True:
                try:
                    conn.execute(begin_sql)
                    break
                except sqlite3.Error as exc:
                    if "locked" not in str(exc).lower():
                        raise self._tx_error(exc, "begin transaction") from exc
                    remaining = begin_deadline - time.monotonic()
                    if remaining <= 0:
                        raise self._tx_error(exc, "begin transaction") from exc
                    try:
                        note_writer_wait(self._path)
                    except Exception:
                        pass
                    time.sleep(min(_BEGIN_RETRY_SLEEP_S, remaining))
            # The write lock is ours — restore the effective busy
            # allowance for the statements and COMMIT inside the tx.
            conn.execute(f"PRAGMA busy_timeout = {busy_ms}")
            try:
                try:
                    yield conn
                except sqlite3.Error as exc:
                    raise self._tx_error(exc, "transaction") from exc
                try:
                    conn.execute("COMMIT")
                except sqlite3.Error as exc:
                    raise self._tx_error(exc, "commit") from exc
                else:
                    self._write_epoch += 1
                    try:
                        with self._commit_cond:
                            self._commit_cond.notify_all()
                    except Exception:
                        pass
                    # V6-02.07: also fire the process-wide per-path
                    # registry so readiness barriers blocked on OTHER
                    # Store objects opened on this same file wake on the
                    # commit (advisory — never fails the commit).
                    try:
                        commit_notify.commit_fired(self._path)
                    except Exception:
                        pass
            finally:
                if conn.in_transaction:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
        finally:
            # Restore the configured contention allowance for the next
            # (possibly unbudgeted) writer — the BEGIN phase above always
            # leaves the connection at busy_timeout=0 or the clamped value.
            try:
                conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            except sqlite3.Error:
                pass
            self._record_write(
                wait_us=wait_us,
                hold_us=int((time.monotonic_ns() - hold_start) // 1000),
            )
            self._tx_holder = None
            self._write_lock.release()

    @contextlib.contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """Consistent WAL snapshot; nested calls reuse the open snapshot."""
        if self._closed:
            raise VerbatimError(ErrorCode.VALIDATION, "store is closed")
        conn = self._reader()
        if conn.in_transaction:
            yield conn
            return
        try:
            conn.execute("BEGIN")
        except sqlite3.Error as exc:
            raise _db_error(exc, "snapshot read") from exc
        try:
            yield conn
        except sqlite3.Error as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise _db_error(exc, "snapshot read") from exc
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        else:
            try:
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                raise _db_error(exc, "snapshot read") from exc

    # ------------------------------------------------------------------
    # keying, meta, and the logical clock
    # ------------------------------------------------------------------

    def hmac(self, data: Optional[bytes]) -> bytes:
        """Profile-keyed HMAC-SHA256 for persisted dedup fingerprints.

        Copies the pre-padded template so the keyed-pad compression is
        not re-run for every small message — the digest is identical to
        ``hmac.new(key, data, sha256)`` in every case, including the
        ``data=None`` form ``hmac.new`` tolerates (empty message).
        """
        h = self._hmac_template.copy()
        if data is not None:
            h.update(data)
        return h.digest()

    def _meta_get(self, conn: sqlite3.Connection, key: str) -> Any:
        row = conn.execute(
            "SELECT value_json FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return safe_json_loads(row[0]) if row else None

    def _meta_set(self, conn: sqlite3.Connection, key: str, value: Any) -> None:
        conn.execute(
            "INSERT INTO meta(key, value_json) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
            (key, json_dumps(value)),
        )

    def db_id(self) -> Optional[str]:
        return self._meta_get(self._reader(), "db_id")

    def projection_generation(self) -> int:
        value = self._meta_get(self._reader(), "projection_generation")
        return int(value) if value is not None else 0

    def bump_generation(self, conn: sqlite3.Connection) -> int:
        """Increment the projection generation inside the caller's write tx.

        Generation bumps ride in the same transaction as the projection
        change they describe, so readers either see the old projection or the
        new one — never a half-applied rebuild (SPEC §20).
        """
        current = self._meta_get(conn, "projection_generation")
        nxt = int(current or 0) + 1
        self._meta_set(conn, "projection_generation", nxt)
        return nxt

    def policy_epoch(self) -> int:
        value = self._meta_get(self._reader(), "policy_epoch")
        return int(value) if value is not None else 0

    def set_policy_epoch(self, conn: sqlite3.Connection, epoch: int) -> int:
        if epoch < 0:
            raise VerbatimError(ErrorCode.VALIDATION, "policy_epoch must be >= 0")
        self._meta_set(conn, "policy_epoch", int(epoch))
        return int(epoch)

    def next_event_us(self) -> int:
        """Nondecreasing logical UTC micros for event recorded_us.

        Recorded time is the engine's monotonic event order (SPEC §14); when
        the wall clock regresses the logical value keeps advancing one tick
        past the last issued value. ``observed_wall_us`` preserves the real
        reading separately.

        The open()-time seed is process-local: two processes sharing one
        database could otherwise mint duplicate or inverted recorded_us. So
        the last-issued value is clamped against the live ``MAX(recorded_us)``
        on the writer connection — inside a write tx this reads the
        in-flight maximum, keeping the logical clock monotone across
        processes as well as across clock regressions.
        """
        with self._clock_lock:
            last = self._last_event_us
            writer = self._writer
            if writer is not None:
                try:
                    live = writer.execute(
                        "SELECT COALESCE(MAX(recorded_us), 0) FROM events"
                    ).fetchone()[0]
                    if live > last:
                        last = live
                except sqlite3.Error:
                    pass
            now = now_us()
            nxt = now if now > last else last + 1
            self._last_event_us = nxt
            return nxt

    # ------------------------------------------------------------------
    # diagnostics, backup, shutdown
    # ------------------------------------------------------------------

    def check_integrity(self) -> dict[str, Any]:
        """Foreign-key audit, FTS capability, canonical table counts, and
        content-digest verification.

        ``content_integrity`` re-verifies every persisted digest against
        its bytes — ``payload_hmac`` on source revisions, ``excerpt_hmac``
        on spans, ``integrity_digest`` on source views. ``checked`` is the
        rows verified; ``failed`` rows mean stored bytes no longer match
        their digest (corruption, tampering, or a bad scrub). NULL digests
        are legacy rows and are counted in ``unverified``, never failed."""
        out: dict[str, Any] = {
            "db_id": self.db_id(),
            "schema_version": self._schema_version,
            "projection_generation": self.projection_generation(),
            "policy_epoch": self.policy_epoch(),
            "fts5": False,
            "source_fts5": False,
            "foreign_key_violations": 0,
            "quick_check": "unknown",
            "counts": {},
        }
        with self.read() as conn:
            try:
                violations = conn.execute("PRAGMA foreign_key_check").fetchall()
                out["foreign_key_violations"] = len(violations)
            except sqlite3.Error as exc:
                out["foreign_key_violations"] = -1
                out["foreign_key_error"] = str(exc)
            try:
                row = conn.execute("PRAGMA quick_check(1)").fetchone()
                out["quick_check"] = row[0] if row else "unknown"
            except sqlite3.Error as exc:
                out["quick_check"] = f"error: {exc}"
            for table in _COUNT_TABLES:
                try:
                    out["counts"][table] = conn.execute(
                        "SELECT COUNT(*) FROM " + table
                    ).fetchone()[0]
                except sqlite3.Error:
                    out["counts"][table] = None
            if self._fts_enabled:
                try:
                    conn.execute("SELECT COUNT(*) FROM facts_fts_idx").fetchone()
                    out["fts5"] = True
                except sqlite3.Error:
                    out["fts5"] = False
            if self._source_fts_enabled:
                try:
                    conn.execute(
                        "SELECT COUNT(*) FROM source_fts_idx"
                    ).fetchone()
                    out["source_fts5"] = True
                except sqlite3.Error:
                    out["source_fts5"] = False
            out["content_integrity"] = self._verify_content_digests(conn)
        return out

    def _verify_content_digests(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Re-verify every persisted content digest (see check_integrity)."""
        import hmac as _hmac

        res: dict[str, Any] = {"checked": 0, "failed": 0, "unverified": 0, "failures": []}

        def _fail(kind: str, ident: str) -> None:
            res["failed"] += 1
            if len(res["failures"]) < 16:
                res["failures"].append(f"{kind}:{ident}")

        def _b(v: Any) -> bytes:
            # BLOB columns can hold TEXT values on type-affinity-free
            # writes — normalize before digesting.
            if isinstance(v, (bytes, bytearray, memoryview)):
                return bytes(v)
            return str(v).encode("utf-8")

        def _has_col(table: str, col: str) -> bool:
            try:
                return any(
                    r[1] == col for r in conn.execute(f"PRAGMA table_info({table})")
                )
            except sqlite3.Error:
                return False

        try:
            for sid, rev, payload, digest in conn.execute(
                "SELECT source_id, revision, payload, payload_hmac"
                " FROM source_revisions"
            ):
                if digest is None:
                    res["unverified"] += 1
                    continue
                res["checked"] += 1
                if not _hmac.compare_digest(self.hmac(_b(payload)), bytes(digest)):
                    _fail("source_revision", f"{sid}@{rev}")
        except sqlite3.Error as exc:
            res["scan_error"] = str(exc)

        # Span excerpts are the persisted byte slice (SQLite substr is
        # 1-indexed); purged payloads are X'' — nothing left to check.
        if _has_col("spans", "excerpt_hmac"):
            try:
                for span_id, excerpt, digest in conn.execute(
                    "SELECT s.span_id,"
                    " substr(r.payload, s.start_byte + 1,"
                    "        s.end_byte - s.start_byte) AS excerpt,"
                    " s.excerpt_hmac"
                    " FROM spans s JOIN source_revisions r"
                    "   ON r.source_id = s.source_id"
                    "  AND r.revision = s.revision"
                    " WHERE length(r.payload) > 0"
                ):
                    if digest is None:
                        res["unverified"] += 1
                        continue
                    res["checked"] += 1
                    if not _hmac.compare_digest(
                        self.hmac(_b(excerpt)), bytes(digest)
                    ):
                        _fail("span", span_id)
            except sqlite3.Error as exc:
                res["scan_error"] = str(exc)
        else:
            res["span_excerpts"] = "skipped:no_excerpt_hmac_column"

        if _has_col("source_views", "integrity_digest"):
            try:
                for view_id, derived, digest in conn.execute(
                    "SELECT view_id, derived_bytes, integrity_digest"
                    " FROM source_views"
                ):
                    if derived is None or digest is None:
                        res["unverified"] += 1
                        continue
                    res["checked"] += 1
                    if not _hmac.compare_digest(
                        self.hmac(_b(derived)), bytes(digest)
                    ):
                        _fail("source_view", view_id)
            except sqlite3.Error as exc:
                res["scan_error"] = str(exc)
        return res

    def checkpoint_passive(self) -> dict[str, int]:
        """Best-effort PASSIVE WAL checkpoint on a private connection.

        Runs on its own short-lived connection rather than the shared
        writer so it never queues behind ``_write_lock`` and can never
        inject a statement between a write tx's BEGIN and COMMIT.
        ``PASSIVE`` never blocks readers or writers — it checkpoints only
        frames no active reader still needs and returns immediately when
        the log is busy.

        Keeping the WAL small between drain passes matters for latency:
        SQLite's autocheckpoint otherwise fires *inside* whichever write
        tx crosses the page threshold, adding 100-250 ms of checkpoint
        work to an unrelated commit. The worker calls this after each
        productive drain pass so the steady-state WAL stays near zero and
        the autocheckpoint threshold is never reached mid-measurement.
        Durability is unchanged — committed frames remain in the WAL
        either way; a checkpoint only copies them into the db file.
        """
        if self._writer is None or self._closed:
            return {"busy": 0, "log_frames": 0, "checkpointed_frames": 0}
        conn = sqlite3.connect(self._path, isolation_level=None)
        try:
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        finally:
            conn.close()
        if row is None:
            return {"busy": 0, "log_frames": 0, "checkpointed_frames": 0}
        return {
            "busy": int(row[0]),
            "log_frames": int(row[1]),
            "checkpointed_frames": int(row[2]),
        }

    def diagnostics(self) -> dict[str, Any]:
        """Operator diagnostics (SPEC_V4 V4-40.07): WAL bytes, checkpoint
        progress, oldest reader age, and measured write wait/hold.

        Write metrics are recorded on every ``_write_tx`` — wait is the
        time spent in writer admission (the in-process lock), hold is the
        time between lock acquisition and release, and
        ``hold_over_target`` counts transactions that exceeded the
        ordinary-batch target of 25 ms (V4-40.04).
        """
        with self._metrics_lock:
            m = dict(self._write_metrics)
        with self._readers_lock:
            reader_ages_us = [now_us() - t for t in self._readers.values()]
            reader_count = len(self._readers)
        try:
            wal_bytes = os.path.getsize(self._path + "-wal")
        except OSError:
            wal_bytes = 0
        ckpt: Any = "unavailable"
        conn = self._writer or getattr(self._tls, "conn", None)
        if conn is not None and not self._closed:
            try:
                row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                if row is not None:
                    ckpt = {
                        "busy": row[0],
                        "log_frames": row[1],
                        "checkpointed_frames": row[2],
                    }
            except sqlite3.Error as exc:
                ckpt = f"unavailable: {exc}"
        count = m["tx_count"]
        return {
            "schema_version": self._schema_version,
            "readonly": self._readonly,
            "fts5": self._fts_enabled,
            "source_fts5": self._source_fts_enabled,
            "write_tx": {
                "count": count,
                "wait_us_last": m["wait_us_last"],
                "wait_us_total": m["wait_us_total"],
                "wait_us_max": m["wait_us_max"],
                "wait_us_avg": m["wait_us_total"] // count if count else 0,
                "hold_us_last": m["hold_us_last"],
                "hold_us_total": m["hold_us_total"],
                "hold_us_max": m["hold_us_max"],
                "hold_us_avg": m["hold_us_total"] // count if count else 0,
                "hold_target_ms": _WRITE_HOLD_TARGET_MS,
                "hold_over_target": m["hold_over_target"],
                "admission_timeouts": m["admission_timeouts"],
                "busy_errors": m["busy_errors"],
            },
            "wal_bytes": wal_bytes,
            "wal_checkpoint": ckpt,
            "reader_count": reader_count,
            "oldest_reader_age_us": max(reader_ages_us) if reader_ages_us else 0,
        }

    def backup(self, dest_path: str) -> str:
        """Transactionally consistent copy via the SQLite backup API.

        A blind file copy could tear across a WAL commit; the backup API
        copies committed pages so the destination is always a coherent
        snapshot (SPEC §21, §40).
        """
        dest = _require_file(dest_path, "backup destination")
        if dest == self._path:
            raise VerbatimError(ErrorCode.VALIDATION, "backup destination is the database")
        parent = os.path.dirname(dest) or "."
        if not os.path.isdir(parent):
            try:
                os.makedirs(parent, mode=0o700)
            except OSError as exc:
                raise VerbatimError(
                    ErrorCode.STORE_WRITE_FAILED,
                    f"cannot create backup directory {parent}: {exc}",
                ) from exc
            _chmod(parent, 0o700, strict=False)
        dest_existed = os.path.exists(dest)
        src = self._connect(readonly=self._readonly)
        try:
            dst = sqlite3.connect(dest, isolation_level=None, check_same_thread=False)
            try:
                src.backup(dst)
            finally:
                dst.close()
        except sqlite3.Error as exc:
            # never leave a partial file that looks like a valid backup
            if not dest_existed:
                try:
                    os.remove(dest)
                except OSError:
                    pass
            raise _db_error(exc, "backup") from exc
        finally:
            src.close()
        _chmod(dest, 0o600, strict=False)
        return dest

    def close(self) -> None:
        """Close writer and every thread-local reader; idempotent."""
        if self._closed:
            return
        self._closed = True
        with self._readers_lock:
            readers = list(self._readers)
            self._readers.clear()
        for conn in readers:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        if getattr(self._tls, "conn", None) is not None:
            self._tls.conn = None
        if self._writer is not None:
            try:
                self._writer.close()
            except sqlite3.Error:
                pass
        _chmod_sidecars(self._path)

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        mode = "ro" if self._readonly else "rw"
        return f"<Store {self._path} {mode} schema={self._schema_version}>"


def file_permissions(path: str) -> int:
    """Permission bits of a store artifact; diagnostics only (SPEC §21)."""
    return stat.S_IMODE(os.stat(path).st_mode)
