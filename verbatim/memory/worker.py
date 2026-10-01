"""Managed worker lifecycle for the v5 consumer facade (SPEC_V5 §09, V5-33.08/09).

One :class:`ManagedWorker` per resolved store identity owns a single bounded
daemon thread that drains the *existing* durable job queue through the
*existing* coordinator machinery — ``Ingester.drain_report`` →
``JobQueue.lease``/``assert_lease``/``complete``/``fail`` — so generation
fencing, lane priority, retry backoff, lease reclaim, and idempotent effect
commits are inherited, never re-implemented. There is no fifth queue and no
second effect-commit protocol here (V5-09.02).

Sharing (V5-09.02): facades do not own threads. ``acquire()`` returns a
ref-counted :class:`WorkerHandle` into a process-level registry keyed by the
resolved store identity (device+inode of the DB file, realpath fallback).
The last ``release()`` performs bounded shutdown of the shared worker;
releasing one of several owners never stops a sibling's worker (V5-09.07).

Enrolled namespaces (V5-09.13): the worker leases only inside scope ids its
attached owners explicitly enrolled — never a store-wide ``scope=None``
sweep. Submitters need not equal the owner: handlers re-verify each job's
persisted authority at commit, so a consented agent submission in an
enrolled namespace progresses under its own plan without inheriting any
facade's delivery/review grants. The permitted job set is explicit
(:data:`MANAGED_JOB_KINDS`): the local deterministic pipeline, *excluding*
the remote/optional categories ``connector_pull`` and ``projection_sync`` —
``local_memory`` activation cannot enable a remote/generative job category
without separate authorized configuration (V5-09.04). ``extra_kinds`` is
the opt-in channel for such authorization.

Isolation of connections (V5-09.08): the worker opens its *own* ``Store``
on the same resolved path. No facade ever closes a connection the worker
uses, and the worker's store is closed only after its thread has joined.
If cooperative shutdown misses its deadline the worker reports
``worker_stopped=False``/incomplete and its store stays open — retained,
not silently leaked — until a later stop succeeds or the process exits.

Fork policy (V5-09.10): the registry registers an ``os.register_at_fork``
three-phase protocol so a forked child sees an empty registry, never
touches the parent's worker objects, locks, or SQLite connections — and,
just as importantly, never inherits a *poisoned* SQLite/WAL lock table.

SQLite's POSIX WAL bookkeeping (``unixShmNode.aLock[]``/``sharedMask`` /
``exclMask``, ``pShmMutex``, ``unixBigLock``, the ``unixInodeInfo`` lock
table) is process-global and keyed by inode, not pid: a forked child
inherits it as a frozen snapshot. If any thread holds a WAL read-mark or
the write lock — or is merely inside a sqlite-internal mutex — at the
fork instant, a *fresh* child connection on the same DB file reuses that
stale node and sees phantom ``SQLITE_BUSY`` results (no fcntl is ever
issued, so busy_timeout can never win) or deadlocks inside a mutex whose
owner died in the fork. Resetting the Python registry cannot reach that
C-level state, so the ``before`` hook instead *quiesces* the only
background thread that can hold such state: every live managed worker
parks outside :func:`_sqlite_critical` sections until the fork completes,
guaranteeing the child inherits a lock table with no phantom held slots.

``after_in_parent`` resumes the workers; ``after_in_child`` resets the
registry and the quiesce state (inherited copies may show a stale
``_FORK_QUIESCED``). An inherited ``WorkerHandle``/``ManagedWorker`` in
the child answers ``status()``/``release()`` honestly (``forked_child``)
without acquiring any inherited lock (a parent's lock may be held by a
thread that died in the fork); every other entry point rejects or no-ops
on a pid mismatch *before* touching an inherited lock or connection, and
the drain loop self-terminates if it ever observes a PID change.
Correct usage in a child is a *fresh open* — a new ``Store`` +
``acquire()`` — which lands a brand-new worker under durable fencing.
Reusing inherited live objects is the facade's rejection duty; this
module only guarantees the child cannot inherit a runnable worker.

Known bound: the quiesce covers only threads this module controls — a
caller that forks while *its own* thread is inside a facade/store
transaction can still inherit a mid-lock snapshot; fork from a
quiescent point is the contract.

Crash safety (V5-09.11): a SIGKILL mid-lease leaves the job ``leased`` until
its persisted ``lease_until_us`` expires; the next drain's
``reclaim_expired`` returns it to ``retry_wait`` with a bumped generation
that fences the dead worker out. No accepted work is lost, and nothing is
acknowledged early — every effect commits inside the queue's fenced
transactions.

Backpressure (V5-33.09): bounded batch per enrolled scope per pass
(``batch_size``), a monotonic yield between busy passes, and an idle wait
bounded at ``idle_ms`` — default 25 ms, inside the ≤50 ms group-commit
cadence target (V5-33.08). The facade's ``add`` may call ``wake()`` to cut
pickup latency below the poll cadence; polling alone still bounds it.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import re
import threading
import time
from typing import Any, Iterable, Iterator, Optional

from ..config import VerbatimConfig
from ..core.time import now_us
from ..core.types import ErrorCode, JobKind, VerbatimError, require_id
from ..jobs.queue import DEFAULT_LEASE_S
from ..storage.store import Store
from .types import CloseReport

# ------------------------------------------------------------------
# tunables (V5-33.08/09)
# ------------------------------------------------------------------

#: Jobs leased-and-executed per enrolled scope per drain pass. Bounds the
#: work between stop-checks and lets round-robin cross-scope fairness and
#: the write-lock yield apply (V5-33.09).
DEFAULT_BATCH_SIZE = 8
#: Idle poll cadence. A newly enqueued job is picked up within ~idle_ms even
#: without a ``wake()`` poke — inside the ≤50 ms group-commit target.
DEFAULT_IDLE_MS = 25.0
#: Cooperative yield between busy passes (scheduler yield, not a busy loop).
DEFAULT_YIELD_MS = 1.0
#: Backoff cap after infrastructure-level drain errors (STORE_BUSY etc.).
_ERROR_BACKOFF_CAP_MS = 500.0
#: Bound for best-effort GC/atexit shutdown — cleanup is best-effort, never
#: a durability guarantee (V5-09.09).
_FINALIZER_TIMEOUT_MS = 1000.0

#: The explicit permitted job/profile set for ``local_memory`` managed
#: workers (V5-09.04/09.13): every registered kind *except* the remote /
#: optional-service categories. ``connector_pull`` pulls from external
#: connectors and ``projection_sync`` pushes to external projections —
#: both are remote-capable categories a capture-only profile may not
#: enable without separate authorized configuration. Kinds with no
#: registered handler in this build (e.g. ``compare``, or v5 ``source_*``
#: before their module lands) still lease and then fail loudly under the
#: existing dispatcher's CAPABILITY_UNAVAILABLE path — an honest terminal
#: state, never a silent no-op and never an infinite retry of
#: unimplementable work.
_EXCLUDED_KINDS = frozenset({JobKind.CONNECTOR_PULL, JobKind.PROJECTION_SYNC})
MANAGED_JOB_KINDS = frozenset(k for k in JobKind if k not in _EXCLUDED_KINDS)

_TAG_CLEAN = re.compile(r"[^A-Za-z0-9_.:-]+")

# ------------------------------------------------------------------
# fork quiesce (V5-09.10)
# ------------------------------------------------------------------
#
# A forked child inherits this process's *snapshot* of libsqlite3's
# process-global POSIX bookkeeping — ``unixShmNode.aLock[]`` /
# ``sharedMask`` / ``exclMask``, ``pShmMutex``, ``unixBigLock`` and the
# ``unixInodeInfo`` lock table — all keyed by inode, not pid. Any WAL
# read-mark or write lock (or merely a held sqlite-internal mutex) live
# at the fork instant freezes into the child's copy: its fresh
# connections then see phantom ``SQLITE_BUSY`` results that no busy
# timeout can outlast, or deadlock inside a mutex whose owner died in
# the fork. The only reliable cure is to ensure no thread this module
# controls is inside a SQLite call when the fork happens — so the
# ``before`` hook below parks every worker outside its
# :func:`_sqlite_critical` sections until the fork completes.
#
# ``_FORK_QUIESCED`` is only ever set while a fork is in flight;
# ``_FORK_IN_TX`` counts threads inside a section that may hold WAL
# locks or sqlite mutexes. Both are rebuilt in the child by
# ``_reset_registry_after_fork`` — the inherited lock objects may be in
# an arbitrary mid-operation state and are never reused there.

#: Bound on the pre-fork wait for in-flight SQLite sections to drain.
#: One ``drain_report`` call is bounded by ``batch_size`` jobs, so this
#: comfortably covers a pass; on expiry the fork proceeds degraded (the
#: child may inherit stale WAL bookkeeping) rather than deadlocking the
#: forking thread.
_FORK_QUIESCE_TIMEOUT_S = 1.5

# RLock, not Lock: ``_FORK_IN_TX`` is a per-thread depth counter, and a
# GC finalizer (``Memory.__del__`` → close → worker ``stop`` →
# ``_sqlite_critical``) may fire inside the counter-mutation window on a
# thread already holding the lock — a plain Lock would self-deadlock.
_FORK_LOCK = threading.RLock()
_FORK_ZERO = threading.Condition(_FORK_LOCK)
_FORK_QUIESCED = False
_FORK_IN_TX: dict[int, int] = {}  # thread ident -> live section depth


@contextlib.contextmanager
def _sqlite_critical() -> Iterator[None]:
    """A section in which this thread may hold WAL locks/sqlite mutexes.

    Entering parks the caller while a fork is in flight; the per-thread
    in-flight count lets the ``before`` hook wait for every *other*
    thread's section to drain, so the child never inherits a mid-lock
    snapshot. Threads hold ``_FORK_LOCK`` only to mutate the counter —
    never while inside SQLite — so the hook can never deadlock against
    a lock-holder. The forking thread's own section, if any, is
    inherently un-quiescable (the call is already in flight) and is
    simply not waited on.
    """
    ident = threading.get_ident()
    with _FORK_LOCK:
        while _FORK_QUIESCED:
            _FORK_ZERO.wait(0.05)
        _FORK_IN_TX[ident] = _FORK_IN_TX.get(ident, 0) + 1
    try:
        yield
    finally:
        with _FORK_LOCK:
            n = _FORK_IN_TX.get(ident, 0) - 1
            if n > 0:
                _FORK_IN_TX[ident] = n
            else:
                _FORK_IN_TX.pop(ident, None)
            _FORK_ZERO.notify_all()


def _quiesce_workers_before_fork() -> None:
    """``os.register_at_fork`` ``before`` hook (runs in the forking thread,
    other threads still live): raise the quiesce flag, then wait —
    bounded — until no *other* thread sits inside a
    :func:`_sqlite_critical` section. Workers outside a section hold no
    WAL locks and cannot enter one while the flag is set, so the child's
    inherited ``aLock[]``/mutex table is a clean snapshot (V5-09.10)."""
    global _FORK_QUIESCED
    me = threading.get_ident()
    try:
        acquired = _FORK_LOCK.acquire(timeout=_FORK_QUIESCE_TIMEOUT_S)
    except Exception:
        acquired = False
    if not acquired:
        # Never deadlock the forker: set the flag locklessly (workers
        # will still park) and let the fork proceed degraded.
        _FORK_QUIESCED = True
        return
    try:
        _FORK_QUIESCED = True
        deadline = time.monotonic() + _FORK_QUIESCE_TIMEOUT_S
        while any(t != me for t in _FORK_IN_TX):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _FORK_ZERO.wait(min(remaining, 0.05))
    finally:
        _FORK_LOCK.release()


#: Grace window after a fork during which the parent's workers stay
#: parked. The child's first act is a fresh ``Store`` open + bootstrap —
#: a *single-shot* ``BEGIN IMMEDIATE`` bounded by the store's 250 ms
#: busy_timeout — while the parent's worker would otherwise resume
#: churning WAL write locks. Holding the quiesce a beat longer gives the
#: child a quiet WAL to land in; the parent's drain simply resumes a
#: moment later (V5-09.10).
_FORK_RESUME_DELAY_S = 0.1


def _resume_workers_after_fork_parent() -> None:
    """``after_in_parent`` hook: release the quiesce so parked workers
    resume — after a short grace window so the child can land its fresh
    open in a quiet WAL. The delay runs on a daemon timer so the
    forking thread returns immediately."""
    try:
        if not _REGISTRY:
            # Nothing was quiesced — release inline.
            _release_quiesce()
            return
        timer = threading.Timer(_FORK_RESUME_DELAY_S, _release_quiesce)
        timer.daemon = True
        timer.start()
    except Exception:
        # Post-fork parent hooks must not raise into os.fork(); fall back
        # to an immediate release — degraded, never broken.
        _release_quiesce()


def _release_quiesce() -> None:
    global _FORK_QUIESCED
    try:
        with _FORK_LOCK:
            _FORK_QUIESCED = False
            _FORK_ZERO.notify_all()
    except Exception:
        _FORK_QUIESCED = False


def _store_tag(path: str) -> str:
    base = os.path.basename(path.rstrip(os.sep)) or "store"
    tag = _TAG_CLEAN.sub("_", base)
    return tag[:24] or "store"


def store_identity(store: Any) -> str:
    """Resolved store identity for the process-level worker registry.

    The DB file's ``(st_dev, st_ino)`` is the identity — two ``Store``
    objects opened on the same file (hardlinks included) share one worker,
    and a deleted+recreated file correctly reads as a *different* store.
    Falls back to the realpath when stat fails, then to object identity
    for non-file stores (which ``acquire`` rejects anyway — a managed
    worker needs a path to open its own connections).
    """
    path = getattr(store, "path", None)
    if isinstance(path, str) and path:
        try:
            st = os.stat(path)
            return f"ino:{st.st_dev:x}:{st.st_ino:x}"
        except OSError:
            return "path:" + os.path.realpath(path)
    return f"obj:{id(store):x}"


def _kind_set(kinds: Optional[Iterable[Any]], extra: Iterable[Any]) -> tuple[str, ...]:
    """Resolve the permitted kind set; unknown kinds are a typed error."""
    try:
        base = (
            set(MANAGED_JOB_KINDS)
            if kinds is None
            else {JobKind(k) for k in kinds}
        )
        for k in extra:
            base.add(JobKind(k))
    except ValueError as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION, f"unknown job kind in worker set: {exc}"
        ) from exc
    if not base:
        raise VerbatimError(
            ErrorCode.VALIDATION, "managed worker kind set must be non-empty"
        )
    return tuple(sorted(k.value for k in base))


# ------------------------------------------------------------------
# ManagedWorker — the shared per-store drain owner
# ------------------------------------------------------------------


class ManagedWorker:
    """One bounded daemon drain thread bound to one store.

    Constructed by :func:`acquire` (ref-counted registry); direct
    construction is allowed for embedding/tests — call :meth:`start` then
    :meth:`stop`. The worker opens its own ``Store`` on ``store.path`` so
    connection lifetime is worker-owned (V5-09.08) and drains through a
    private :class:`~verbatim.ingest.Ingester` — the same drain machinery
    the CLI, SDK, and provider session-end seams use.
    """

    def __init__(
        self,
        store: Store,
        cfg: VerbatimConfig,
        *,
        kinds: Optional[Iterable[Any]] = None,
        extra_kinds: Iterable[Any] = (),
        batch_size: int = DEFAULT_BATCH_SIZE,
        idle_ms: float = DEFAULT_IDLE_MS,
        yield_ms: float = DEFAULT_YIELD_MS,
        lease_s: float = DEFAULT_LEASE_S,
        spawn_thread: Optional[Any] = None,
        encoder: Any = None,
        judge: Any = None,
        transport_broker: Any = None,
        generation: int = 0,
    ) -> None:
        if not isinstance(cfg, VerbatimConfig):
            raise VerbatimError(
                ErrorCode.CONFIG_INVALID, "cfg must be a VerbatimConfig"
            )
        if batch_size < 1:
            raise VerbatimError(ErrorCode.VALIDATION, "batch_size must be >= 1")
        if idle_ms <= 0 or yield_ms < 0:
            raise VerbatimError(
                ErrorCode.VALIDATION, "idle_ms must be > 0 and yield_ms >= 0"
            )
        if lease_s <= 0:
            raise VerbatimError(ErrorCode.VALIDATION, "lease_s must be > 0")
        self._kinds = _kind_set(kinds, extra_kinds)
        self._cfg = cfg
        self._pid = os.getpid()
        self._owner = f"mw:{_store_tag(store.path)}:{generation}:{self._pid}"
        self._batch_size = int(batch_size)
        self._idle_s = float(idle_ms) / 1000.0
        self._yield_s = float(yield_ms) / 1000.0
        self._lease_s = float(lease_s)
        self._spawn = spawn_thread
        # Worker's own connection set on the same store file — opened in
        # the acquiring thread so a failure surfaces at acquire time.
        # The open runs inside the fork-quiesce section: a sibling
        # thread forking while this conn is mid-open would leave the
        # child a half-initialised unixShmNode/unixInodeInfo snapshot
        # (V5-09.10).
        with _sqlite_critical():
            self._store = Store.open(store.path, hmac_key_path=store.key_path)
        # Foreground-writer interleave (V6): this drain owner opens a
        # real unlocked window BEFORE each of its write transactions
        # while a peer writer on the same file is stalled in BEGIN —
        # the between-job yield alone cannot cover the multi-tx bodies
        # one job commits (lease + handler commits + settle), which is
        # how a facade writer's 250 ms busy cap starved into
        # "begin transaction: database is locked" under drain load.
        self._store._yield_to_blocked_writers = True
        try:
            # Deferred import: worker → ingest edge (mirrors sdk/capture.py).
            from ..ingest import Ingester

            self._ingester = Ingester(
                self._store,
                cfg,
                judge,
                encoder=(
                    encoder
                    if encoder is not None
                    else getattr(store, "encoder", None)
                ),
                transport_broker=(
                    transport_broker
                    if transport_broker is not None
                    else getattr(store, "transport_broker", None)
                ),
            )
        except BaseException:
            with _sqlite_critical():
                self._store.close()
            raise

        # RLock, not Lock: a GC-triggered ``Memory.__del__`` running on
        # this thread while we hold ``_lock`` (e.g. inside ``acquire``'s
        # refcount section or an ``enroll`` dict write) re-enters via
        # ``_release`` — a plain Lock would self-deadlock (V6/F24 hang).
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._enrolled: dict[str, int] = {}  # scope_id -> owner refcount
        self._refs = 0  # live WorkerHandles (mutated under registry lock)
        self._state = "created"  # created|running|stopping|stopped|defunct_fork
        self._thread: Optional[threading.Thread] = None
        self._started_us: Optional[int] = None
        self._counters = {
            "passes": 0,
            "jobs_processed": 0,
            "jobs_succeeded": 0,
            "jobs_failed": 0,
            "jobs_deferred": 0,
            "jobs_expired": 0,
        }
        self._last_error: Optional[dict[str, Any]] = None
        self._last_pass_us: Optional[int] = None
        self._consecutive_errors = 0
        self._stop_report: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # identity / lifecycle
    # ------------------------------------------------------------------

    @property
    def owner(self) -> str:
        """Lease-owner token written on rows this worker leases."""
        return self._owner

    @property
    def store(self) -> Store:
        """The worker's own Store — never the facade's connection."""
        return self._store

    @property
    def running(self) -> bool:
        t = self._thread
        return (
            self._state == "running"
            and t is not None
            and t.is_alive()
            and os.getpid() == self._pid
        )

    def start(self) -> None:
        """Spawn the bounded daemon drain thread (idempotent)."""
        if os.getpid() != self._pid:
            # PID check BEFORE the inherited lock: in a forked child it
            # may be held by a thread that died in the fork (V5-09.10).
            self._state = "defunct_fork"
            return
        with self._lock:
            if self._state != "created":
                return
            self._state = "running"
            self._started_us = now_us()
            spawn = self._spawn
            target = self._loop
            name = f"verbatim-{self._owner}"
        if spawn is not None:
            # HostAdapter.spawn_thread returns an already-started thread.
            thread = spawn(target, name)
            with self._lock:
                if self._thread is None:
                    self._thread = thread
        else:
            thread = threading.Thread(target=target, name=name, daemon=True)
            with self._lock:
                if self._thread is None:
                    self._thread = thread
            thread.start()

    # -- namespace enrollment (V5-09.13) ----------------------------------

    def enroll(self, scope_id: str) -> None:
        """Enroll a namespace: the worker may lease its durable jobs."""
        sid = require_id(scope_id, "scope_id")
        if os.getpid() != self._pid:
            # A forked child must not mutate the parent's enrollment —
            # and must not acquire the inherited lock to find that out
            # (V5-09.10). Enrollment is a no-op on a foreign object.
            return
        with self._lock:
            self._enrolled[sid] = self._enrolled.get(sid, 0) + 1

    def leave(self, scope_id: str) -> None:
        """Detach one owner of ``scope_id``; unenrolls at zero."""
        if os.getpid() != self._pid:
            return  # never touch inherited lock/state (V5-09.10)
        with self._lock:
            n = self._enrolled.get(scope_id, 0)
            if n <= 1:
                self._enrolled.pop(scope_id, None)
            else:
                self._enrolled[scope_id] = n - 1

    def enrolled(self) -> tuple[str, ...]:
        if os.getpid() != self._pid:
            # The child's honest view of a foreign worker is empty —
            # answered without acquiring the inherited lock.
            return ()
        with self._lock:
            return tuple(sorted(self._enrolled))

    # -- drain loop ---------------------------------------------------------

    def _loop(self) -> None:
        """Bounded drain passes until stop/fork; never dies on a bad job."""
        try:
            while True:
                if os.getpid() != self._pid:
                    # Inherited loop in a forked child — must not touch
                    # inherited SQLite connections or locks.
                    self._state = "defunct_fork"  # plain attr: no inherited lock
                    return
                if self._stop.is_set():
                    return
                try:
                    progressed = self._drain_pass()
                    with self._lock:
                        self._consecutive_errors = 0
                except Exception as exc:  # infrastructure-level failure
                    self._note_error(exc)
                    if self._store_is_closed():
                        return
                    # Bounded exponential backoff on infra errors so a
                    # contended store does not spin (V5-33.09).
                    with self._lock:
                        n = self._consecutive_errors
                    backoff_s = min(
                        _ERROR_BACKOFF_CAP_MS / 1000.0,
                        self._idle_s * (2 ** min(n, 5)),
                    )
                    self._wake.wait(backoff_s)
                    self._wake.clear()
                    continue
                if self._stop.is_set():
                    return
                if progressed:
                    # Yield between busy batches — foreground writers and
                    # sibling scopes are never starved (V5-33.09).
                    if self._yield_s > 0:
                        time.sleep(self._yield_s)
                    else:
                        time.sleep(0)
                else:
                    self._wake.wait(self._idle_s)
                    self._wake.clear()
        finally:
            with self._lock:
                if self._state in ("running", "stopping"):
                    self._state = "stopped"

    def _drain_pass(self) -> bool:
        """One bounded round over enrolled namespaces; True if work ran."""
        with self._lock:
            scopes = sorted(self._enrolled)
        if not scopes:
            return False
        processed = 0
        # V6-02.08: the sources live session barriers are blocked on ride
        # this pass as an advisory unblock-first hint — fetched once per
        # pass, reaped after it once their obligations settle. An absent
        # or failing commit_notify registry reads as "nothing blocked".
        blocked = self._blocked_sources()
        for sid in scopes:
            if self._stop.is_set() or _FORK_QUIESCED:
                break
            # A drain pass is where this thread holds WAL locks and
            # sqlite mutexes — it must not straddle a fork (V5-09.10).
            with _sqlite_critical():
                report = self._ingester.drain_report(
                    scope=sid,
                    limit=self._batch_size,
                    owner=self._owner,
                    kinds=self._kinds,
                    lease_s=self._lease_s,
                    priority_sources=blocked,
                )
            self._absorb(report)
            processed += int(report.get("processed") or 0)
        if blocked:
            self._reap_barrier_sources(blocked)
        with self._lock:
            self._counters["passes"] += 1
            self._last_pass_us = now_us()
        return processed > 0

    # -- barrier-blocked source marks (V6-02.08) --------------------------

    @staticmethod
    def _commit_notify() -> Any:
        """The w6-wake per-path registry module, or None — the drain
        tolerates its absence (the blocked set simply reads as empty)."""
        try:
            from ..storage import commit_notify

            return commit_notify
        except Exception:
            return None

    def _store_path(self) -> Optional[str]:
        path = getattr(self._store, "_path", None)
        if not path:
            path = getattr(self._store, "path", None)
        return path if isinstance(path, str) and path else None

    def _blocked_sources(self) -> frozenset:
        """``commit_notify.blocked_sources(path)`` — empty frozenset on
        ANY error: the mark is advisory ordering, never correctness, so
        a broken/absent registry must never wedge the drain (V6-02.08)."""
        cn = self._commit_notify()
        fn = getattr(cn, "blocked_sources", None) if cn is not None else None
        path = self._store_path()
        if fn is None or path is None:
            return frozenset()
        try:
            marked = fn(path)
            return frozenset(marked) if marked else frozenset()
        except Exception:
            return frozenset()

    def _reap_barrier_sources(self, blocked: frozenset) -> None:
        """Drop marks whose ``source_lexical_ready`` obligations settled.

        Conservative: a source id keeps its mark while ANY live
        ``source_*`` job or outstanding ``source_lexical_ready``
        obligation still resolves to it — only ids with nothing left to
        drain are cleared, so a later barrier never inherits a stale
        hint and a still-pending id is never swept by a sibling's settle.
        ``clear_barrier_sources`` drops the whole set, so on a partial
        settle the still-owed ids are re-noted immediately. All mark
        handling is advisory — any failure keeps the marks.
        """
        try:
            with _sqlite_critical():
                still = self._blocked_still_owed(blocked)
        except Exception:
            return
        if still == set(blocked):
            return
        settled = set(blocked) - still
        cn = self._commit_notify()
        clear = getattr(cn, "clear_barrier_sources", None) if cn is not None else None
        note = getattr(cn, "note_barrier_sources", None) if cn is not None else None
        blocked_fn = getattr(cn, "blocked_sources", None) if cn is not None else None
        path = self._store_path()
        if clear is None or path is None:
            return
        try:
            keep = set(still)
            if blocked_fn is not None:
                # Marks noted between this pass's snapshot and now were
                # never evaluated — conservatively preserve them; only
                # ids this pass proved settled are allowed to clear.
                try:
                    keep |= set(blocked_fn(path)) - settled
                except Exception:
                    pass
            if keep and note is not None:
                # Partial settle: clear-all then restore only the ids
                # whose obligations are still owed (plus any marks this
                # pass never evaluated).
                clear(path)
                note(path, sorted(keep))
            elif not keep:
                clear(path)
            # keep non-empty but no re-note seam: leave the marks in
            # place rather than dropping a live barrier's hint.
        except Exception:
            pass

    def _blocked_still_owed(self, blocked: frozenset) -> set:
        """The subset of ``blocked`` whose lexical work is not terminal.

        A source still owes barrier work when either holds:

        * a live ``source_project``/``source_embed``/``source_backfill``
          job names it (``queued``/``retry_wait``/``leased`` — the
          drainer's own outstanding work); or
        * a ``source_lexical_ready`` obligation row for one of its
          receipts is still ``pending``/``running``/``deferred`` —
          deferred is owed-but-unprovisioned work, not a settle.
        """
        ids = sorted({s for s in blocked if isinstance(s, str) and s})
        if not ids:
            return set()
        owed: set = set()
        ph = ",".join("?" for _ in ids)
        with self._store.read() as conn:
            for (sid,) in conn.execute(
                "SELECT DISTINCT json_extract(input_refs_json, '$.source_id')"
                " FROM jobs"
                " WHERE kind IN"
                "   ('source_project','source_embed','source_backfill')"
                "   AND state IN ('queued','retry_wait','leased')"
                f"  AND json_extract(input_refs_json, '$.source_id') IN ({ph})",
                ids,
            ).fetchall():
                if sid:
                    owed.add(str(sid))
            remaining = [s for s in ids if s not in owed]
            if remaining and self._has_rows_table(conn, "readiness_obligations") \
                    and self._has_rows_table(conn, "source_revisions"):
                try:
                    from ..readiness import CAP_SOURCE_LEXICAL, receipt_ids_for_source
                except Exception:
                    return owed
                for sid in remaining:
                    revs = [
                        r[0]
                        for r in conn.execute(
                            "SELECT revision FROM source_revisions"
                            " WHERE source_id = ?",
                            (sid,),
                        ).fetchall()
                    ]
                    for rev in revs:
                        try:
                            rids = receipt_ids_for_source(conn, sid, int(rev))
                        except Exception:
                            continue
                        if not rids:
                            continue
                        rph = ",".join("?" for _ in rids)
                        hit = conn.execute(
                            "SELECT 1 FROM readiness_obligations"
                            " WHERE capability = ?"
                            "   AND state IN ('pending','running','deferred')"
                            f"  AND receipt_id IN ({rph}) LIMIT 1",
                            (CAP_SOURCE_LEXICAL, *rids),
                        ).fetchone()
                        if hit is not None:
                            owed.add(sid)
                            break
        return owed

    @staticmethod
    def _has_rows_table(conn: Any, name: str) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (name,),
            ).fetchone()
            is not None
        )

    def _absorb(self, report: dict[str, Any]) -> None:
        with self._lock:
            c = self._counters
            c["jobs_processed"] += int(report.get("processed") or 0)
            c["jobs_succeeded"] += int(report.get("succeeded") or 0)
            c["jobs_failed"] += int(report.get("failed") or 0)
            c["jobs_deferred"] += int(report.get("deferred") or 0)
            c["jobs_expired"] += int(report.get("expired") or 0)
            errors = report.get("errors") or ()
            if errors:
                err = errors[-1]
                self._last_error = {
                    "code": str(err.get("code") or "UNKNOWN"),
                    "kind": str(err.get("kind") or ""),
                    "at_us": now_us(),
                }

    def _note_error(self, exc: BaseException) -> None:
        code = (
            exc.code.value if isinstance(exc, VerbatimError) else type(exc).__name__
        )
        with self._lock:
            self._consecutive_errors += 1
            # Codes only — exception text can embed tenant-derived values.
            self._last_error = {"code": code, "kind": "drain", "at_us": now_us()}

    def _store_is_closed(self) -> bool:
        return bool(getattr(self._store, "_closed", False))

    # -- wakeup ---------------------------------------------------------

    def wake(self) -> None:
        """Poke the drain loop — the facade's ``add`` calls this so fresh
        work is picked up below the idle poll cadence (V5-33.08)."""
        if os.getpid() != self._pid:
            # A forked child never signals the parent's Event objects —
            # even Event.set() acquires an inherited lock (V5-09.10).
            return
        self._wake.set()

    # -- shutdown ---------------------------------------------------------

    def request_stop(self) -> None:
        """Stop intake: the loop exits after its current bounded pass —
        finishing or fencing the in-flight lease, taking no new ones."""
        if os.getpid() != self._pid:
            return  # signals are advisory; never touch inherited Events
        self._stop.set()
        self._wake.set()

    def stop(
        self,
        *,
        timeout_ms: float = 5000.0,
        scopes: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        """Bounded cooperative shutdown; honest report dict.

        Finishes the in-flight drain pass (bounded by ``batch_size`` per
        enrolled scope), joins the daemon thread within ``timeout_ms``, and
        only then closes the worker's own store connections (V5-09.08). A
        missed deadline returns ``worker_stopped=False`` with the thread
        left to finish — its leases expire durably and are reclaimable, so
        nothing is lost. A completed stop replays its report; a timed-out
        stop retries the join on the next call.

        ``scopes`` pins which namespaces the pending backlog is measured
        over — the releasing owner's namespaces — because ``_release``
        unenrolls them before this join runs and an empty enrollment would
        falsely report a drained store.
        """
        if os.getpid() != self._pid:
            return {
                "worker_stopped": False,
                "incomplete": ["forked_child"],
                "warnings": [
                    "managed worker object inherited across fork — the "
                    "parent's worker is untouched; open a fresh store in "
                    "the child"
                ],
            }
        with self._lock:
            if self._stop_report is not None:
                return self._stop_report
            if self._state in ("running", "created"):
                self._state = "stopping"
        self.request_stop()
        thread = self._thread
        joined = True
        if thread is not None:
            thread.join(max(timeout_ms, 0.0) / 1000.0)
            joined = not thread.is_alive()
        incomplete: list[str] = []
        warnings: list[str] = []
        # Pending counts are read BEFORE closing the worker's store — the
        # thread is joined (or abandoned), so the durable view is stable and
        # an honest nonzero backlog can never collapse to a fake zero. The
        # backlog is measured over ``scopes`` (the releasing owner's
        # namespaces) when given, since ``_release`` unenrolls them first.
        pending = self.pending_counts(scopes)
        if joined:
            # Connections close only after the live worker has stopped
            # using them (V5-09.08). The close is itself a SQLite
            # section — it must not straddle a fork (V5-09.10).
            try:
                with _sqlite_critical():
                    self._store.close()
            except Exception:
                warnings.append("worker store close raised — resources retained")
                incomplete.append("store_close_failed")
        else:
            incomplete.append("worker_join_timeout")
            warnings.append(
                "worker thread did not join within timeout_ms — it exits "
                "after the current bounded pass; outstanding leases expire "
                "durably and are reclaimed on next open"
            )
        with self._lock:
            if self._last_error is not None:
                warnings.append(f"last_error: {self._last_error['code']}")
        report = {
            "worker_stopped": joined,
            "incomplete": incomplete,
            "warnings": warnings,
            "pending": pending,
        }
        if joined:
            with self._lock:
                self._state = "stopped"
                self._stop_report = report
        return report

    # -- introspection ------------------------------------------------------

    def pending_counts(
        self, scopes: Optional[Iterable[str]] = None
    ) -> dict[str, int]:
        """Durable backlog over *enrolled* namespaces — or an explicit
        ``scopes`` list at close — never a store-wide count, so no other
        tenant's data leaks (V5-09.06). Counts every queued/leased durable
        job in the namespace, including kinds this worker does not lease:
        a queued ``connector_pull`` is still real pending obligation the
        close report must not hide."""
        if os.getpid() != self._pid:
            # Forked child: the durable backlog lives behind the parent's
            # connections — the honest in-process view is empty, answered
            # without touching an inherited lock or conn (V5-09.10).
            return {"jobs_pending": 0, "jobs_leased": 0, "obligations": 0}
        out = {"jobs_pending": 0, "jobs_leased": 0, "obligations": 0}
        try:
            # Store reads inside the fork-quiesce section — a reader's
            # WAL mark held across a fork is exactly the stale aLock[]
            # slot that ghosts a child's fresh connections (V5-09.10).
            with _sqlite_critical():
                try:
                    readiness = self._ingester.readiness_engine()
                except VerbatimError:
                    readiness = None
                scope_iter = self.enrolled() if scopes is None else scopes
                for sid in scope_iter:
                    stats = self._ingester.jobs.stats(sid)
                    out["jobs_pending"] += int(stats.get("pending") or 0)
                    out["jobs_leased"] += int(stats.get("leased") or 0)
                    if readiness is not None:
                        out["obligations"] += readiness.pending_count(sid)
        except Exception:
            # Inspection is best-effort; a closed/broken store reports the
            # counters it could still read rather than lying with zeros.
            pass
        return out

    def status(self) -> dict[str, Any]:
        """Health introspection for ``Memory.status()`` (V5-09.06).

        ``running`` is a thread aliveness + state check, not a proxy for
        progress — ``jobs_processed``/``last_error``/``backlog_estimate``
        carry the honest operational picture over enrolled namespaces.
        """
        if os.getpid() != self._pid:
            return {
                "mode": "managed",
                "running": False,
                "state": "forked_child",
                "forked": True,
                "detail": "worker object inherited across fork; the "
                "parent's worker owns the drain — open a fresh store",
            }
        with self._lock:
            counters = dict(self._counters)
            last_error = (
                dict(self._last_error) if self._last_error is not None else None
            )
            state = self._state
            started = self._started_us
            last_pass = self._last_pass_us
            enrolled_n = len(self._enrolled)
            refs = self._refs
        now = now_us()
        return {
            "mode": "managed",
            "running": self.running,
            "state": state,
            "owner": self._owner,
            "pid": self._pid,
            "owners": refs,
            "namespaces_enrolled": enrolled_n,
            "jobs_processed": counters["jobs_processed"],
            "jobs_succeeded": counters["jobs_succeeded"],
            "jobs_failed": counters["jobs_failed"],
            "jobs_deferred": counters["jobs_deferred"],
            "jobs_expired": counters["jobs_expired"],
            "passes": counters["passes"],
            "last_error": last_error,
            "backlog_estimate": self.pending_counts(),
            "uptime_ms": round((now - started) / 1000.0, 3) if started else 0.0,
            "last_pass_age_ms": (
                round((now - last_pass) / 1000.0, 3) if last_pass else None
            ),
            "batch_size": self._batch_size,
            "idle_ms": round(self._idle_s * 1000.0, 3),
            "lease_s": self._lease_s,
            "kinds": len(self._kinds),
        }

    def __del__(self) -> None:  # best-effort only (V5-09.09)
        try:
            if (
                os.getpid() == self._pid
                and getattr(self, "_state", None) == "running"
            ):
                self.stop(timeout_ms=_FINALIZER_TIMEOUT_MS)
        except Exception:
            pass


# ------------------------------------------------------------------
# process-level registry — ref-counted per resolved store identity
# ------------------------------------------------------------------

_REGISTRY: dict[str, ManagedWorker] = {}
_REGISTRY_PID = os.getpid()
# RLock, not Lock: a GC finalizer (``Memory.__del__`` → ``close`` →
# ``release`` → ``_release``) can fire on a thread already inside this
# guard — e.g. while ``acquire`` constructs a ManagedWorker under it —
# and a plain Lock self-deadlocks the registry (V6/F24 observed hang).
_REGISTRY_LOCK = threading.RLock()
_REGISTRY_SEQ = [0]


def _reset_registry_after_fork() -> None:
    """After-fork child reset: the parent's worker objects, threads,
    (possibly mid-lock) registry mutex, and the fork-quiesce locks are
    all unusable here — rebuild the registry and quiesce state from
    scratch so a child can only ever start a *fresh* worker on a
    *freshly opened* store, and its fresh workers never see the
    inherited ``_FORK_QUIESCED`` snapshot (V5-09.10)."""
    global _REGISTRY, _REGISTRY_LOCK, _REGISTRY_PID
    global _FORK_LOCK, _FORK_ZERO, _FORK_QUIESCED, _FORK_IN_TX
    _REGISTRY = {}
    _REGISTRY_LOCK = threading.RLock()
    _REGISTRY_PID = os.getpid()
    _FORK_LOCK = threading.RLock()
    _FORK_ZERO = threading.Condition(_FORK_LOCK)
    _FORK_QUIESCED = False
    _FORK_IN_TX = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_quiesce_workers_before_fork,
        after_in_parent=_resume_workers_after_fork_parent,
        after_in_child=_reset_registry_after_fork,
    )


def _registry_guard() -> threading.RLock:
    """PID-checked registry lock — defensive reset if a fork path skipped
    the atfork hooks (e.g. a raw clone): never block on an inherited lock
    that a dead thread may hold. Reentrant: GC finalizers may re-enter on
    the holding thread (``Memory.__del__`` → release)."""
    if os.getpid() != _REGISTRY_PID:
        _reset_registry_after_fork()
    return _REGISTRY_LOCK


def registry_size() -> int:
    """Live registered workers (diagnostics/tests)."""
    with _registry_guard():
        return len(_REGISTRY)


def active_workers() -> list[dict[str, Any]]:
    """Per-store worker summaries for diagnostics — keys and counts only."""
    with _registry_guard():
        items = [(k, w.owner, w._refs, w.enrolled()) for k, w in _REGISTRY.items()]
    return [
        {
            "store_key": key,
            "owner": owner,
            "owners": refs,
            "namespaces_enrolled": len(enrolled),
        }
        for key, owner, refs, enrolled in items
    ]


def _reset_registry() -> None:
    """Stop and clear every registered worker — test cleanup + atexit."""
    with _registry_guard():
        workers = list(_REGISTRY.values())
        _REGISTRY.clear()
    for w in workers:
        try:
            w.stop(timeout_ms=_FINALIZER_TIMEOUT_MS)
        except Exception:
            pass


def acquire(
    store: Store,
    cfg: VerbatimConfig,
    *,
    scope_ids: Iterable[str] = (),
    kinds: Optional[Iterable[Any]] = None,
    extra_kinds: Iterable[Any] = (),
    batch_size: int = DEFAULT_BATCH_SIZE,
    idle_ms: float = DEFAULT_IDLE_MS,
    yield_ms: float = DEFAULT_YIELD_MS,
    lease_s: float = DEFAULT_LEASE_S,
    spawn_thread: Optional[Any] = None,
    encoder: Any = None,
    judge: Any = None,
    transport_broker: Any = None,
) -> "WorkerHandle":
    """Attach a managed-worker owner to ``store``; returns a handle.

    Multiple handles on one resolved store share a single
    :class:`ManagedWorker` (V5-09.02). The first acquirer's tuning
    (``batch_size``/``idle_ms``/kind set/provisions) configures the shared
    worker — later acquirers only add enrollment + a refcount. ``scope_ids``
    enroll the caller's namespaces for leasing (V5-09.13); a worker with no
    enrolled scopes idles honestly and processes nothing.
    """
    path = getattr(store, "path", None)
    if not isinstance(path, str) or not path:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "managed worker requires a file-backed Store (path + key_path)",
        )
    if getattr(store, "readonly", False):
        raise VerbatimError(
            ErrorCode.CONFIG_INVALID,
            "managed worker requires a writable store; use worker='external' "
            "for read-only attachments",
        )
    sids = tuple(dict.fromkeys(require_id(s, "scope_id") for s in scope_ids))
    key = store_identity(store)
    with _registry_guard():
        while True:
            worker = _REGISTRY.get(key)
            if worker is not None and worker._pid != os.getpid():
                # A registry entry is owned by the PID that created it — a
                # foreign (inherited) entry is dropped untouched: the child
                # must never operate on the parent's worker object, and
                # ``_registry_guard`` has already rebuilt the table under
                # this PID, so this is belt-and-suspenders for raw-clone
                # paths that bypass the atfork hooks (V5-09.10).
                _REGISTRY.pop(key, None)
                worker = None
            if worker is not None and not worker.running:
                # A worker whose thread died (or whose store closed under
                # it) is dropped from the registry; a fresh worker takes
                # its place under normal lease fencing.
                _REGISTRY.pop(key, None)
                worker = None
            if worker is None:
                _REGISTRY_SEQ[0] += 1
                worker = ManagedWorker(
                    store,
                    cfg,
                    kinds=kinds,
                    extra_kinds=extra_kinds,
                    batch_size=batch_size,
                    idle_ms=idle_ms,
                    yield_ms=yield_ms,
                    lease_s=lease_s,
                    spawn_thread=spawn_thread,
                    encoder=encoder,
                    judge=judge,
                    transport_broker=transport_broker,
                    generation=_REGISTRY_SEQ[0],
                )
                _REGISTRY[key] = worker
                try:
                    worker.start()
                except BaseException:
                    _REGISTRY.pop(key, None)
                    try:
                        worker._store.close()
                    except Exception:
                        pass
                    raise
            with worker._lock:
                worker._refs += 1
                # Validate AFTER the refcount bump: a GC finalizer on
                # this thread (``Memory.__del__`` → ``release``) may have
                # re-entered the registry inside the increment and
                # popped/stopped this worker — never hand a new owner to
                # a worker that is no longer registered or running.
                adopted = worker.running and _REGISTRY.get(key) is worker
                if adopted:
                    for sid in sids:
                        worker.enroll(sid)
                    # Re-validate again after enrollment — the same
                    # finalizer window exists inside ``enroll``'s dict
                    # writes.
                    adopted = worker.running and _REGISTRY.get(key) is worker
                if not adopted:
                    worker._refs -= 1
                    for sid in sids:
                        worker.leave(sid)
            if adopted:
                break
            # Loop: the registry slot is now empty or holds a different
            # live worker — re-resolve and adopt that one.
    worker.wake()
    return WorkerHandle(key, worker, scope_ids=sids)


def _close_report(report: dict[str, Any]) -> CloseReport:
    """Map a worker stop/detach report onto the frozen CloseReport shape."""
    pending = report.get("pending") or {}
    jobs_pending = int(pending.get("jobs_pending") or 0) + int(
        pending.get("jobs_leased") or 0
    )
    obligations = int(pending.get("obligations") or 0)
    incomplete = list(report.get("incomplete") or [])
    if jobs_pending:
        incomplete.append("pending_jobs")
    if obligations:
        incomplete.append("pending_obligations")
    return CloseReport(
        closed=True,
        drained=(jobs_pending == 0),
        worker_stopped=bool(report.get("worker_stopped")),
        pending_obligations=obligations or jobs_pending,
        incomplete=incomplete,
        warnings=list(report.get("warnings") or []),
    )


def _release(
    key: str,
    worker: ManagedWorker,
    scope_ids: tuple[str, ...],
    timeout_ms: float,
) -> CloseReport:
    """Drop one owner; the last owner performs bounded worker shutdown."""
    with _registry_guard():
        with worker._lock:
            worker._refs -= 1
            last = worker._refs <= 0
            remaining = worker._refs
        for sid in scope_ids:
            worker.leave(sid)
        if last:
            # Pop BEFORE the slow join: a concurrent acquire must see a
            # fresh worker, never revive one that is stopping (V5-09.07).
            _REGISTRY.pop(key, None)
            worker.request_stop()
    if not last:
        return _close_report(
            {
                "worker_stopped": False,
                "incomplete": [],
                "warnings": [
                    f"shared worker remains running for {remaining} "
                    "owner(s); this facade's detach is complete"
                ],
                "pending": worker.pending_counts(),
            }
        )
    # The backlog is measured over this owner's namespaces — ``leave`` above
    # already unenrolled them, so counting enrolled() here would read an
    # empty set and falsely report a drained store (V5-09.07).
    return _close_report(worker.stop(timeout_ms=timeout_ms, scopes=scope_ids))


class WorkerHandle:
    """One facade's ref-counted ownership of a :class:`ManagedWorker`.

    ``release()`` detaches this owner (idempotent, bounded by
    ``timeout_ms``) and returns an honest :class:`CloseReport` —
    ``worker_stopped``/``pending_obligations``/``incomplete`` reflect what
    actually happened, never a promise to empty an unbounded store-wide
    queue (V5-09.07/08). ``wake()`` pokes the shared drain loop after an
    ``add``. All methods are fork-safe: in a forked child they answer
    honestly without acquiring inherited locks or touching inherited
    threads/connections.
    """

    def __init__(
        self,
        key: str,
        worker: ManagedWorker,
        *,
        scope_ids: tuple[str, ...] = (),
    ) -> None:
        self._key = key
        self._worker = worker
        self._scope_ids = scope_ids
        self._pid = os.getpid()
        self._hdl_lock = threading.Lock()
        self._released = False
        self._report: Optional[CloseReport] = None

    @property
    def worker(self) -> ManagedWorker:
        """The shared worker — introspection/enrollment only; never owned."""
        return self._worker

    @property
    def closed(self) -> bool:
        return self._released

    def _forked(self) -> bool:
        return os.getpid() != self._pid

    def wake(self) -> None:
        if self._forked() or self._released:
            return
        self._worker.wake()

    def status(self) -> dict[str, Any]:
        if self._forked():
            return {
                "mode": "managed",
                "running": False,
                "state": "forked_child",
                "forked": True,
                "detail": "handle inherited across fork — open a fresh "
                "Memory in the child (V5-09.10)",
            }
        out = self._worker.status()
        out["handle_closed"] = self._released
        return out

    def release(self, *, timeout_ms: float = 5000.0) -> CloseReport:
        """Detach this owner; bounded and idempotent.

        The last owner stops new leasing and joins the worker thread within
        ``timeout_ms``. Repeated calls replay the first report (V5-09.09).
        """
        if self._forked():
            # The child must not touch the inherited lock, refcount, or
            # thread object — the parent's worker is unaffected; this
            # handle is simply detached in the child's address space.
            return CloseReport(
                closed=True,
                drained=False,
                worker_stopped=False,
                pending_obligations=0,
                incomplete=["forked_child"],
                warnings=[
                    "worker handle inherited across fork — detached in the "
                    "child only; the parent's worker keeps running"
                ],
            )
        with self._hdl_lock:
            if self._released:
                prev = self._report
                if (
                    prev is not None
                    and not prev.worker_stopped
                    and "worker_join_timeout" in prev.incomplete
                ):
                    # The first release missed its join deadline — a
                    # repeated close retries the join with a fresh budget
                    # and reports the new truth, not a stale replay. The
                    # backlog is still measured over this owner's
                    # (already-unenrolled) namespaces.
                    self._report = _close_report(
                        self._worker.stop(
                            timeout_ms=timeout_ms, scopes=self._scope_ids
                        )
                    )
                return self._report
            self._released = True
            self._report = _release(
                self._key, self._worker, self._scope_ids, timeout_ms
            )
            return self._report


def external_status() -> dict[str, Any]:
    """Honest ``status()`` worker block for ``worker='external'``
    (V5-09.03): no helper runs; an external worker/embedding application
    owns progress and pending obligations simply wait for it."""
    return {
        "mode": "external",
        "running": False,
        "state": "external",
        "detail": "external worker owns progress; pending obligations "
        "wait for that host to drain",
    }


def _atexit_shutdown() -> None:
    try:
        _reset_registry()
    except Exception:
        pass


atexit.register(_atexit_shutdown)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_IDLE_MS",
    "DEFAULT_YIELD_MS",
    "MANAGED_JOB_KINDS",
    "ManagedWorker",
    "WorkerHandle",
    "acquire",
    "active_workers",
    "external_status",
    "registry_size",
    "store_identity",
]
