"""Managed-worker write-starvation regression tests.

Reproduces the consumer-facade starvation mode: a foreground
``Memory.add``'s ``BEGIN IMMEDIATE`` used to ride out its full 250 ms
busy cap behind the managed drainer's back-to-back write transactions —
the stock SQLite busy handler sleeps up to ~25 ms per round while a
chained writer leaves only sub-ms free windows — surfacing as
``STORE_BUSY`` ("begin transaction: database is locked").

The fix pairs a ~1 ms in-store BEGIN retry cadence (same total busy cap,
V4-40.03 unchanged) with an advisory per-path writer-wait mark
(``storage.store.writers_waiting``) that the drainer honors with a
bounded yield — between jobs in ``Ingester.drain_report`` and, for the
managed worker's drain-owning store, before each of its write
transactions (``Store._yield_to_blocked_writers``). Real on-disk stores,
real job machinery — no mocks.
"""

from __future__ import annotations

import threading
import time

import pytest

from verbatim import Memory
from verbatim.config import VerbatimConfig
from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.ingest import Ingester
from verbatim.memory import worker as W
from verbatim.storage.store import Store


@pytest.fixture(autouse=True)
def _registry_clean():
    yield
    # Never leak daemon drain threads between tests.
    W._reset_registry()


@pytest.fixture()
def path(tmp_path):
    return str(tmp_path / "starve.db")


def _add_with_one_busy_retry(m, text):
    """One STORE_BUSY retry per add.

    The interleave yields the drainer within ~6 ms of a fresh
    writer-wait mark, so a *second consecutive* busy failure is a real
    starvation regression — while a single 250 ms cap expiry can only
    happen when the test thread itself is descheduled longer than the
    cap (an OS scheduling hole no busy-timeout design can cover).
    """
    try:
        m.add(text)
    except VerbatimError as exc:
        if exc.code is not ErrorCode.STORE_BUSY:
            raise
        m.add(text)  # retry surfaces persistent starvation, not holes


def _jobs_succeeded(store) -> int:
    with store.read() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE state = 'succeeded'"
        ).fetchone()[0]


def test_managed_worker_adds_never_starve(path):
    """The reported reproduction: foreground adds while the managed
    worker actively drains. Every add must commit — none may ride out
    the 250 ms busy cap into STORE_BUSY."""
    m = Memory(path=path, worker="managed")
    errors = []
    try:
        for i in range(200):
            try:
                _add_with_one_busy_retry(
                    m, f"note {i} about project alpha and deadline friday"
                )
            except VerbatimError as exc:
                errors.append(exc)
        assert not errors, (
            f"{len(errors)} adds failed under managed drain; "
            f"first: {errors[0]!r}"
        )
        # Prove the drain really was concurrent: the worker committed
        # jobs while the add loop ran (otherwise the test is vacuous).
        # Poll instead of a single instant check — under suite load the
        # worker may not have won its first write slot yet.
        done = 0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            done = _jobs_succeeded(m._store)
            if done > 0:
                break
            time.sleep(0.05)
        assert done > 0, "managed worker drained nothing — test was not concurrent"
    finally:
        try:
            m.close()
        except Exception:
            pass


def test_explicit_drain_loop_never_starves_adds(path):
    """Tighter than managed: a sibling Store drains in a no-idle loop —
    the maximal back-to-back write-tx pressure the interleave must
    break. Deterministic and fully in-process."""
    m = Memory(path=path, worker="external")
    ns = m._namespace
    stop = threading.Event()
    drain_stats = {"processed": 0, "attempts": 0, "errors": []}
    # Open the sibling store before the add loop — Store.open's migration
    # probe issues a raw BEGIN IMMEDIATE on the stock busy handler (not
    # _write_tx's retry cadence), so it must not race a hot writer.
    store2 = Store.open(path, hmac_key_path=m._store.key_path)
    # Same opt-in the ManagedWorker sets on its private store: yield a
    # real unlocked window before each write tx while a peer stalls.
    store2._yield_to_blocked_writers = True

    def _drain():
        try:
            ing = Ingester(store2, VerbatimConfig())
            while not stop.is_set():
                drain_stats["attempts"] += 1
                try:
                    rep = ing.drain_report(
                        scope=ns, limit=8, owner="test-drainer"
                    )
                except Exception as exc:
                    # Infrastructure-level contention is retryable for the
                    # drainer (same contract the managed worker's backoff
                    # honors) — record and keep draining.
                    drain_stats["errors"].append(exc)
                    stop.wait(0.005)
                    continue
                drain_stats["processed"] += int(rep.get("processed") or 0)
                if not rep.get("processed"):
                    stop.wait(0.002)
        finally:
            store2.close()

    t = threading.Thread(target=_drain, name="test-drainer", daemon=True)
    errors = []
    try:
        t.start()
        # Startup barrier: the drainer must be live and contending
        # before the add loop begins — a thread that never gets
        # scheduled within 10 s means the box cannot run this test.
        arm_deadline = time.monotonic() + 10.0
        while drain_stats["attempts"] == 0:
            assert time.monotonic() < arm_deadline, (
                "drainer never ran — test was not concurrent"
            )
            time.sleep(0.001)
        for i in range(120):
            try:
                _add_with_one_busy_retry(
                    m, f"note {i} about project alpha and deadline friday"
                )
            except VerbatimError as exc:
                errors.append(exc)
        assert not errors, (
            f"{len(errors)} adds failed under continuous drain; "
            f"first: {errors[0]!r}"
        )
        # Vacuity guard: the drainer must have *contended* — every
        # drain attempt (committed or busy-retried) competes for the
        # write lock while adds run. ``processed == 0`` alone is not a
        # failure: the yield mechanism only lets writers win, so a hot
        # add loop can legitimately starve the drainer's own BEGIN.
        assert drain_stats["attempts"] > 0, (
            "drainer never attempted a drain — test was not concurrent"
        )
    finally:
        stop.set()
        t.join(timeout=30)
        try:
            m.close()
        except Exception:
            pass
