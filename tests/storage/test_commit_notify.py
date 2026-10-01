"""Cross-store commit notification tests (SPEC_V6 §02, V6-02.07).

A commit on ANY ``Store`` object opened on the same database file must
wake barriers waiting on another ``Store`` object in this process — the
managed worker drains on a second ``Store``, and its commits used to
notify only that object's ``_commit_cond``, burning a poll ramp of dead
time per wait. ``commit_notify`` is the per-path fanout; ``wait_ready``
waits on it instead of the per-object condition.

Fixtures use real ``Store.create``/``Store.open`` databases — the
conftest TestStore shim has no file path and no ``_commit_cond``.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from typing import Optional

import pytest

from verbatim.config import VerbatimConfig
from verbatim.core.time import now_us
from verbatim.core.types import (
    Provenance,
    Scope,
    SourceEnvelope,
    SourceKind,
)
from verbatim.ingest import Ingester
from verbatim.readiness import ReadinessEngine, ingest_receipt_id
from verbatim.storage import commit_notify
from verbatim.storage.store import Store


SCOPE = Scope(profile_id="p", principal_id="alice", conversation_id="c1")


def _env(text: str, scope: Scope = SCOPE) -> SourceEnvelope:
    return SourceEnvelope(
        origin="test",
        source_kind=SourceKind.USER_MESSAGE,
        scope=scope,
        speaker_id=scope.principal_id,
        payload=text.encode("utf-8"),
        event_us=now_us(),
        captured_us=now_us(),
        provenance=Provenance.DIRECT_USER,
    )


@pytest.fixture
def cfg() -> VerbatimConfig:
    c = VerbatimConfig()
    c = replace(c, capture=replace(c.capture, enabled=True))
    c = replace(c, admission=replace(c.admission, require_review=False))
    return c


def _commit(store: Store, key: str) -> None:
    """One real COMMIT on ``store`` — a meta upsert is the cheapest
    durable write the schema allows."""
    with store.tx() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value_json) VALUES(?, '1')",
            (key,),
        )


def _await_subscriber(path: str, timeout_s: float = 2.0) -> None:
    """Spin until a ``wait()`` caller registers on ``path`` — removes the
    start-the-thread vs fire-the-commit race from latency assertions."""
    deadline = time.monotonic() + timeout_s
    while commit_notify.subscriber_count(path) < 1:
        assert time.monotonic() < deadline, "barrier never entered wait()"
        time.sleep(0.005)


# ---------------------------------------------------------------------
# V6-02.07 — cross-store commit wake
# ---------------------------------------------------------------------


def test_wait_wakes_on_commit_from_other_store(tmp_path):
    """A barrier thread blocked in ``commit_notify.wait(path)`` wakes
    promptly when a DIFFERENT ``Store`` object on the same file commits —
    well under the ~15ms poll ramp the registry replaces."""
    path = str(tmp_path / "wake.db")
    store_a = Store.create(path)
    store_b = Store.open(path, hmac_key_path=store_a.key_path)
    try:
        key = commit_notify.register(store_a.path)
        assert key == os.path.abspath(path)
        assert store_b.path == key  # same canonical registry key

        woke: dict = {}
        barrier = threading.Thread(
            target=lambda: woke.update(
                t=time.monotonic(),
                ok=commit_notify.wait(store_a.path, 5.0),
            ),
            daemon=True,
        )
        barrier.start()
        _await_subscriber(store_a.path)

        _commit(store_b, "cn_probe_b")
        fired_at = time.monotonic()
        barrier.join(timeout=2.0)
        assert not barrier.is_alive(), "wait() never woke on B's commit"
        assert woke["ok"] is True
        # Wake latency from the commit, not from thread start.
        assert woke["t"] - fired_at < 0.05
    finally:
        store_a.close()
        store_b.close()


def test_wait_wakes_on_same_store_commit(store):
    """The registry is strictly broader than ``_commit_cond``: commits on
    the SAME Store object also fire it (readiness waits on the path
    alone — a local commit must never be missed)."""
    woke: dict = {}
    barrier = threading.Thread(
        target=lambda: woke.update(
            ok=commit_notify.wait(store.path, 5.0)
        ),
        daemon=True,
    )
    barrier.start()
    _await_subscriber(store.path)
    _commit(store, "cn_probe_same")
    barrier.join(timeout=2.0)
    assert not barrier.is_alive()
    assert woke["ok"] is True


def test_wait_times_out_without_commit(store):
    """No commit → bounded wait returns False after the timeout."""
    t0 = time.monotonic()
    assert commit_notify.wait(store.path, 0.05) is False
    assert time.monotonic() - t0 >= 0.05


def test_wait_ignores_commits_before_entry(store):
    """A commit that fired BEFORE ``wait`` began is not observed — the
    generation snapshot is taken at entry, so stale commits cannot
    produce a phantom wake."""
    _commit(store, "cn_probe_stale")
    assert commit_notify.wait(store.path, 0.02) is False


# ---------------------------------------------------------------------
# v6_contracts §2 — barrier-blocked source marks
# ---------------------------------------------------------------------


def test_barrier_sources_roundtrip_and_path_isolation(tmp_path):
    path_a = str(tmp_path / "a.db")
    path_b = str(tmp_path / "b.db")

    commit_notify.note_barrier_sources(path_a, ["s1", "s2"])
    commit_notify.note_barrier_sources(path_a, ["s2", "s3"])
    commit_notify.note_barrier_sources(path_b, ["other"])

    assert commit_notify.blocked_sources(path_a) == frozenset(
        {"s1", "s2", "s3"}
    )
    assert commit_notify.blocked_sources(path_b) == frozenset({"other"})

    commit_notify.clear_barrier_sources(path_a)
    assert commit_notify.blocked_sources(path_a) == frozenset()
    # clear removes only that path's set.
    assert commit_notify.blocked_sources(path_b) == frozenset({"other"})

    # Unknown path reads empty, never creates visible state.
    assert (
        commit_notify.blocked_sources(str(tmp_path / "never.db"))
        == frozenset()
    )


# ---------------------------------------------------------------------
# Fork safety — child inherits no registry state
# ---------------------------------------------------------------------


def test_fork_child_gets_fresh_registry(store):
    """V5-09.10 pattern: the child must not inherit the parent's signal
    objects (conditions a dead thread may hold) nor its blocked-source
    marks — the registry rebuilds and stays usable (same idiom as
    tests/v5/test_scenarios_a.py::test_e20_fork_safety)."""
    if not hasattr(os, "fork"):
        pytest.skip("posix fork required")

    commit_notify.note_barrier_sources(store.path, ["held"])
    commit_notify.commit_fired(store.path)

    pid = os.fork()
    if pid == 0:  # child
        rc = 0
        try:
            if commit_notify.blocked_sources(store.path):
                rc = 41  # inherited the parent's marks — violation
            else:
                # The rebuilt registry is usable: wait + commit round-
                # trip inside the child without touching parent state.
                fired: list = []
                t = threading.Thread(
                    target=lambda: fired.append(
                        commit_notify.wait(store.path, 1.0)
                    )
                )
                t.start()
                time.sleep(0.02)
                commit_notify.commit_fired(store.path)
                t.join(1.0)
                if t.is_alive() or fired != [True]:
                    rc = 42
        except BaseException:
            rc = 43
        finally:
            os._exit(rc)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0, (
        f"fork contract violated (child exit {os.WEXITSTATUS(status)})"
    )
    # Parent state is untouched by the child's rebuilt registry.
    assert commit_notify.blocked_sources(store.path) == frozenset({"held"})
    commit_notify.clear_barrier_sources(store.path)


# ---------------------------------------------------------------------
# Readiness integration — barrier on store A, drain on store B
# ---------------------------------------------------------------------


def test_wait_ready_unblocks_on_cross_store_drain(tmp_path, cfg):
    """A real ``ReadinessEngine`` barrier on store A returns promptly
    when the receipt's obligations settle via a drain on store B — the
    worker topology this exists for. B's job COMMITs fire the shared
    path registry; A's own ``_commit_cond`` is never touched."""
    path = str(tmp_path / "xstore.db")
    store_a = Store.create(path)
    store_b = Store.open(path, hmac_key_path=store_a.key_path)
    result: dict = {}
    barrier: Optional[threading.Thread] = None
    try:
        ing_a = Ingester(store_a, cfg)
        r = ing_a.ingest(_env("cross-store drain fact"))
        rid = ingest_receipt_id(r.accepted[0], 1)
        eng = ReadinessEngine(store_a)

        barrier = threading.Thread(
            target=lambda: result.update(
                t=time.monotonic(),
                snap=eng.wait_ready(
                    rid, deadline_us=now_us() + 10_000_000
                ),
            ),
            daemon=True,
        )
        barrier.start()
        _await_subscriber(store_a.path)

        # Drain the receipt's jobs on the SECOND Store — every job
        # COMMIT on B's writer fires the path registry A waits on.
        rep = Ingester(store_b, cfg).drain_report(scope=None, limit=64)
        assert rep["complete"] is True
        drained_at = time.monotonic()

        barrier.join(timeout=5.0)
        assert not barrier.is_alive(), "barrier never returned"
        snap = result["snap"]
        assert snap["ready"] is True and snap["complete"] is True
        # The barrier re-polls and returns within a fraction of one old
        # blind-poll cycle of the last settling commit — not the ~15ms
        # ramp, and nowhere near the 10s deadline.
        assert result["t"] - drained_at < 0.2
    finally:
        if barrier is not None:
            barrier.join(timeout=5.0)
        store_a.close()
        store_b.close()
