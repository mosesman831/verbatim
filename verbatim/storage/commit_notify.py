"""Cross-store commit notification registry (SPEC_V6 §02, V6-02.07).

A ``Store`` signals ``self._commit_cond`` after every successful COMMIT —
but that condition is per *object*. The managed worker drains jobs on a
second ``Store`` opened on the same database file, so its commits never
wake readiness barriers blocked on the facade's ``Store._commit_cond``;
each barrier then burns a full poll interval of dead time (the measured
A1 miss).

This module is the process-local fix: a ``{abspath: _PathSignal}``
registry keyed by canonical database path. ``Store`` calls
:func:`commit_fired` from the same post-commit block that notifies
``_commit_cond``; :func:`wait` blocks on the *path's* condition and wakes
on a commit from ANY ``Store`` object opened on that file in this
process — strictly broader than any single store's condition.

Design notes:

- Locking: one ``threading.Condition`` per path guards that path's
  generation counter, waiter count, and blocked-source set together —
  a waiter and a committer on the same path serialize on the same
  mutex, so a commit can never be lost between a waiter's snapshot and
  its ``cond.wait``. A single module ``_GUARD`` protects only registry
  insertion/lookup and the fork-pid check; it is never held while
  blocking, so independent paths never contend.
- Advisory only: a missed or spurious wake costs at most one poll —
  callers always re-read and re-verify obligations. Correctness MUST
  NOT depend on wakeup delivery (V6-02.07).
- Fork-safe (V5-09.10 pattern, same as ``facade._init_lock`` /
  ``worker._reset_registry_after_fork``): a forked child must not
  inherit condition objects the parent's dead threads may have held.
  ``os.register_at_fork(after_in_child=...)`` rebuilds the registry and
  guard wholesale; a pid check ahead of every access is the defensive
  fallback for fork paths that skip the hooks (e.g. raw clone).
- No files, sockets, or threads are created — pure stdlib,
  process-local only.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, Iterable, Optional, Tuple

__all__ = [
    "register",
    "commit_fired",
    "wait",
    "subscriber_count",
    "note_barrier_sources",
    "blocked_sources",
    "clear_barrier_sources",
]


class _PathSignal:
    """Per-path commit fanout: one condition, one monotonic generation.

    ``generation`` increments on every observed COMMIT for the path —
    waiters snapshot it at entry and wake when it advances, which makes
    wakeups immune to notify/listen races (a commit between the snapshot
    and ``cond.wait`` still advances the counter the loop checks).
    ``waiters`` is the live ``wait()`` count for diagnostics;
    ``blocked`` is the w6-wake → w6-drain handoff set: source ids whose
    ``source_lexical_ready`` obligations are awaited by live barriers
    (v6_contracts §2 — marked here, consumed by the drainer's
    unblock-first pass; advisory, never correctness).
    """

    __slots__ = ("cond", "generation", "waiters", "blocked")

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.generation = 0
        self.waiters = 0
        self.blocked: set[str] = set()


_SIGNALS: Dict[str, _PathSignal] = {}
_GUARD = threading.Lock()
_REGISTRY_PID = os.getpid()


def _reset_after_fork() -> None:
    """After-fork child reset: rebuild registry + guard wholesale.

    The parent's ``_PathSignal`` conditions (and the ``_GUARD`` mutex
    itself) may be held by threads that no longer exist in the child —
    the child inherits a poisoned snapshot. Fresh objects only; a child
    re-registers on first use (V5-09.10).
    """
    global _SIGNALS, _GUARD, _REGISTRY_PID
    _SIGNALS = {}
    _GUARD = threading.Lock()
    _REGISTRY_PID = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _lookup(path: str, *, create: bool) -> Tuple[str, Optional[_PathSignal]]:
    """Resolve ``path`` to its canonical key + signal (creating on demand).

    The pid check runs BEFORE the guard is touched: an inherited guard
    mutex may be held by a dead thread, so the fork check itself must
    never block on it.
    """
    if os.getpid() != _REGISTRY_PID:
        _reset_after_fork()
    key = os.path.abspath(path)
    with _GUARD:
        sig = _SIGNALS.get(key)
        if sig is None and create:
            sig = _SIGNALS[key] = _PathSignal()
    return key, sig


def register(path: str) -> str:
    """Register ``path`` and return its canonical registry key (abspath).

    Idempotent; called by ``Store`` construction/open so diagnostics can
    enumerate known paths. Cheap enough to call on every operation.
    """
    key, _ = _lookup(path, create=True)
    return key


def commit_fired(path: str) -> None:
    """Signal that a COMMIT landed on ``path`` — called by ``Store`` only.

    Bumps the path's generation and wakes every waiter. Never raises
    into the commit path: this signal is advisory, so the caller wraps
    it in the same ``try/except: pass`` guard as ``_commit_cond``.
    """
    _, sig = _lookup(path, create=True)
    assert sig is not None
    with sig.cond:
        sig.generation += 1
        sig.cond.notify_all()


def wait(path: str, timeout_s: float) -> bool:
    """Block up to ``timeout_s`` for a commit on ``path``.

    Returns ``True`` when the path's generation advanced past the
    snapshot taken at entry — i.e. a commit on ANY ``Store`` object for
    this file fired since the wait began. ``False`` on timeout.
    Spurious wakeups are fine (callers re-verify); ``timeout_s <= 0``
    polls once without blocking.
    """
    _, sig = _lookup(path, create=True)
    assert sig is not None
    # Snapshot BEFORE taking the condition: a commit that lands between
    # this read and ``cond.wait`` still advanced the counter the loop
    # checks, so it can never be slept through.
    gen0 = sig.generation
    deadline = time.monotonic() + max(timeout_s, 0.0)
    with sig.cond:
        sig.waiters += 1
        try:
            while sig.generation == gen0:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                sig.cond.wait(remaining)
            return True
        finally:
            sig.waiters -= 1


def subscriber_count(path: str) -> int:
    """Live ``wait()`` callers on ``path`` — diagnostics only."""
    _, sig = _lookup(path, create=False)
    if sig is None:
        return 0
    with sig.cond:
        return sig.waiters


# ---------------------------------------------------------------------
# Barrier-blocked source marking (v6_contracts §2)
#
# ``facade.search`` notes the source ids behind pending barrier receipts;
# the managed drainer reads ``blocked_sources`` for its unblock-first
# pass and calls ``clear_barrier_sources`` once they settle. The marks
# are advisory ordering hints only — dequeue rules, fencing, and lane
# priorities are unchanged.
# ---------------------------------------------------------------------


def note_barrier_sources(path: str, source_ids: Iterable[str]) -> None:
    """Union ``source_ids`` into ``path``'s barrier-blocked set."""
    _, sig = _lookup(path, create=True)
    assert sig is not None
    with sig.cond:
        sig.blocked.update(source_ids)


def blocked_sources(path: str) -> frozenset:
    """Snapshot of the source ids currently blocking live barriers."""
    _, sig = _lookup(path, create=False)
    if sig is None:
        return frozenset()
    with sig.cond:
        return frozenset(sig.blocked)


def clear_barrier_sources(path: str) -> None:
    """Clear ``path``'s blocked-source set (only that path's)."""
    _, sig = _lookup(path, create=False)
    if sig is None:
        return
    with sig.cond:
        sig.blocked.clear()
