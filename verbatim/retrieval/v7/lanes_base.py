"""V7 lane registry + per-lane call wrapper (§04.2 S2, V7-05.01–09).

Ownership: w-pipeline (``docs/v7_contracts.md``). Lane modules —
``lexical.py``, ``fuzzy.py``, ``temporal.py``, ``dense.py``, ``graph.py``
etc., each owned by its own worker — self-register at import time via
:func:`register_lane`. This module never imports them: an enabled lane
with no registered implementation reports ``unavailable`` and the
pipeline continues (that is how wave A integrates incrementally).

Contract:

- ``LANE_REGISTRY`` maps :class:`LaneName` -> lane callable
  ``(ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice)
  -> LaneOutput``. Registering a :class:`LaneV7` instance or subclass
  also works — its ``.run`` method is invoked.
- :func:`run_one` is the only way the pipeline invokes a lane. It
  enforces the slice (a lane that overruns ``slice.deadline_ms`` cannot
  report ``ok`` through the orchestrator — it is downgraded to
  ``partial``/``deadline_overrun``) and the pool cap (overflow
  candidates are truncated, the drop recorded in
  ``stats["cap_truncated"]``).
- Fail closed: a ``VerbatimError`` carrying an authorization/integrity
  code (``FAIL_CLOSED_CODES``) propagates out of :func:`run_one` —
  retrieval never silently absorbs an authz/integrity failure
  (V3-28.12 carried). Every other exception becomes
  ``LaneOutput(status=unavailable, reason=repr(exc)[:200])``.

Sequential-execution note (carried from v3, V4-27.07): lanes run one at
a time on the caller's single pinned read snapshot. Parallel workers
would each need an independent snapshot that can diverge from
``ctx.generation`` mid-run, and the graph lane deliberately consumes
seeds accumulated from earlier lanes (§32.6: top-10 of L-ent ∪ L-lex,
propagated via ``ctx.manifest["seeds"]``).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from ...core.types import ErrorCode, VerbatimError
from ...core.types_v7 import (
    LaneContextV7,
    LaneName,
    LaneOutput,
    LaneSlice,
    LaneStatus,
    LaneV7,
    QueryViewV7,
)

# Errors that must abort the whole search (fail closed): authorization
# denial, integrity failure, epoch fencing, quarantine/lock refusal.
# A lane raising one of these is never converted into a degraded
# LaneOutput — the pipeline must not keep running on a broken
# authorization boundary.
FAIL_CLOSED_CODES = frozenset(
    {
        ErrorCode.NOT_FOUND_OR_UNAUTHORIZED,
        ErrorCode.NOT_FOUND_OR_FORBIDDEN,
        ErrorCode.INTEGRITY,
        ErrorCode.STALE_EPOCH,
        ErrorCode.LOCKED,
        ErrorCode.QUARANTINED,
        ErrorCode.CONSENT_REQUIRED,
        ErrorCode.EGRESS_DENIED,
    }
)

# The registry. Lane modules call ``register_lane(LaneName.LEX,
# lane_lexical)`` at import; tests register fakes the same way.
LANE_REGISTRY: dict[LaneName, Callable] = {}


def register_lane(name: "LaneName | str", fn: Any) -> Any:
    """Register ``fn`` as the implementation of lane ``name``.

    ``fn`` may be a plain callable ``(ctx, query, slice) -> LaneOutput``,
    a :class:`LaneV7` instance, or a :class:`LaneV7` subclass (instantiated
    with no arguments per call). Returns ``fn`` so it can be used as a
    decorator::

        @register_lane(LaneName.LEX)
        def lane_lexical(ctx, query, slice): ...

    Re-registering a name overwrites the previous entry (tests swap fakes
    in and out; production lane modules register exactly once at import).
    """

    LANE_REGISTRY[LaneName(name)] = fn
    return fn


def unregister_lane(name: "LaneName | str") -> None:
    """Remove lane ``name`` from the registry (test hygiene helper)."""

    LANE_REGISTRY.pop(LaneName(name), None)


def registered_lanes() -> tuple[LaneName, ...]:
    """Lane names with a registered implementation, in registry order."""

    return tuple(LANE_REGISTRY.keys())


def _invoke(fn: Any, ctx: LaneContextV7, query: QueryViewV7, slice: LaneSlice) -> LaneOutput:
    """Dispatch one registered lane implementation.

    Accepts ``LaneV7`` subclasses (instantiated), ``LaneV7`` instances
    (``.run``), plain callables, and objects exposing a ``run`` method.
    """

    if isinstance(fn, type) and issubclass(fn, LaneV7):
        return fn().run(ctx, query, slice)
    if isinstance(fn, LaneV7):
        return fn.run(ctx, query, slice)
    if callable(fn):
        return fn(ctx, query, slice)
    run = getattr(fn, "run", None)
    if callable(run):
        return run(ctx, query, slice)
    raise TypeError(f"registered lane {fn!r} is not callable")


def run_one(
    ctx: LaneContextV7,
    query: QueryViewV7,
    name: "LaneName | str",
    slice: LaneSlice,
    *,
    clock: Optional[Callable[[], float]] = None,
) -> LaneOutput:
    """Invoke lane ``name`` under its ``slice`` with honest degradation.

    Guarantees:

    - unregistered lane -> ``unavailable`` / ``reason="not_registered"``
      (pipeline continues — incremental integration, V7-04.03);
    - ``slice.deadline_ms <= 0`` -> ``deadline`` / ``"no_slice_budget"``
      without calling the lane;
    - fail-closed ``VerbatimError`` codes propagate unchanged;
    - any other exception -> ``unavailable`` with ``repr`` reason;
    - candidates beyond ``slice.cap`` are truncated and the drop is
      recorded in ``stats["cap_truncated"]``;
    - a lane that returns ``ok`` but overran its slice is downgraded to
      ``partial`` / ``"deadline_overrun"`` — a slow lane cannot report a
      completed scan it did not finish inside its budget (V7-04.03).

    ``clock`` defaults to ``time.monotonic``; the pipeline injects its
    own clock so per-lane accounting shares one time base.
    """

    clock = clock or time.monotonic
    lname = LaneName(name)
    key = lname.value

    fn = LANE_REGISTRY.get(lname)
    if fn is None:
        return LaneOutput(lane=key, status=LaneStatus.UNAVAILABLE, reason="not_registered")
    if slice is None:
        return LaneOutput(lane=key, status=LaneStatus.UNAVAILABLE, reason="no_slice")
    if slice.deadline_ms <= 0:
        return LaneOutput(lane=key, status=LaneStatus.DEADLINE, reason="no_slice_budget")

    t0 = clock()
    try:
        out = _invoke(fn, ctx, query, slice)
    except VerbatimError as exc:
        if exc.code in FAIL_CLOSED_CODES:
            raise
        return LaneOutput(lane=key, status=LaneStatus.UNAVAILABLE, reason=repr(exc)[:200])
    except Exception as exc:  # noqa: BLE001 — degradation is the contract
        return LaneOutput(lane=key, status=LaneStatus.UNAVAILABLE, reason=repr(exc)[:200])
    elapsed_ms = (clock() - t0) * 1000.0

    if out is None:
        return LaneOutput(lane=key, status=LaneStatus.UNAVAILABLE, reason="returned_none")
    if not isinstance(out, LaneOutput):
        return LaneOutput(
            lane=key,
            status=LaneStatus.UNAVAILABLE,
            reason=f"bad_return:{type(out).__name__}",
        )
    out.lane = key

    # Pool-cap enforcement (V7-05.07): the cap is a policy value; the
    # pipeline enforces it even when a lane overproduces.
    if slice.cap is not None and slice.cap >= 0 and len(out.candidates) > slice.cap:
        dropped = len(out.candidates) - slice.cap
        out.candidates = out.candidates[: slice.cap]
        out.stats["cap_truncated"] = dropped

    out.stats.setdefault("elapsed_ms", round(elapsed_ms, 3))
    out.stats.setdefault("slice_ms", slice.deadline_ms)

    # Deadline honesty: an ``ok`` lane that overran its slice is partial.
    if out.status == LaneStatus.OK and elapsed_ms > slice.deadline_ms:
        out.status = LaneStatus.PARTIAL
        out.reason = "deadline_overrun"
        out.stats["overrun_ms"] = round(elapsed_ms - slice.deadline_ms, 3)

    return out


__all__ = [
    "FAIL_CLOSED_CODES",
    "LANE_REGISTRY",
    "register_lane",
    "unregister_lane",
    "registered_lanes",
    "run_one",
]
