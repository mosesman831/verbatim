"""Developer-experience workflow probe (E65 / SPEC_V5 §21 DX slice).

Runs the documented consumer workflow end-to-end through the public
``verbatim.Memory`` surface — open → add → wait → search → inspect →
forget → status → close — and scores each step with a denominator, a
typed outcome, and wall-clock latency. Error-path checks (invalid
calls must raise *typed* errors, contract violations must not corrupt
state) are part of DX: an API that fails loudly and cleanly is part of
the product surface.

Every step is attempted for real; a step that cannot run is recorded
``unavailable``/``error`` — never skipped out of the denominator.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Step:
    """One scripted DX step."""

    name: str
    status: str = "unmeasured"   # ok | error | unavailable
    ms: float = 0.0
    detail: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status,
            "ms": round(self.ms, 3), "detail": self.detail,
            "error": self.error,
        }


def _step(name: str, fn, *args, **kw) -> Tuple[Step, Any]:
    s = Step(name=name)
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kw)
        s.status = "ok"
        return s, out
    except Exception as exc:  # noqa: BLE001 — typed error recorded
        s.status = "error"
        s.error = f"{type(exc).__name__}: {exc}"
        return s, None
    finally:
        s.ms = (time.perf_counter() - t0) * 1000.0


def _expect_typed_error(name: str, fn, *args, **kw) -> Step:
    """Contract checks: a call that should fail must raise a *typed*
    VerbatimError — silent success or a raw crash both fail."""
    s = Step(name=name)
    t0 = time.perf_counter()
    try:
        fn(*args, **kw)
        s.status = "error"
        s.error = "expected a typed error; call succeeded"
    except Exception as exc:  # noqa: BLE001
        from verbatim.core.types import VerbatimError
        if isinstance(exc, VerbatimError):
            s.status = "ok"
            s.detail = f"typed:{exc.code.value if hasattr(exc.code,'value') else exc.code}"
        else:
            s.status = "error"
            s.error = f"untyped error {type(exc).__name__}: {exc}"
    finally:
        s.ms = (time.perf_counter() - t0) * 1000.0
    return s


def run_dx_probe(*, workdir: Optional[str] = None,
                 quick_adds: int = 3) -> dict:
    """The scripted consumer workflow — one pass, real store."""
    from verbatim import Memory

    tmp = workdir or tempfile.mkdtemp(prefix="verbatim-v5-dx-")
    steps: List[Step] = []
    mem = None

    s, mem = _step("open", Memory, os.path.join(tmp, "mem.db"),
                   user_id="dx-user")
    steps.append(s)
    if mem is None:
        return {"suite": "dx", "verdict": "failed",
                "steps": [st.to_dict() for st in steps],
                "denominator": {"attempted": len(steps), "ok": 0}}

    added: List[Any] = []
    for i in range(quick_adds):
        s, res = _step(f"add[{i}]", mem.add,
                       f"dx probe note {i}: locker code DX-{400 + i}")
        steps.append(s)
        if res is not None:
            added.append(res)

    if added:
        s, rd = _step("wait_ready", mem.wait_ready, added[-1],
                      timeout_ms=5000)
        s.detail = getattr(rd, "state", "") if rd is not None else ""
        steps.append(s)

        s, sr = _step("search", mem.search, f"DX-{400}", limit=4)
        if sr is not None:
            s.detail = f"status={sr.status} items={len(sr.items)}"
        steps.append(s)

        s, ins = _step("inspect", mem.inspect, added[-1].ref)
        if ins is not None:
            s.detail = f"found={ins.found}"
        steps.append(s)

    # error-path checks — the typed-error contract is part of DX
    steps.append(_expect_typed_error(
        "err:search_non_string", mem.search, 1234))
    steps.append(_expect_typed_error(
        "err:search_empty", mem.search, "   "))
    steps.append(_expect_typed_error(
        "err:search_bad_limit", mem.search, "q", limit=0))
    steps.append(_expect_typed_error(
        "err:add_bad_change", mem.add, "x", change="bogus"))
    steps.append(_expect_typed_error(
        "err:inspect_garbage_ref", mem.inspect, "not-a-ref"))

    if added:
        s, fr = _step("forget", mem.forget, added[-1].ref)
        if fr is not None:
            s.detail = str(getattr(fr, "status", ""))
        steps.append(s)

    s, status = _step("status", mem.status)
    if status is not None:
        s.detail = "keys=" + ",".join(sorted(
            status.to_dict().keys()))[:120] if hasattr(
                status, "to_dict") else "ok"
    steps.append(s)

    s, rep = _step("close", mem.close)
    if rep is not None:
        s.detail = str(getattr(rep, "status",
                               getattr(rep, "state", "")))
    steps.append(s)

    # post-close guard: calls after close must raise typed errors
    steps.append(_expect_typed_error("err:use_after_close",
                                     mem.search, "anything"))

    attempted = len(steps)
    ok = sum(1 for s in steps if s.status == "ok")
    errs = [s for s in steps if s.status == "error"]
    return {
        "suite": "dx",
        "qualification": "locally_measured",
        "verdict": "passed" if not errs else "failed",
        "denominator": {"attempted": attempted, "ok": ok,
                        "errors": len(errs)},
        "total_ms": round(sum(s.ms for s in steps), 2),
        "steps": [s.to_dict() for s in steps],
    }


__all__ = ["Step", "run_dx_probe"]
