"""The V6 eval portfolio — one entry point that runs every landed
suite and assembles the combined results record (SPEC_V6 §06).

Suites landed so far:

* ``comparators`` — the executed comparator registry
  (:func:`eval.v6.comparators.run_registry`) over the shared V5
  consumer corpus: always-run offline arms plus honestly-probed
  competitor rows, and the ``verbatim_memory``-vs-each comparison set.
* ``v5_tail`` — the V5-tail disposition check: reads
  ``eval/v5/dispositions_v5.json`` and counts requirements still
  unresolved (not ``locally_measured``/``qualified``/
  ``not_applicable``/``deferred``), so the V6 report keeps the
  inherited tail visible instead of implying it closed.

Each suite runs in a guarded call: a suite that raises records
``status=failed`` with the exception — the portfolio never crashes out
of an honest half-empty report. New suites register in
:func:`_suite_runners` — a name → callable map — and join the same
``--suite`` selection, guard, and verdict roll-up.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from collections import Counter
from typing import Any, Callable, Dict, Optional

from eval.v5.harness import environment


def _guard(name: str, fn: Callable[..., Any], **kw: Any) -> dict:
    try:
        out = fn(**kw)
        if hasattr(out, "to_dict"):
            out = out.to_dict()
        elif not isinstance(out, dict):
            out = {"value": out}
        out.setdefault("suite", name)
        out.setdefault("status", "executed")
        return out
    except Exception as exc:  # noqa: BLE001 — honest failure record
        return {
            "suite": name,
            "status": "failed",
            "verdict": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }


# ---------------------------------------------------------------------------
# suite: v5_tail — inherited-requirements tail check
# ---------------------------------------------------------------------------

#: Disposition statuses that count as closed for the tail check.
#: ``deferred`` is included because SPEC_V6-04.02/04.03 name it an
#: explicit terminal V6 disposition for V5-inherited rows
#: (``deferred`` with owner+reason; the deferral list is published in
#: the V6 report — deferred rows stay visible in the ledger's carried
#: tail, so closure here never hides still-owed work).
RESOLVED_STATUSES = frozenset({
    "locally_measured", "qualified", "not_applicable", "deferred",
})


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def run_v5_tail(*, dispositions_path: Optional[str] = None,
                **_: Any) -> dict:
    """Count unresolved V5 requirements in the curated dispositions.

    "Unresolved" = the row's ``qualification_status`` is missing or not
    one of ``locally_measured``/``qualified``/``not_applicable``/
    ``deferred`` — i.e. ``planned``/``in_progress``/
    ``implemented_unmeasured``/``failed``/``not_run`` tails inherited
    into V6. ``deferred`` resolves only per V6-04.02/04.03: the row
    must carry owner+reason in its note, and deferred rows remain
    listed in the v6 ledger's carried tail (the published deferral
    list). The ids are listed (bounded) so the tail stays auditable,
    never summarized away.
    """
    path = dispositions_path or os.path.join(
        _repo_root(), "eval", "v5", "dispositions_v5.json")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    reqs = data.get("requirements") or {}
    counts: Counter = Counter()
    unresolved = []
    for rid, row in reqs.items():
        row = row or {}
        status = row.get("qualification_status") or "unset"
        counts[status] += 1
        if status not in RESOLVED_STATUSES:
            unresolved.append(rid)
            continue
        if status == "deferred" and not (
            str(row.get("owner") or "").strip()
            and len(str(row.get("note") or "")) > 20
        ):
            # A bare ``deferred`` label is not the V6-04.03 deferral
            # record — owner+reason required.
            unresolved.append(rid)
    return {
        "suite": "v5_tail",
        "status": "executed",
        "dispositions_path": os.path.relpath(path, _repo_root()),
        "requirements_total": len(reqs),
        "status_counts": dict(sorted(counts.items())),
        "unresolved_count": len(unresolved),
        "unresolved_ids": sorted(unresolved)[:200],
        "resolved_statuses": sorted(RESOLVED_STATUSES),
        "verdict": "executed",
        "notes": [
            "unresolved = qualification_status not in "
            "locally_measured/qualified/not_applicable/deferred "
            "(deferred = V6-04.02/04.03 disposition with owner+reason)",
        ],
    }


# ---------------------------------------------------------------------------
# suite registry — extend here as more V6 suites land
# ---------------------------------------------------------------------------


def _run_comparators(*, scale: dict, seed: int,
                     workdir: Optional[str]) -> dict:
    from . import comparators
    from eval.v5.corpus import seed_corpus

    corpus = seed_corpus(
        memories=scale["comparators_memories"], seed=seed)
    return comparators.run_registry(
        corpus, workdir=workdir, k=scale["comparators_k"])


def _run_envelopes(*, scale: dict, seed: int,
                   workdir: Optional[str]) -> dict:
    """The §02.2 envelope set — A0/A0-cache/A0-neural/A1/A3.

    ``scale["envelopes_params"]`` may raise the quick floors (the
    wave-1 measurement pins: A0/A0-cache/A1 at memories>=512, A1 at
    reader>=40/writer>=16 — published per envelope in its own ``scale``
    block, never hidden behind the "quick" label).
    """
    from . import envelopes
    return envelopes.run_envelopes(
        scale=scale["envelopes_scale"], seed=seed,
        params=scale.get("envelopes_params"), workdir=workdir)


def _run_add_ack_scaling(*, scale: dict, seed: int,
                         workdir: Optional[str]) -> dict:
    """The standalone V6-02.16 stage-scaling probe (V6-02.16/F21)."""
    from . import envelopes
    return envelopes.measure_add_ack_scaling(
        scale["ack_checkpoints"], seed=seed,
        profile_adds=scale["ack_profile_adds"], workdir=workdir)


def _run_twins(*, scale: dict, seed: int,
               workdir: Optional[str]) -> dict:
    """The auto_safe twin suite (V6-03.03). The pair corpus is frozen
    in ``eval.v6.twins.TWIN_PAIRS`` — no seed knob; ``verdict`` is
    normalized to the portfolio vocabulary (pass→passed, fail→failed)
    while the native flag stays in ``suite_verdict``."""
    from . import twins
    wd = os.path.join(workdir, "twins") if workdir else None
    rep = twins.run_twins(wd)
    rep["suite_verdict"] = rep["verdict"]
    rep["verdict"] = "passed" if rep["verdict"] == "pass" else "failed"
    return rep


def _run_neural(*, scale: dict, seed: int,
                workdir: Optional[str]) -> dict:
    """The paired hashing-vs-artifact quality gate (V6-03.10).

    ``neural_recommended`` False is a *measured outcome*, not a suite
    failure — the suite verdict stays ``executed`` and the gate detail
    lives in ``gate_verdict``/``neural_recommended``.
    """
    from . import neural
    wd = workdir or tempfile.mkdtemp(prefix="v6-neural-")
    rep = neural.paired_quality(
        os.path.join(wd, "neural"),
        memories=scale["neural_memories"], seed=seed,
        k=scale["neural_k"])
    gate = rep.pop("verdict") or {}
    rep["gate_verdict"] = gate
    rep["neural_recommended"] = gate.get("neural_recommended")
    rep["gate_reason"] = gate.get("reason")
    rep["verdict"] = "executed"
    return rep


def _suite_runners() -> Dict[str, Callable[..., dict]]:
    """name -> callable(scale=…, seed=…, workdir=…) -> suite dict."""
    return {
        "comparators": _run_comparators,
        "envelopes": _run_envelopes,
        "add_ack_scaling": _run_add_ack_scaling,
        "twins": _run_twins,
        "neural": _run_neural,
        "v5_tail": lambda **_: run_v5_tail(),
    }


def run_portfolio(*, quick: bool = True, seed: int = 42,
                  suites: Optional[list] = None,
                  workdir: Optional[str] = None) -> Dict[str, Any]:
    """Run the V6 suite set and return the combined record."""
    scale = {
        "comparators_memories": 48,
        "comparators_k": 8,
        # quick-but-real floors: the wave-1 A0/A0-cache/A1 measurements
        # pin memories>=512 / reader>=40 / writer>=16 — every envelope
        # publishes its own realized n inside its ``scale`` block.
        "envelopes_scale": "quick",
        "envelopes_params": {
            "a0_memories": 512,
            "cache_memories": 512,
            "neural_memories": 512,
            "a1_memories": 512,
        },
        "ack_checkpoints": (512, 1_024),
        "ack_profile_adds": 16,
        "neural_memories": 48,
        "neural_k": 8,
    } if quick else {
        "comparators_memories": 128,
        "comparators_k": 8,
        "envelopes_scale": "full",
        "envelopes_params": None,
        "ack_checkpoints": (1_024, 4_096),
        "ack_profile_adds": 24,
        "neural_memories": 128,
        "neural_k": 8,
    }

    runners = _suite_runners()
    want = set(suites or list(runners))
    unknown = sorted(want - set(runners))
    out: Dict[str, Any] = {
        "portfolio": "v6",
        "quick": quick,
        "seed": seed,
        "qualification": "locally_measured",
        "environment": environment(),
        "suites": {},
        "unmeasured": [
            "operator_time cost category (no human-time meter in run)",
            "hosted/priced comparator costs (no paid benchmarks run)",
            "deployment cost beyond local embedded store",
            "competitor editions absent from this environment "
            "(mem0ai, graphiti_core+neo4j, holographic, hosted rows)",
        ],
    }
    if unknown:
        out["notes"] = [
            f"unknown suite(s) requested and skipped: {unknown} — "
            f"available: {sorted(runners)}"
        ]
    s = out["suites"]
    for name, runner in runners.items():
        if name not in want:
            continue
        s[name] = _guard(
            name, runner, scale=scale, seed=seed, workdir=workdir)

    # Honest roll-up: any suite failure or explicit loss marks the run.
    suite_states = [
        (s[n].get("verdict") or s[n].get("status")) for n in s
    ]
    out["verdict"] = (
        "failed" if any(v == "failed" for v in suite_states)
        else "passed" if all(v in ("passed", "executed")
                             for v in suite_states)
        else "inconclusive"
    )
    return out


__all__ = ["RESOLVED_STATUSES", "run_portfolio", "run_v5_tail"]
