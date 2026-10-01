"""Conformance suite (§53.01): requirement-tagged cases executed
through real public entry points against real stores.

Design (§56.01/§56.02):

- Each declared *file spec* names test modules whose bodies assert
  observable engine behavior through public APIs — the pytest node is
  the check function; nothing is imported-and-inspected.
- Requirement tags attach at file granularity, except the v2
  acceptance file whose scenarios map per-scenario (``test_aNN`` → the
  V3 requirement ids whose behavior the scenario exercises).
- The runner collects and executes each file in-process via pytest
  hooks, maps per-node outcomes into conformance records, and keeps
  every declared case in the denominator — ``fail``/``error``/
  ``skipped`` all report, never drop (§53.05).
- Conformance is arm-independent: it measures the v3 implementation
  itself, not a baseline comparison. ``run.py`` invokes it once with
  the ``verbatim_v3`` arm label for gate attribution.

Live-host conformance (real Hermes gateway, real ADK runtime) stays an
explicitly unavailable lane: the adapter cases here exercise the
translate/dispatch/matrix contract in-process (§13), never a live host
(V3-49.03 keeps Hermes a thin out-of-tree plugin).
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .baselines import SuiteRun

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# case declarations
# ---------------------------------------------------------------------------

# Per-scenario requirement tags for the v2 acceptance file. Keys match the
# ``test_aNN`` prefix of the node name; suffix variants (a15b, a16b) inherit
# their base scenario's tags.
_A_REQS: Dict[str, Tuple[str, ...]] = {
    "a01": ("V3-18.09", "V3-15.01"),   # unstructured admitted claim searchable
    "a02": ("V3-15.01",),             # span/offset fidelity through harvest
    "a03": ("V3-18.12",),             # atomic admit + durable across restart
    "a04": ("V3-40.01",),             # durable obligations, stable op keys
    "a05": ("V3-08.01", "V3-09.01"),  # bound caller; deterministic authz
    "a06": ("V3-08.03",),             # audience on evidence; owner-private held
    "a07": ("V3-18.05", "V3-18.07"),  # change vs correction; historical recall
    "a08": ("V3-18.04",),             # incompatible → unresolved group
    "a09": ("V3-18.01",),             # conditional applicability coexistence
    "a10": ("V3-15.03", "V3-18.04"),  # negation preserved; polarity conflict
    "a11": ("V3-18.01",),             # late arrival keeps validity interval
    "a12": ("V3-15.05",),             # bounded reference resolution
    "a13": ("V3-17.03", "V3-18.07"),  # source edit → new revision; old historical
    "a14": ("V3-28.03", "V3-41.03"),  # lane status; honest encoder capability
    "a15": ("V3-15.02",),             # unicode-aware handling, all scripts
    "a16": ("V3-30.01",),             # pack byte ceiling; groups drop whole
    "a17": ("V3-09.04", "V3-09.05"),  # authz epoch; revoked verbs fenced
    "a18": ("V3-46.03",),             # outage degrades; no fake empty success
    "a19": ("V3-05.02",),             # embedded profile runs fully offline
    "a20": ("V3-40.01",),             # lease generations fence stale workers
    "a21": ("V3-21.01",),             # procedure drift flagged/withheld
    "a22": ("V3-31.01",),             # recalled content cannot gain authority
    "a23": ("V3-36.01", "V3-36.03"),  # purge closure; restore cannot resurrect
    "a24": ("V3-13.01",),             # one engine/authz/receipt across depths
    "a25": ("V3-08.01",),             # owner-private survives, never leaks
}


@dataclass(frozen=True)
class _FileSpec:
    path: str                       # repo-relative pytest target
    surface: str                    # conformance matrix row
    requirement_ids: Tuple[str, ...] = ()  # file-level tags
    per_node: Optional[Dict[str, Tuple[str, ...]]] = None  # prefix → tags


_FILE_SPECS: Tuple[_FileSpec, ...] = (
    _FileSpec(
        "tests/test_v2_acceptance.py",
        "core-api",
        per_node=_A_REQS,
    ),
    _FileSpec(
        "tests/test_v2_api_cli.py",
        "cli",
        ("V3-51.01", "V3-47.03", "V3-46.04"),
    ),
    _FileSpec(
        "tests/test_v2_mcp.py",
        "mcp-stdio",
        ("V3-48.01",),
    ),
    _FileSpec(
        "tests/api_v3/test_api_v3.py",
        "api-v3",
        ("V3-47.01", "V3-47.02", "V3-47.03"),
    ),
    _FileSpec(
        "tests/api_v3/test_mcp_v3.py",
        "mcp-v3",
        ("V3-48.01", "V3-48.04", "V3-48.05"),
    ),
    _FileSpec(
        "tests/sdk/test_capture.py",
        "capture-sdk",
        ("V3-13.02", "V3-12.01"),
    ),
    _FileSpec(
        "tests/adapters/test_adapters.py",
        "adapters",
        ("V3-13.01", "V3-13.10", "V3-49.01", "V3-49.03"),
    ),
)

#: Declared-but-unavailable conformance lanes — they appear in the matrix
#: so the gap is visible, never silently absent (§56.01 skipped ≠ absent).
_UNAVAILABLE_SURFACES: Tuple[Tuple[str, str], ...] = (
    (
        "hermes-live",
        "live Hermes gateway host — requires separate approval "
        "(provider activation is never implicit)",
    ),
    (
        "adk-live",
        "live ADK runtime — requires a provisioned ADK host",
    ),
)

_A_PREFIX = re.compile(r"test_(a\d\d)")


# ---------------------------------------------------------------------------
# pytest hook plugins
# ---------------------------------------------------------------------------


class _Collector:
    def __init__(self) -> None:
        self.nodeids: List[str] = []

    def pytest_collection_modifyitems(self, session, config, items):
        self.nodeids.extend(i.nodeid for i in items)


class _Recorder:
    """Per-node outcome recorder; the worst phase outcome wins so a
    setup/teardown failure is never masked by a skipped call phase."""

    _RANK = {"passed": 0, "skipped": 1, "failed": 2}

    def __init__(self) -> None:
        self.outcomes: Dict[str, str] = {}
        self.detail: Dict[str, str] = {}

    def pytest_runtest_logreport(self, report):
        nid = report.nodeid
        prev = self.outcomes.get(nid)
        if prev is None or self._RANK[report.outcome] > self._RANK[prev]:
            self.outcomes[nid] = report.outcome
        if report.failed:
            self.detail[nid] = str(report.longrepr)[:400]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def _a_reqs(nodeid: str) -> Tuple[str, ...]:
    name = nodeid.rsplit("::", 1)[-1]
    m = _A_PREFIX.match(name)
    if m:
        return _A_REQS.get(m.group(1), ())
    return ()


def run_conformance_suite(
    corpus,
    baseline: str,
    *,
    k: int = 5,
    task_ids: Optional[Sequence[str]] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    pytest_args: Sequence[str] = (),
    quiet: bool = False,
) -> SuiteRun:
    """Execute every declared conformance case through pytest's real
    harness. ``baseline`` labels the run (``verbatim_v3`` is the system
    under test); the suite is arm-independent by design."""
    import pytest  # local import: the eval itself stays pytest-optional

    run = SuiteRun(suite="conformance", baseline=baseline, k=k)
    t0 = time.perf_counter()

    # Declared surfaces that produced no executable cases — they get the
    # same explicit unavailable row the live lanes get (dashes + reason)
    # so a missing file can never silently drop a surface (§53.05).
    missing_surfaces: List[Dict[str, str]] = []
    for spec in _FILE_SPECS:
        target = os.path.join(_REPO_ROOT, spec.path)
        if not os.path.exists(target):
            run.notes.append(f"conformance file missing: {spec.path}")
            missing_surfaces.append({
                "surface": spec.surface,
                "reason": f"declared test file missing: {spec.path}",
            })
            continue
        collector = _Collector()
        rc = pytest.main(
            ["--collect-only", "-q", "-p", "no:cacheprovider", target],
            plugins=[collector],
        )
        if not collector.nodeids:
            run.notes.append(
                f"conformance collection empty for {spec.path} (rc={rc})"
            )
            missing_surfaces.append({
                "surface": spec.surface,
                "reason": (
                    "pytest collected no conformance cases from "
                    f"{spec.path} (rc={getattr(rc, 'name', rc)})"
                ),
            })
            continue
        recorder = _Recorder()
        argv = [
            "-q", "-p", "no:cacheprovider", "--no-header",
            *pytest_args, target,
        ]
        if quiet:
            argv.insert(0, "--tb=no")
        rc_run = pytest.main(argv, plugins=[recorder])
        # A wholesale run-phase collapse (interrupt, internal/usage
        # error — anything but OK/TESTS_FAILED) or an early-stop flag
        # truncating a failing run leaves collected nodes with no
        # recorded outcome.  Those are *failed*, never skipped: an
        # unexecuted check is not a skip (§53.05).  A node pytest
        # genuinely skipped already carries a "skipped" outcome.
        early_stop = any(
            a in ("-x", "--exitfirst") or a.startswith("--maxfail")
            for a in pytest_args
        )
        collapsed = rc_run not in (
            pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED,
        ) or (early_stop and rc_run == pytest.ExitCode.TESTS_FAILED)

        for nid in collector.nodeids:
            reqs = spec.per_node and _a_reqs(nid) or spec.requirement_ids
            outcome = recorder.outcomes.get(nid)
            detail = recorder.detail.get(nid, "")
            if outcome is None:
                if collapsed:
                    outcome = "failed"
                    detail = detail or (
                        "pytest run phase exited "
                        f"{getattr(rc_run, 'name', rc_run)} before this "
                        "node executed"
                    )
                else:
                    outcome = "skipped"
            run.records.append(
                {
                    "case_id": nid.rsplit("::", 1)[-1],
                    "node": nid,
                    "surface": spec.surface,
                    "requirement_ids": reqs,
                    "result": outcome,
                    "detail": detail,
                }
            )
            if outcome == "failed":
                run.errors += 1

    surfaces: Dict[str, Dict[str, int]] = {}
    for rec in run.records:
        s = surfaces.setdefault(
            rec["surface"], {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        )
        s["total"] += 1
        s[rec["result"] if rec["result"] in s else "skipped"] += 1

    run.metrics.update(
        {
            "cases_total": len(run.records),
            "cases_passed": sum(1 for r in run.records if r["result"] == "passed"),
            "cases_failed": sum(1 for r in run.records if r["result"] == "failed"),
            "cases_skipped": sum(
                1 for r in run.records if r["result"] == "skipped"
            ),
            "surfaces": surfaces,
            "unavailable_surfaces": [
                {"surface": s, "reason": why} for s, why in _UNAVAILABLE_SURFACES
            ] + missing_surfaces,
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
    )
    run.unavailable_lanes.extend(s for s, _ in _UNAVAILABLE_SURFACES)
    run.unavailable_lanes.extend(m["surface"] for m in missing_surfaces)
    run.notes.append(
        "conformance measures the v3 implementation once; it is "
        "arm-independent — baseline label tags gate attribution only"
    )
    return run


__all__ = ["run_conformance_suite"]
