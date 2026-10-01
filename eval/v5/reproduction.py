"""Reproduction manifest for V5 runs (E71 / SPEC_V5 §24.11).

Everything an independent operator needs to re-run exactly this
measurement: pinned seeds, corpus digests, scale parameters, package
versions, environment fingerprint, the exact CLI invocation, and a
checksum over the produced artifacts. The manifest is *generated from
the executed results* — it can only describe what actually ran.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Dict, Optional


def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))),
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))),
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def build_manifest(results: Dict[str, Any], *,
                   command: str,
                   artifact_paths: Sequence = ()) -> Dict[str, Any]:
    """The reproduction manifest for one executed run."""
    from .harness import environment
    suites = results.get("suites") or {}
    digests = {}
    scales = {}
    seeds = set()
    for name, suite in suites.items():
        c = suite.get("corpus") or {}
        if c.get("digest"):
            digests[name] = c["digest"]
        if c.get("seed") is not None:
            seeds.add(c["seed"])
        if suite.get("scale"):
            scales[name] = suite["scale"]
    artifacts = {}
    for p in artifact_paths or ():
        try:
            with open(p, "rb") as fh:
                artifacts[os.path.basename(p)] = hashlib.sha256(
                    fh.read()).hexdigest()
        except OSError:
            artifacts[os.path.basename(p)] = "unreadable"
    return {
        "manifest_version": "v5-repro-1",
        "command": command,
        "seeds": sorted(seeds),
        "corpus_digests": digests,
        "scales": scales,
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "environment": environment(),
        "dependency_probe": _dependency_probe(),
        "artifact_sha256": artifacts,
        "qualification": results.get("qualification"),
        "notes": [
            "stdlib + installed repo deps only; no network",
            "managed/external worker modes recorded per suite",
            "scaled-down A3/A0 runs are locally measured, never "
            "presented as release qualification",
        ],
    }


def _dependency_probe() -> Dict[str, Any]:
    import importlib
    out = {}
    for mod in ("verbatim", "mem0", "mem0ai"):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "present")
        except Exception as exc:  # noqa: BLE001
            out[mod] = f"absent: {type(exc).__name__}"
    return out


def write_manifest(results: Dict[str, Any], path: str, *,
                   command: str,
                   artifact_paths: Sequence = ()) -> Dict[str, Any]:
    m = build_manifest(results, command=command,
                       artifact_paths=artifact_paths)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2, sort_keys=True, default=str)
    return m


__all__ = ["build_manifest", "write_manifest"]
