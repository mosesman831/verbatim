"""V7 run-manifest builder (SPEC_V7 V7-22.10, V7-00.07, V7-26.03,
§32.17 hardware/load fields).

Every reported V7 number is produced from a frozen run manifest whose
digest is printed beside the number. A manifest records:

* ``vcs`` — git revision + dirty-tree flag (V7-22.10). A dirty tree is
  recorded, never laundered; a non-git root reports ``unavailable``
  rather than fabricating a revision.
* ``environment`` — Python version/implementation, SQLite library
  version, CPU count, platform, and probed dependency versions
  (``importlib.metadata`` only — packages are never imported, so
  probing is deterministic and side-effect free).
* ``formula`` — the §32 tag set the run used plus its selection
  status. Until owner decision O9 accepts the formula-search report
  every manifest carries ``formula=unselected`` and status
  ``provisional/v7-r0`` (V7-26.03, H104).
* ``datasets``, ``artifacts`` — sha256 pins of every input. A missing
  artifact is recorded ``{"missing": true}`` — never invented.
* ``policy``/``reader``/``judge`` — lane-policy digest, reader id +
  decoding + prompt digest, judge id + prompt digest (V7-22.10).
* ``seeds``, ``hardware``, ``load``, ``started_at``/``ended_at`` —
  the §32.17 envelope-run context fields (the stage-profile record
  itself is ``eval/v7/stage_profile.py``'s job; the manifest pins its
  schema tag so a run can name the profile format it emitted).
* ``digest`` — sha256 of the canonical serialization of the manifest
  with the digest field removed. Canonical form is
  ``verbatim.core.serialize.json_dumps`` (sorted keys, compact, UTF-8,
  no NaN) — identical inputs always produce the identical digest.

Manifests persist as ``eval/v7/manifests/<digest>.json`` (V7-22.10)
via :func:`write_manifest`. This module only builds/verifies; the wave
that executes envelopes writes the files.

CLI: ``python -m eval.v7.manifest --example`` prints a skeleton
manifest (no execution) — handy for shape checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Manifest schema tag (versioned like the §32 formula tags).
MANIFEST_SCHEMA = "run_manifest/v7"

#: §32.17 stage-profile record tag — manifests pin the schema of any
#: stage-profile output they aggregate.
STAGE_PROFILE_SCHEMA = "stage_profile/v7"

#: Where manifests persist (V7-22.10): eval/v7/manifests/<digest>.json.
MANIFEST_DIR = os.path.join("eval", "v7", "manifests")

#: Optional dependencies whose versions are probed (never imported).
#: ``None`` in the map means "not installed" — honest absence.
PROBE_DEPENDENCIES = (
    "numpy",
    "onnxruntime",
    "tokenizers",
    "sqlite-vec",
    "usearch",
    "hnswlib",
    "pytest",
)


def _formula_status_provisional() -> str:
    """The provisional §32 tag (``provisional/v7-r0`` until O9).

    Sourced from the contract module ``verbatim.core.types_v7``;
    lazy-imported per the wave-A rule (falls back to the literal when
    the sibling module is absent).
    """
    try:
        from verbatim.core.types_v7 import FORMULA_STATUS_PROVISIONAL
        return FORMULA_STATUS_PROVISIONAL
    except Exception:
        return "provisional/v7-r0"


def _json_dumps(value: Any) -> str:
    """Canonical JSON via the repo serialization seam (V5-06.01)."""
    try:
        from verbatim.core.serialize import json_dumps
        return json_dumps(value)
    except Exception:
        # seam unavailable: same canonical form inline
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False)


# ---------------------------------------------------------------------------
# environment + vcs collection
# ---------------------------------------------------------------------------


def collect_environment(
    dep_names: Iterable[str] = PROBE_DEPENDENCIES,
) -> dict:
    """Deterministic environment facts — versions probed via
    ``importlib.metadata`` so nothing is imported."""
    deps = {}
    for name in dep_names:
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            deps[name] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "sqlite_version": sqlite3.sqlite_version,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": deps,
    }


def git_state(root: str) -> dict:
    """``{"system": "git", "sha": ..., "dirty": bool}`` for ``root``.

    A non-git root (or absent git) reports ``status: "unavailable"``
    with a reason — the manifest then carries ``sha: null`` /
    ``dirty: null`` rather than a fabricated revision.
    """
    def _git(*args) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=root, capture_output=True,
                text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {
            "system": "git",
            "status": "unavailable",
            "reason": "not a git work tree (or git missing)",
            "sha": None,
            "dirty": None,
        }
    porcelain = _git("status", "--porcelain")
    return {
        "system": "git",
        "status": "ok",
        "sha": sha,
        # `git status` failing is treated as dirty-adjacent: unknown is
        # worse than dirty for provenance, and None stays honest.
        "dirty": None if porcelain is None else bool(porcelain),
    }


# ---------------------------------------------------------------------------
# artifact + dataset pins
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> Optional[str]:
    """sha256 of file bytes, or None when unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def artifact_entry(path: str, root: Optional[str] = None) -> dict:
    """Pin one artifact file: path + sha256 + byte length.

    A missing/unreadable file yields ``{"missing": true}`` — the
    manifest records the expectation honestly instead of a fake digest.
    """
    full = os.path.join(root, path) if root else path
    digest = sha256_file(full)
    entry = {"path": path, "sha256": digest}
    if digest is None:
        entry["missing"] = True
        return entry
    entry["missing"] = False
    entry["bytes"] = os.path.getsize(full)
    return entry


def pin_artifacts(paths: Iterable[str], root: Optional[str] = None) -> list:
    """Sorted, de-duplicated artifact pins (deterministic ordering)."""
    seen = sorted(dict.fromkeys(paths))
    return [artifact_entry(p, root=root) for p in seen]


def pin_datasets(datasets: Optional[dict]) -> dict:
    """``{name: {"digest": ..., "split": ...}}`` per V7-22.10.

    ``datasets`` may already carry pins (the dataset registry computes
    digests); entries lacking ``digest`` are passed through with
    ``digest: null`` — an unpinned dataset is labeled, never invented.
    """
    out = {}
    for name in sorted(datasets or {}):
        body = dict(datasets[name] or {})
        body.setdefault("digest", None)
        body.setdefault("split", None)
        out[name] = body
    return out


# ---------------------------------------------------------------------------
# build + digest
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    root: Optional[str] = None,
    run_id: Optional[str] = None,
    seeds: Iterable[int] = (),
    profile: Optional[str] = None,
    tiers: Iterable[str] = (),
    model_hashes: Optional[dict] = None,
    lane_policy_digest: Optional[str] = None,
    reader: Optional[dict] = None,
    judge: Optional[dict] = None,
    datasets: Optional[dict] = None,
    artifacts: Iterable[str] = (),
    artifact_digests: Optional[dict] = None,
    formula_tags: Optional[dict] = None,
    formula_selected: bool = False,
    formula_status: Optional[str] = None,
    hardware: Optional[dict] = None,
    load: Optional[dict] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    environment: Optional[dict] = None,
    vcs: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build one run manifest (V7-22.10 field set + §32.17 context).

    ``artifacts`` are file paths pinned by sha256; ``artifact_digests``
    supplies pre-computed ``{path: sha256}`` pins for artifacts that do
    not live on this filesystem. ``formula_selected`` is False until
    owner decision O9 — the manifest then carries
    ``formula=unselected`` per V7-26.03.
    """
    if environment is None:
        environment = collect_environment()
    if vcs is None:
        vcs = git_state(root) if root else {
            "system": "git", "status": "unavailable",
            "reason": "no root given", "sha": None, "dirty": None,
        }

    artifact_pins = pin_artifacts(artifacts, root=root)
    for path, digest in sorted((artifact_digests or {}).items()):
        artifact_pins.append({
            "path": path, "sha256": digest, "missing": False,
            "pinned_externally": True,
        })

    if formula_status is None:
        formula_status = (
            "selected" if formula_selected
            else _formula_status_provisional())
    formula = {
        # V7-26.03 / H104: until O9 every published number carries
        # formula=unselected plus the provisional tag set it ran under.
        "selected": bool(formula_selected),
        "label": (
            "formula=selected" if formula_selected
            else "formula=unselected"),
        "status": formula_status,
        "tags": dict(sorted((formula_tags or {}).items())),
    }

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "vcs": vcs,
        "environment": environment,
        "seeds": sorted({int(s) for s in seeds}),
        "profile": profile,
        "tiers": sorted(set(tiers)),
        "model_hashes": dict(sorted((model_hashes or {}).items())),
        "lane_policy_digest": lane_policy_digest,
        "reader": dict(reader) if reader else None,
        "judge": dict(judge) if judge else None,
        "datasets": pin_datasets(datasets),
        "artifacts": artifact_pins,
        "formula": formula,
        "stage_profile_schema": STAGE_PROFILE_SCHEMA,
        "hardware": dict(hardware) if hardware else {
            "cpu_count": environment.get("cpu_count"),
            "machine": environment.get("machine"),
        },
        "load": dict(load) if load else None,
        "started_at": started_at,
        "ended_at": ended_at,
        "extra": dict(extra) if extra else {},
    }
    manifest["digest"] = manifest_digest(manifest)
    return manifest


def manifest_digest(manifest: dict) -> str:
    """Canonical digest: sha256 over ``json_dumps`` of the manifest
    with the ``digest`` field removed (V7-22.10 / V7-00.07)."""
    body = {k: v for k, v in manifest.items() if k != "digest"}
    return hashlib.sha256(_json_dumps(body).encode("utf-8")).hexdigest()


def verify_manifest(manifest: dict) -> bool:
    """Recompute the canonical digest; False on any drift."""
    got = manifest.get("digest")
    return isinstance(got, str) and got == manifest_digest(manifest)


def write_manifest(manifest: dict, root: str) -> str:
    """Persist to ``eval/v7/manifests/<digest>.json``; returns the
    relative path. Byte-stable: canonical pretty JSON."""
    digest = manifest.get("digest") or manifest_digest(manifest)
    rel = os.path.join(MANIFEST_DIR, f"{digest}.json")
    full = os.path.join(root, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2, sort_keys=True,
                           ensure_ascii=False) + "\n")
    return rel


def load_manifest(path: str) -> dict:
    """Strict load via the repo JSON seam (bounded, dup-rejecting)."""
    try:
        from verbatim.core.serialize import json_loads
        with open(path, "r", encoding="utf-8") as f:
            return json_loads(f.read(), max_bytes=4 << 20, max_depth=64)
    except ImportError:
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v7.manifest",
        description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        help="repo root (default: auto)")
    ap.add_argument("--example", action="store_true",
                    help="print a skeleton manifest for this root")
    args = ap.parse_args(argv)
    m = build_manifest(
        root=args.root, run_id="example",
        seeds=[0], profile="local_memory",
        formula_tags={"bm25f": "bm25f/v1", "fusion": "rrf/v1"},
        started_at=None, ended_at=None)
    print(json.dumps(m, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
