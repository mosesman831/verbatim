#!/usr/bin/env python3
"""Mutation harness for enforcement checks — SPEC_V4 §62 / V4-52.*, gate G4-11.

Reads a declared mutation set (``eval/v4/mutations.yaml``) and, for each
mutation, applies it to the working tree as an *exact string replacement*,
runs the named pytest node ids in a subprocess, restores the file, and
verifies the restore is byte-identical via SHA-256.

Outcomes per mutation:

- ``killed``   — at least one named test failed or errored under the mutant.
                 The enforcement check is real; removing it breaks a test.
- ``survived`` — every named test passed under the mutant. This is a genuine
                 enforcement gap and must be closed by adding a regression
                 test (through the real Store.create path), not by editing
                 the declared mutation set to hide it.
- ``timeout``  — the test run exceeded the per-test timeout (default 60s).
                 Inconclusive; never counts as killed.
- ``error``    — anchor missing/non-unique, harness failure, pytest internal
                 error, or restore-hash mismatch. Never counts as killed.

Sequential, deterministic, and leaves the working tree clean: the mutated
file is restored in a ``finally`` even on timeout, crash, or test failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MUTATIONS = REPO_ROOT / "eval" / "v4" / "mutations.yaml"
DEFAULT_TIMEOUT_S = 60.0

KILLED = "killed"
SURVIVED = "survived"
TIMEOUT = "timeout"
ERROR = "error"

# pytest exit codes: 0 = all passed, 1 = tests failed, 2 = interrupted,
# 3 = internal error, 4 = usage error, 5 = no tests collected.
# Only exit code 1 is proof a named test exercised and caught the mutant.
_PYTEST_FAILED = 1


@dataclass
class Mutation:
    id: str
    file: str
    anchor: str
    replacement: str
    killed_by: List[str]
    requirements: List[str] = field(default_factory=list)
    check: str = ""


@dataclass
class MutationResult:
    id: str
    file: str
    outcome: str
    returncode: Optional[int]
    timed_out: bool
    duration_s: float
    restored_sha256_ok: bool
    killed_by: List[str]
    requirements: List[str]
    check: str
    detail: str = ""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _drop_stale_pyc(source_path: Path) -> None:
    """Remove the cached bytecode for ``source_path`` if present.

    Mutated sources can share mtime+size with the original (mutations are
    same-length single-token swaps); a stale ``__pycache__`` entry would let
    pytest import the *unmutated* bytecode. PYTHONDONTWRITEBYTECODE stops new
    caches being written; this deletes any pre-existing one.
    """
    import importlib.util

    try:
        pyc = Path(importlib.util.cache_from_source(str(source_path)))
    except Exception:
        return
    try:
        pyc.unlink()
    except FileNotFoundError:
        pass


def load_mutations(path: Path) -> List[Mutation]:
    """Load the declared mutation set (YAML or JSON)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        doc = yaml.safe_load(text)
    else:
        doc = json.loads(text)
    muts: List[Mutation] = []
    for entry in doc["mutations"]:
        muts.append(
            Mutation(
                id=entry["id"],
                file=entry["file"],
                anchor=entry["anchor"],
                replacement=entry["replacement"],
                killed_by=list(entry["killed_by"]),
                requirements=list(entry.get("requirements", [])),
                check=entry.get("check", ""),
            )
        )
    return muts


def apply_mutation(mut: Mutation, repo_root: Path, original: bytes) -> None:
    """Apply ``mut`` exactly once over ``original`` bytes.

    Raises ``ValueError`` if the anchor is missing or non-unique — a fuzzy
    or ambiguous mutation is a declaration bug, not evidence.
    """
    path = repo_root / mut.file
    text = original.decode("utf-8")
    count = text.count(mut.anchor)
    if count != 1:
        raise ValueError(
            f"{mut.id}: anchor occurs {count}x in {mut.file} (need exactly 1)"
        )
    mutated = text.replace(mut.anchor, mut.replacement, 1)
    if mutated == text:
        raise ValueError(f"{mut.id}: replacement produces no change")
    path.write_bytes(mutated.encode("utf-8"))


def run_named_tests(
    repo_root: Path, test_ids: Sequence[str], timeout_s: float
) -> tuple[Optional[int], bool, str]:
    """Run pytest node ids in a subprocess with a hard timeout.

    Returns (returncode, timed_out, tail-of-output). On timeout the process
    group is killed so no orphaned test keeps the file locked.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *test_ids,
        "-x",
        "-q",
        "-p",
        "no:cacheprovider",
        "--tb=short",
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,  # own process group → group kill on timeout
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        return proc.returncode, False, out or ""
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, _ = proc.communicate()
        return proc.returncode, True, out or ""


def run_mutation(
    mut: Mutation, repo_root: Path = REPO_ROOT, timeout_s: float = DEFAULT_TIMEOUT_S
) -> MutationResult:
    """Apply one mutation, run its named tests, restore, verify hash."""
    path = repo_root / mut.file
    original: Optional[bytes] = None
    try:
        original = path.read_bytes()
    except OSError as exc:
        return MutationResult(
            id=mut.id,
            file=mut.file,
            outcome=ERROR,
            returncode=None,
            timed_out=False,
            duration_s=0.0,
            restored_sha256_ok=False,
            killed_by=list(mut.killed_by),
            requirements=list(mut.requirements),
            check=mut.check,
            detail=f"cannot read target: {exc}",
        )
    original_sha = _sha256(original)
    outcome = ERROR
    detail = ""
    returncode: Optional[int] = None
    timed_out = False
    started = time.monotonic()
    try:
        try:
            apply_mutation(mut, repo_root, original)
        except (ValueError, OSError) as exc:
            detail = f"apply failed: {exc}"
        else:
            _drop_stale_pyc(path)
            returncode, timed_out, out = run_named_tests(
                repo_root, mut.killed_by, timeout_s
            )
            tail = "\n".join(out.splitlines()[-15:])
            if timed_out:
                outcome = TIMEOUT
                detail = f"timed out after {timeout_s:.0f}s\n{tail}"
            elif returncode == _PYTEST_FAILED:
                outcome = KILLED
                detail = tail
            elif returncode == 0:
                outcome = SURVIVED
                detail = f"all named tests passed under the mutant\n{tail}"
            else:
                outcome = ERROR
                detail = f"pytest exit code {returncode} (not a test failure)\n{tail}"
    finally:
        # Restore unconditionally — timeout, crash, or test failure alike —
        # then prove the restore is byte-identical via SHA-256.
        if original is not None:
            path.write_bytes(original)
    restored_ok = _sha256(path.read_bytes()) == original_sha
    if not restored_ok:
        outcome = ERROR
        detail = f"RESTORE INTEGRITY FAILURE for {mut.file}\n{detail}"
    return MutationResult(
        id=mut.id,
        file=mut.file,
        outcome=outcome,
        returncode=returncode,
        timed_out=timed_out,
        duration_s=round(time.monotonic() - started, 2),
        restored_sha256_ok=restored_ok,
        killed_by=list(mut.killed_by),
        requirements=list(mut.requirements),
        check=mut.check,
        detail=detail.strip(),
    )


def run_suite(
    muts: Sequence[Mutation],
    repo_root: Path = REPO_ROOT,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    progress=None,
) -> List[MutationResult]:
    """Run every mutation sequentially; never overlap mutations."""
    results: List[MutationResult] = []
    for mut in muts:
        result = run_mutation(mut, repo_root=repo_root, timeout_s=timeout_s)
        results.append(result)
        if progress:
            progress(result)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mutations",
        type=Path,
        default=DEFAULT_MUTATIONS,
        help="mutation declaration file (yaml or json)",
    )
    ap.add_argument(
        "--only", nargs="*", default=None, help="run only these mutation ids"
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="per-mutation pytest timeout in seconds (default 60)",
    )
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--json", type=Path, default=None, help="write JSON report")
    args = ap.parse_args(argv)

    muts = load_mutations(args.mutations)
    if args.only:
        wanted = set(args.only)
        muts = [m for m in muts if m.id in wanted]
        missing = wanted - {m.id for m in muts}
        if missing:
            print(f"unknown mutation ids: {sorted(missing)}", file=sys.stderr)
            return 4

    started = time.monotonic()

    def _progress(r: MutationResult) -> None:
        print(
            f"{r.id}  {r.outcome:<8} {r.duration_s:>6.1f}s  {r.file}  ::  {r.check}",
            flush=True,
        )

    results = run_suite(
        muts, repo_root=args.repo_root, timeout_s=args.timeout, progress=_progress
    )

    killed = sum(r.outcome == KILLED for r in results)
    survived = [r for r in results if r.outcome == SURVIVED]
    other = [r for r in results if r.outcome in (TIMEOUT, ERROR)]
    elapsed = time.monotonic() - started

    print("\n=== mutation report ===")
    print(f"mutations run : {len(results)}")
    print(f"killed        : {killed}")
    print(f"survived      : {len(survived)}")
    print(f"timeout/error : {len(other)}")
    print(f"runtime       : {elapsed:.1f}s")
    for r in survived:
        print(f"  SURVIVED {r.id}: {r.check} — enforcement gap, add a regression test")
    for r in other:
        print(f"  {r.outcome.upper()} {r.id}: {r.check} :: {r.detail.splitlines()[-1] if r.detail else ''}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "mutations": len(results),
                    "killed": killed,
                    "survived": len(survived),
                    "runtime_s": round(elapsed, 2),
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
            + "\n"
        )

    return 0 if results and all(r.outcome == KILLED for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
