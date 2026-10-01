"""Mutation suite — SPEC_V4 §62 / V4-52.* + gate G4-11 attack resistance.

Each declared mutation in ``eval/v4/mutations.yaml`` removes exactly one
enforcement check. This suite applies every mutation sequentially through
``tools/run_mutation.py`` (exact-string replace, subprocess pytest of the
named killer tests, ≤60s per-test timeout, unconditional restore with
SHA-256 verification) and asserts the mutation is **killed**: at least one
named test fails or errors, proving the check is real enforcement and not
dead code.

A ``survived`` outcome is a genuine enforcement gap — the fix is a new
regression test through the real ``Store.create`` path (see
``test_purge_suppression_commit.py`` for the MUT-16 example), never an edit
of the declared set to hide it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MUTATIONS_YAML = REPO_ROOT / "eval" / "v4" / "mutations.yaml"
RUNNER = REPO_ROOT / "tools" / "run_mutation.py"
TIMEOUT_S = 60.0  # per-test budget, per the G4-11 contract


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_mutation", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    # dataclass() resolves cls.__module__ via sys.modules — register first.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rm = _load_runner()
MUTATIONS = rm.load_mutations(MUTATIONS_YAML)


def _git_porcelain() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


# Snapshot before any mutation runs — the worktree may legitimately be
# dirty from unrelated work; the contract is that mutation runs leave it
# *unchanged*, i.e. every mutated file restored byte-identical.
_GIT_STATUS_AT_LOAD = _git_porcelain()


def test_declared_mutation_set_is_wellformed() -> None:
    """Static validation, cheap and first: unique ids, real files, and
    anchors that occur exactly once — a non-unique anchor is a declaration
    bug, not mutation evidence."""
    ids = [m.id for m in MUTATIONS]
    assert len(ids) == len(set(ids)), "duplicate mutation ids"
    assert len(MUTATIONS) >= 18, "declared set must cover every seam"
    for mut in MUTATIONS:
        assert mut.id.startswith("MUT-"), mut.id
        assert mut.killed_by, f"{mut.id} names no killer tests"
        assert mut.requirements, f"{mut.id} cites no requirements"
        path = REPO_ROOT / mut.file
        src = path.read_bytes().decode("utf-8")
        n = src.count(mut.anchor)
        assert n == 1, f"{mut.id}: anchor occurs {n}x in {mut.file}"
        assert mut.anchor != mut.replacement


@pytest.mark.parametrize(
    "mut", MUTATIONS, ids=[m.id for m in MUTATIONS]
)
def test_mutation_is_killed(mut) -> None:
    """Apply the mutant, run its named tests, restore, verify hash.

    Sequential by construction: pytest executes these in declaration
    order, one mutation in the tree at a time.
    """
    result = rm.run_mutation(mut, repo_root=REPO_ROOT, timeout_s=TIMEOUT_S)
    assert result.restored_sha256_ok, (
        f"{mut.id}: {mut.file} not byte-identical after restore"
    )
    assert result.outcome == rm.KILLED, (
        f"{mut.id} {result.outcome}: {mut.check}\n{result.detail}"
    )


def test_worktree_unchanged_after_mutation_runs() -> None:
    """Last check: mutation runs must leave the tree exactly as found.

    Byte-identical restore is already asserted per mutation via SHA-256;
    this catches anything outside the mutated files (stray writes,
    leftover state)."""
    assert _git_porcelain() == _GIT_STATUS_AT_LOAD, (
        "mutation runs changed the worktree:\n" + _git_porcelain()
    )
