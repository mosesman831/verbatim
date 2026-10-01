"""Mutation suite — SPEC_V7 §35 / V7-35.01 (V4-52.* discipline carried).

Each declared mutation in ``eval/v7/mutations.yaml`` removes exactly one
landed enforcement check. This suite applies every promoted mutation
sequentially through ``tools/run_mutation.py`` (exact-string replace,
subprocess pytest of the named killer tests, ≤60s per-test timeout,
unconditional restore with SHA-256 verification) and asserts the mutation
is **killed**: at least one named test fails or errors, proving the check
is real enforcement and not dead code.

A ``survived`` outcome is a genuine enforcement gap — the fix is a new
regression test through the real path, never an edit of the declared set
to hide it.

The §35 column also declares targets under ``pending:`` — modules whose
enforcement has not landed or cannot yet be killed by an existing test.
The loader ignores ``pending:`` entirely; an empty ``mutations:`` list is
a valid zero-mutant set, so this suite skips the per-mutant run when
nothing is promoted rather than failing on absence.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MUTATIONS_YAML = REPO_ROOT / "eval" / "v7" / "mutations.yaml"
RUNNER = REPO_ROOT / "tools" / "run_mutation.py"
TIMEOUT_S = 60.0  # per-test budget, per the V4-52 discipline carried

#: The 23 §35 mutation targets as <area>/<mutation> slugs — every row of
#: the §35 "Mutation targets" column, in table order. Promoted and
#: pending entries together must cover this list exactly.
EXPECTED_TARGETS = [
    "analyzer/skip-clitic-split",
    "analyzer/fold-identifiers",
    "lexical/universe-stats",
    "lexical/skip-held-subtraction",
    "fuzzy/respell-identifiers",
    "dense/ann-without-eligibility-recheck",
    "entities/auto-apply-candidate-aliases",
    "graph/traverse-held-units",
    "temporal/drop-precision",
    "temporal/wrong-anchor",
    "events/unpinned-tuple-delivered",
    "fusion/constant-signal-weight",
    "fusion/unbounded-boost",
    "verdict/floor-based-deletion",
    "packs/computed-item-undelivered-input",
    "consolidation/stale-observation-alone",
    "standing/stale-pack-after-hold",
    "t2/accept-unverified-quote",
    "reflect/cite-undelivered-ref",
    "trust/skip-rescan",
    "trust/drop-blocked-silently",
    "performance/extra-snapshot-per-search",
    "eval/tripwire-bypass",
]


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


def test_declared_target_coverage_is_complete() -> None:
    """The §35 column is fully declared: promoted + pending == 23 targets,
    in table order, with no target silently dropped between lists."""
    doc = yaml.safe_load(MUTATIONS_YAML.read_text(encoding="utf-8"))
    promoted = doc["mutations"] or []
    pending = doc["pending"] or []
    all_entries = promoted + pending
    ids = [p["id"] for p in all_entries]
    assert len(set(ids)) == len(ids), "duplicate ids across lists"
    by_id = {p["id"]: p for p in all_entries}
    targets = [
        by_id[i]["target"]
        for i in sorted(by_id, key=lambda i: int(i.rsplit("-", 1)[1]))
    ]
    assert targets == EXPECTED_TARGETS
    for p in pending:
        # a pending target is an honest gap, not a hidden survivor — it
        # must say why it cannot run
        assert p["status"] == "pending_anchor"
        assert p.get("anchor") is None and p.get("replacement") is None
        assert isinstance(p.get("pending_reason"), str) \
            and p["pending_reason"].strip()


def test_declared_mutation_set_is_wellformed() -> None:
    """Static validation, cheap and first: unique ids, real files, and
    anchors that occur exactly once — a non-unique anchor is a declaration
    bug, not mutation evidence."""
    ids = [m.id for m in MUTATIONS]
    assert len(ids) == len(set(ids)), "duplicate mutation ids"
    for mut in MUTATIONS:
        assert mut.id.startswith("MUT-V7-"), mut.id
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
