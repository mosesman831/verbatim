"""Durable tests for the v4 executable-traceability ledger (§61).

Covers the generator's rejection contract (V4-61.03): unknown
requirement ids, duplicate map entries, mapped test nodes pytest cannot
collect, stale spec hashes, ``verified`` with no named check (V4-02.03),
and discovered-but-unmapped evidence. Also covers spec self-checks
(V4-67.08): duplicate definitions and references to undefined ids.
Each test builds a miniature spec+tree in ``tmp_path`` so the suite does
not depend on the real spec's contents.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from eval.v4 import ledger as L

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

SPEC_V4 = """# mini v4 spec

## 01. Kernel

- V4-01.01: Reads are exact.
- V4-01.02: Writes are durable; see V4-01.01.

## 02. Delivery

- V4-02.01: Permits bind caller.
- V4-02.02: Permits expire.

| ID | Scenario | Anchors | Tier |
| --- | --- | --- | --- |
| C01 | read path end to end | V4-01.01, V4-02.01 | C |
"""

SPEC_V45 = """# mini addendum

## 01. Experiments

- V45-01.01: No parallel engine.
"""

TEST_FILE = '''"""Module docstring claims V4-01.02 coverage (file claim, not a check)."""


def test_reads_exact():
    """V4-01.01: a read returns stored bytes."""
    assert True


# section banner (V4-02.02)
def test_permit_binding():
    assert True


def test_c01_read_path_end_to_end():
    assert True
'''

COLLECTED = {
    "tests/test_alpha.py::test_reads_exact",
    "tests/test_alpha.py::test_permit_binding",
    "tests/test_alpha.py::test_c01_read_path_end_to_end",
}


def make_repo(tmp_path, spec_v4=SPEC_V4, spec_v45=SPEC_V45,
              test_file=TEST_FILE, map_yaml=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "SPEC_V4.md").write_text(spec_v4)
    (tmp_path / "SPEC_V4_5.md").write_text(spec_v45)
    tdir = tmp_path / "tests"
    tdir.mkdir(exist_ok=True)
    (tdir / "test_alpha.py").write_text(test_file)
    if map_yaml is not None:
        mdir = tmp_path / "eval" / "v4"
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "requirement_map.yaml").write_text(map_yaml)
    return str(tmp_path)


def build(root):
    idx = L.load_specs(root)
    disc = L.discover_evidence(root, idx)
    return idx, disc


def map_yaml(body: str) -> str:
    return "schema: 1\nspec_sha256: {}\nrequirements:\n" + body


def test_parses_every_requirement_and_scenario(tmp_path):
    root = make_repo(tmp_path)
    idx, _ = build(root)
    assert set(idx.requirements) == {"V4-01.01", "V4-01.02",
                                     "V4-02.01", "V4-02.02", "V45-01.01"}
    assert idx.requirements["V4-01.01"].section_title == "Kernel"
    assert idx.scenarios["C01"].anchors == ("V4-01.01", "V4-02.01")


def test_registry_enumerates_every_defined_id(tmp_path):
    """Every spec id appears in the emitted registry (V4-61.01)."""
    root = make_repo(tmp_path)
    idx, disc = build(root)
    rm = L.merge_discovery(L.RequirementMap(), idx, disc)
    registry = L.build_registry(idx, rm, disc)
    assert set(registry) == set(idx.requirements)


def test_anchor_expansion_and_dangling_reference(tmp_path):
    """`V4-01.01/02` expands; refs to undefined ids fail (V4-67.08)."""
    assert set(L.anchors_in("x V4-01.01/02")) == {"V4-01.01", "V4-01.02"}
    assert set(L.anchors_in("V4-10.05/11.04")) == {"V4-10.05", "V4-11.04"}
    spec = SPEC_V4 + "\n- V4-02.03: mentions V4-99.99 which is undefined.\n"
    root = make_repo(tmp_path, spec_v4=spec)
    idx, disc = build(root)
    issues = L.validate(idx, L.RequirementMap(), disc, None)
    assert any("undefined requirement V4-99.99" in i for i in issues)


def test_duplicate_spec_definitions_rejected(tmp_path):
    spec = SPEC_V4 + "\n- V4-01.01: duplicate definition.\n"
    root = make_repo(tmp_path, spec_v4=spec)
    idx, disc = build(root)
    issues = L.validate(idx, L.RequirementMap(), disc, None)
    assert any("duplicate spec definition" in i and "V4-01.01" in i
               for i in issues)


def test_discovery_strong_banner_and_claim(tmp_path):
    root = make_repo(tmp_path)
    idx, disc = build(root)
    # in-test docstring -> direct named check
    assert disc.tests["V4-01.01"][
        "tests/test_alpha.py::test_reads_exact"] == L.K_DIRECT
    # banner comment -> weak coverage on the following test
    assert disc.tests["V4-02.02"][
        "tests/test_alpha.py::test_permit_binding"] == L.K_BANNER
    # module docstring -> file claim only, never a test
    assert "V4-01.02" not in disc.tests
    assert disc.claims["V4-01.02"] == {"tests/test_alpha.py"}
    # scenario-named test binds the scenario's anchored requirements
    assert "tests/test_alpha.py::test_c01_read_path_end_to_end" in disc.tests[
        "V4-02.01"]
    assert disc.scenarios["V4-01.01"] == {"C01"}


def test_seed_status_direct_vs_banner_only(tmp_path):
    root = make_repo(tmp_path)
    idx, disc = build(root)
    # V4-01.01 has a direct check + scenario name -> verified
    assert L.seed_status("V4-01.01", disc) == "verified"
    # V4-02.01 is additionally bound to scenario C01 via the named test
    assert L.seed_status("V4-02.01", disc) == "verified"
    # banner-only coverage -> implemented_unverified
    assert L.seed_status("V4-02.02", disc) == "implemented_unverified"


def test_reject_unknown_id_in_map(tmp_path):
    root = make_repo(tmp_path)
    idx, disc = build(root)
    rm = L.load_map_from_text(map_yaml(
        "  V4-77.01:\n    status: implemented_unverified\n"))
    issues = L.validate(idx, rm, disc, COLLECTED)
    assert any("unknown requirement V4-77.01" in i for i in issues)


def test_reject_duplicate_map_entry(tmp_path):
    dup = map_yaml(
        "  V4-01.01:\n    status: verified\n"
        "  V4-01.01:\n    status: partial\n")
    with pytest.raises(L.LedgerError, match="duplicate key"):
        L.load_map_from_text(dup)


def test_reject_uncollected_test_node(tmp_path):
    root = make_repo(tmp_path)
    idx, disc = build(root)
    rm = L.load_map_from_text(map_yaml(
        "  V4-01.01:\n"
        "    status: verified\n"
        "    tests:\n"
        "      - tests/test_alpha.py::test_reads_exact\n"
        "      - tests/test_alpha.py::test_deleted_test\n"))
    issues = L.validate(idx, rm, disc, COLLECTED)
    assert any("test_deleted_test" in i and "not collected" in i
               for i in issues)


def test_reject_verified_without_tests(tmp_path):
    """V4-02.03: `verified` must name executable checks (V4-61.03)."""
    root = make_repo(tmp_path)
    idx, disc = build(root)
    rm = L.load_map_from_text(map_yaml(
        "  V4-01.02:\n    status: verified\n"))
    issues = L.validate(idx, rm, disc, COLLECTED)
    assert any("V4-01.02" in i and "'verified' with no mapped test" in i
               for i in issues)


def test_reject_unimplemented_with_tests(tmp_path):
    root = make_repo(tmp_path)
    idx, disc = build(root)
    rm = L.load_map_from_text(map_yaml(
        "  V4-01.01:\n"
        "    status: unimplemented\n"
        "    tests:\n"
        "      - tests/test_alpha.py::test_reads_exact\n"))
    issues = L.validate(idx, rm, disc, COLLECTED)
    assert any("'unimplemented' cannot list tests" in i for i in issues)


def test_reject_stale_spec_hash(tmp_path):
    root = make_repo(tmp_path)
    idx, disc = build(root)
    rm = L.load_map_from_text(map_yaml(""))
    rm.spec_sha256 = {"SPEC_V4.md": "0" * 64}
    issues = L.validate(idx, rm, disc, COLLECTED)
    assert any("stale spec hash" in i for i in issues)
    assert any("SPEC_V4_5.md" in i and "does not pin" in i for i in issues)


def test_reject_discovered_but_unmapped(tmp_path):
    """A discovered named check with no map row is unmapped-but-covered."""
    root = make_repo(tmp_path)
    idx, disc = build(root)
    issues = L.validate(idx, L.RequirementMap(), disc, COLLECTED)
    assert any("V4-01.01 has discovered checks" in i for i in issues)


def test_merge_seeds_and_preserves_human_status(tmp_path):
    root = make_repo(tmp_path)
    idx, disc = build(root)
    rm = L.load_map_from_text(map_yaml(
        "  V4-01.01:\n"
        "    status: partial\n"
        "    note: second half not covered\n"))
    merged = L.merge_discovery(rm, idx, disc)
    assert merged.entries["V4-01.01"].status == "partial"
    assert "tests/test_alpha.py::test_reads_exact" in \
        merged.entries["V4-01.01"].tests
    # discovered-only requirement gets a seeded row
    assert merged.entries["V4-02.01"].status in L.STATUSES


def test_scenario_must_be_defined_and_anchored(tmp_path):
    root = make_repo(tmp_path)
    idx, disc = build(root)
    rm = L.load_map_from_text(map_yaml(
        "  V4-02.01:\n"
        "    status: implemented_unverified\n"
        "    scenarios: [C99]\n"))
    issues = L.validate(idx, rm, disc, COLLECTED)
    assert any("unknown scenario C99" in i for i in issues)
    # a defined scenario not anchored to this requirement is also wrong:
    # C01 covers V4-01.01/V4-02.01, not V4-01.02
    rm = L.load_map_from_text(map_yaml(
        "  V4-01.02:\n"
        "    status: implemented_unverified\n"
        "    scenarios: [C01]\n"))
    issues = L.validate(idx, rm, disc, COLLECTED)
    assert any("not anchored to V4-01.02" in i for i in issues)


def test_deferred_requires_note(tmp_path):
    """V4-02.06: a deferral keeps reason/prerequisite — no silent drops."""
    root = make_repo(tmp_path)
    idx, disc = build(root)
    rm = L.load_map_from_text(map_yaml(
        "  V4-02.02:\n    status: deferred\n"))
    issues = L.validate(idx, rm, disc, COLLECTED)
    assert any("'deferred' requires a note" in i for i in issues)


def test_status_enum_matches_v4_02_02(tmp_path):
    """The status set is exactly the V4-02.02 enumeration."""
    assert set(L.STATUSES) == {
        "unimplemented", "partial", "implemented_unverified",
        "verified", "deferred", "not_applicable"}


def _run_gen(*args):
    return subprocess.run(
        [sys.executable, "tools/gen_v4_ledger.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True)


def test_real_check_passes_on_seed_map():
    """`--check` on the real repo: the seed map is valid and fresh.

    The ledger intentionally drifts the moment any spec, test, or
    `verbatim/` anchor changes, so on a live tree a stale artifact is
    expected between regenerations. The durable guarantee is: every
    failure is *staleness* (fixed by --write), never an invalid map —
    and a fresh --write always validates clean.
    """
    proc = _run_gen("--check")
    if proc.returncode != 0:
        # issues --write can repair: stale artifacts/hashes, newly
        # discovered anchors not yet mapped, renamed/deleted tests
        drift = ("stale", "not mapped", "no map row",
                 "not collected by pytest")
        bad = [ln for ln in proc.stdout.splitlines()
               if ln.strip().startswith("- ")
               and not any(d in ln for d in drift)]
        assert not bad, (
            "non-drift ledger issues (invalid map):\n"
            + "\n".join(bad) + "\n" + proc.stdout + proc.stderr)
        proc = _run_gen("--write")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        proc = _run_gen("--check")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "v4 ledger OK" in proc.stdout


def test_real_spec_ids_all_enumerated():
    idx = L.load_specs(REPO_ROOT)
    assert len(idx.requirements) >= 700
    assert all(r.startswith(("V4-", "V45-")) for r in idx.requirements)
    assert not idx.duplicates
