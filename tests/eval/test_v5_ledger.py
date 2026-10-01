"""Durable tests for the v5 requirements ledger (SPEC_V5 §27.2).

Covers the generator's contract (V5-27.03): every `V5-NN.MM` definition
is registered, duplicate/unknown ids are rejected, spec references to
undefined requirements/scenarios/gates fail, dispositions rows for
unknown requirements fail, ``locally_measured``/``qualified`` require
named executed evidence (V5-27.05), stale spec hashes and missing code
surfaces are caught, and regeneration is deterministic. The real-repo
tests assert the honest starting state: 369 requirements, 96 scenarios,
15 gates, everything ``planned``/``not_run``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from eval.v5 import ledger as L

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def run_gen(*args, root=REPO_ROOT):
    return subprocess.run(
        [sys.executable, "tools/gen_v5_ledger.py", *args,
         "--root", root],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

SPEC_MINI = """# mini v5 spec

## 01. Kernel

- V5-01.01: Reads are exact.
- V5-01.02: Writes are durable; see V5-01.01 and V5-02.01/02.

## 02. Delivery

- V5-02.01: Permits bind caller.
- V5-02.02: Permits expire.

| ID | Observable acceptance result | Primary section | Stage |
| --- | --- | --- | --- |
| E01 | read path end to end | §01–§02 | P1 |
| E02 | reopen preserves state | §02 | P2 |

| Gate | Required evidence | Claims it permits |
| --- | --- | --- |
| G5-00 Authority | §01 kernel audit | safe exposure |

| Stage | Main work | Exit / dependency |
| --- | --- | --- |
| P0 — Freeze | §01 contracts | approved scope |
| P1 — Front door | §02 delivery | G5-00 |
"""


def make_repo(tmp_path, spec=SPEC_MINI, dispositions=None, write=False):
    """Create a miniature repo root; returns (root, idx, disp)."""
    (tmp_path / "SPEC_V5.md").write_text(spec)
    root = str(tmp_path)
    if write:
        proc = run_gen("--write", root=root)
        assert proc.returncode == 0, proc.stderr + proc.stdout
    idx = L.load_specs(root)
    dpath = tmp_path / "eval" / "v5" / "dispositions_v5.json"
    if dpath.exists():
        disp = L.load_dispositions(str(dpath))
    elif dispositions is not None:
        disp = L.load_dispositions_from_text(dispositions, "<test>")
    else:
        disp = L.Dispositions()
    return root, idx, disp


# ---------------------------------------------------------------------------
# real-spec coverage
# ---------------------------------------------------------------------------


def test_real_spec_all_requirements_parsed():
    idx = L.load_specs(REPO_ROOT)
    # independent recount: definition lines in the file, unique ids
    raw = open(os.path.join(REPO_ROOT, "SPEC_V5.md")).read()
    def_lines = re.findall(r"^- V5-(\d{2}\.\d{2}):", raw, re.M)
    assert len(idx.requirements) == len(def_lines) == 369
    assert len(set(def_lines)) == len(def_lines)
    assert idx.duplicates == []


def test_real_spec_scenarios_and_gates():
    idx = L.load_specs(REPO_ROOT)
    assert sorted(idx.scenarios) == [f"E{n:02d}" for n in range(1, 97)]
    assert sorted(idx.gates) == [f"G5-{n:02d}" for n in range(15)]
    assert sorted(idx.stages) == [f"P{n}" for n in range(7)]
    # every scenario anchors to at least one real section
    for sc in idx.scenarios.values():
        assert sc.sections, sc.sc_id
        assert all(s in idx.section_titles for s in sc.sections)


def test_real_spec_honest_start_state():
    """The honesty rule: with an EMPTY overlay every requirement derives
    to ``planned``/``not_run`` — measurement claims exist only where a
    disposition names executed evidence (V5-00.05). The live overlay now
    carries real evidence entries; the seed-state invariant is checked
    against an empty Dispositions, not the curated file."""
    idx = L.load_specs(REPO_ROOT)
    reg = L.build_registry(idx, L.Dispositions())
    for rid, r in reg.items():
        assert r["status"] == "planned", rid
        assert r["implementation_status"] == "planned", rid
        assert r["qualification_status"] == "not_run", rid
        assert r["owner"] == "unset", rid
        assert r["executed_evidence"] == [], rid
        assert r["code_surfaces"] == [], rid
        assert r["blocker"], rid
        assert r["stage"] in L.STAGES + ("",), rid
    # inherited C/D ids resolve for curated overlay validation
    assert "C01" in idx.cd_scenarios and "D01" in idx.cd_scenarios


def test_live_overlay_claims_carry_evidence():
    """The curated overlay's measurement claims name executed evidence —
    the honesty gate E72 relies on (locally_measured/qualified without
    executed_evidence must fail validate())."""
    idx = L.load_specs(REPO_ROOT)
    disp = L.load_dispositions(
        os.path.join(REPO_ROOT, L.DEFAULT_DISPOSITIONS_PATH))
    reg = L.build_registry(idx, disp)
    claimed = [
        rid for rid, r in reg.items()
        if r["qualification_status"] in ("locally_measured", "qualified")
    ]
    assert claimed, "expected the overlay to carry measured claims"
    for rid in claimed:
        assert reg[rid]["executed_evidence"], rid
    issues = L.validate(idx, disp, root=REPO_ROOT)
    assert issues == [], issues[:5]


def test_real_spec_stage_inference_spot_checks():
    idx = L.load_specs(REPO_ROOT)
    sec_stage = L.infer_section_stages(idx)
    reqs = idx.requirements
    # §30 memory-ops: cited by P1/P3 work cells and P1..P3 scenarios -> P1
    assert L.infer_requirement_stage(idx, reqs["V5-30.05"], sec_stage) == (
        "P1", "stage_table")
    # §35 shadow learning: only P5 names it
    assert L.infer_requirement_stage(idx, reqs["V5-35.01"], sec_stage)[0] == "P5"
    # V5-04.01's own text assigns the mapping work to P0
    assert L.infer_requirement_stage(idx, reqs["V5-04.01"], sec_stage) == (
        "P0", "text")
    # §27 traceability is named by the P0 work cell
    assert L.infer_requirement_stage(idx, reqs["V5-27.03"], sec_stage) == (
        "P0", "stage_table")
    # §25 release-gate discipline binds per release level, not one stage
    assert L.infer_requirement_stage(idx, reqs["V5-25.01"], sec_stage) == (
        "", "fallback")


def test_real_spec_gate_and_scenario_anchors():
    idx = L.load_specs(REPO_ROOT)
    reqs = idx.requirements
    # G5-13's evidence cell names §30; G5-14's names §32
    assert "G5-13" in L.infer_requirement_gates(idx, reqs["V5-30.22"])
    assert "G5-14" in L.infer_requirement_gates(idx, reqs["V5-32.01"])
    # §07 ingest is exercised by E02 (§06–§08), E07–E11, and E22 (§07,§09)
    assert L.infer_requirement_scenarios(idx, reqs["V5-07.01"]) == [
        "E02", "E07", "E08", "E09", "E10", "E11", "E22"]
    # E72's "All" stage spans §25–§27 without pinning a stage
    assert idx.scenarios["E72"].stage == "All"
    assert "V5-27.01" in [
        r.req_id for r in reqs.values()
        if "E72" in L.infer_requirement_scenarios(idx, r)]


def test_real_repo_check_passes():
    proc = run_gen("--check")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "369 requirements" in proc.stdout
    assert "96 scenarios" in proc.stdout
    assert "15 gates" in proc.stdout


# ---------------------------------------------------------------------------
# parsing primitives
# ---------------------------------------------------------------------------


def test_anchor_shorthand_expansion():
    assert L.anchors_in("V5-13.02/13.05") == ["V5-13.02", "V5-13.05"]
    assert L.anchors_in("V5-10.05/11.04") == ["V5-10.05", "V5-11.04"]
    assert L.eids_in("E01–E06/E17–E21") == [
        f"E{n:02d}" for n in list(range(1, 7)) + list(range(17, 22))]
    assert L.gates_in("G5-00/01/02") == ["G5-00", "G5-01", "G5-02"]
    # subsection ranges stay inside their section; top-level ranges expand
    assert L.sections_in("§05.12–05.13") == [5]
    assert L.sections_in("§10–§12") == [10, 11, 12]
    assert L.sections_in("§30.1, §30.6") == [30]


# ---------------------------------------------------------------------------
# rejection contract (V5-27.03)
# ---------------------------------------------------------------------------


def test_duplicate_definition_rejected(tmp_path):
    dup = SPEC_MINI + "\n## 09. Later\n\n- V5-01.01: again.\n"
    _root, idx, disp = make_repo(tmp_path, spec=dup)
    issues = L.validate(idx, disp)
    assert any("duplicate spec definition" in i and "V5-01.01" in i
               for i in issues)


def test_undefined_reference_rejected(tmp_path):
    bad = SPEC_MINI + "\n## 09. Later\n\n- V5-09.01: see V5-99.99 and E99.\n"
    _root, idx, disp = make_repo(tmp_path, spec=bad)
    issues = L.validate(idx, disp)
    assert any("undefined requirement V5-99.99" in i for i in issues)
    assert any("undefined scenario E99" in i for i in issues)


def test_section_mismatch_rejected(tmp_path):
    bad = SPEC_MINI + "\n## 09. Later\n\n- V5-07.03: wrong section.\n"
    _root, idx, disp = make_repo(tmp_path, spec=bad)
    issues = L.validate(idx, disp)
    assert any("id/section mismatch" in i for i in issues)


def test_disposition_unknown_requirement_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {"V5-77.01": {"owner": "x"}},
    }))
    issues = L.validate(idx, disp)
    assert any("unknown requirement V5-77.01" in i for i in issues)


def test_disposition_duplicate_key_rejected(tmp_path):
    with pytest.raises(L.LedgerError):
        L.load_dispositions_from_text(
            '{"requirements": {}, "requirements": {}}')


def test_qualified_requires_executed_evidence(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V5-01.01": {
                "implementation_status": "implemented",
                "qualification_status": "qualified",
            },
        },
    }))
    issues = L.validate(idx, disp)
    assert any("named executed evidence" in i for i in issues)


def test_locally_measured_requires_executed_evidence(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V5-01.01": {"qualification_status": "locally_measured"},
        },
    }))
    issues = L.validate(idx, disp)
    assert any("locally_measured" in i for i in issues)


def test_implemented_unmeasured_is_honest(tmp_path):
    """V5-27.05: implemented + not_run is a legal, distinct state."""
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V5-01.01": {"implementation_status": "implemented"},
        },
    }))
    issues = L.validate(idx, disp)
    assert issues == []
    reg = L.build_registry(idx, disp)
    r = reg["V5-01.01"]
    assert r["status"] == "implemented_unmeasured"
    assert r["qualification_status"] == "not_run"


def test_measured_but_unqualified_is_honest(tmp_path):
    """The V5-27.05 middle state: measured locally, still not qualified."""
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V5-01.01": {
                "implementation_status": "implemented",
                "qualification_status": "locally_measured",
                "executed_evidence": ["eval/v5/exp1.json"],
                "evidence_type": ["LOCAL"],
            },
        },
    }))
    issues = L.validate(idx, disp)
    assert issues == []
    reg = L.build_registry(idx, disp)
    assert reg["V5-01.01"]["status"] == "locally_measured"


def test_qualified_full_path(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V5-01.01": {
                "implementation_status": "implemented",
                "qualification_status": "qualified",
                "executed_evidence": ["tests/eval/test_x.py::test_y"],
                "evidence_type": ["LOCAL"],
                "gate": ["G5-00"],
                "scenarios": ["E02", "C01"],
            },
        },
    }))
    assert L.validate(idx, disp) == []
    r = L.build_registry(idx, disp)["V5-01.01"]
    assert r["status"] == "qualified"
    # spec-anchored E01 + curated E02/C01 both land in scenarios
    assert r["scenarios"] == ["C01", "E01", "E02"]
    assert r["spec_scenarios"] == ["E01"]
    assert "G5-00" in r["gate"]


def test_stale_spec_hash_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": {"SPEC_V5.md": "0" * 64},
        "requirements": {},
    }))
    issues = L.validate(idx, disp)
    assert any("stale spec hash" in i for i in issues)


def test_missing_code_surface_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V5-01.01": {"code_surfaces": ["verbatim/no_such_file.py"]},
        },
    }))
    issues = L.validate(idx, disp, root=_root)
    assert any("does not exist" in i for i in issues)


def test_unknown_disposition_key_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {"V5-01.01": {"statsu": "planned"}},
    }))
    issues = L.validate(idx, disp)
    assert any("unknown key 'statsu'" in i for i in issues)


def test_unknown_scenario_in_overlay_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {"V5-01.01": {"scenarios": ["E77"]}},
    }))
    issues = L.validate(idx, disp)
    assert any("unknown scenario E77" in i for i in issues)


# ---------------------------------------------------------------------------
# determinism + end-to-end
# ---------------------------------------------------------------------------


def test_regeneration_deterministic(tmp_path):
    root, idx, disp = make_repo(tmp_path, write=True)
    first = {}
    for rel in (L.DEFAULT_JSON_PATH, L.DEFAULT_SUMMARY_PATH,
                L.DEFAULT_DISPOSITIONS_PATH):
        first[rel] = open(os.path.join(root, rel)).read()
    proc = run_gen("--write", root=root)
    assert proc.returncode == 0, proc.stderr
    for rel, content in first.items():
        assert open(os.path.join(root, rel)).read() == content
    assert run_gen("--check", root=root).returncode == 0


def test_check_detects_stale_artifacts(tmp_path):
    root, _idx, _d = make_repo(tmp_path, write=True)
    jpath = os.path.join(root, L.DEFAULT_JSON_PATH)
    doc = json.loads(open(jpath).read())
    doc["requirements"]["V5-01.01"]["status"] = "qualified"
    open(jpath, "w").write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    proc = run_gen("--check", root=root)
    assert proc.returncode == 1 and "stale" in proc.stdout


def test_check_detects_spec_drift(tmp_path):
    root, _idx, _d = make_repo(tmp_path, write=True)
    with open(os.path.join(root, "SPEC_V5.md"), "a") as f:
        f.write("\n## 09. Later\n\n- V5-09.01: new work.\n")
    proc = run_gen("--check", root=root)
    assert proc.returncode == 1
    assert "stale spec hash" in proc.stdout


def test_summary_tables_render(tmp_path):
    _root, idx, disp = make_repo(tmp_path)
    reg = L.build_registry(idx, disp)
    md = L.render_summary(idx, reg, disp)
    assert "## By section" in md and "## Gates" in md
    assert "E01" in md and "G5-00" in md
    led = L.render_ledger(idx, reg, disp)
    assert led["summary"]["total"] == 4
    assert led["scenarios"]["E01"]["anchored_requirements"] == [
        "V5-01.01", "V5-01.02", "V5-02.01", "V5-02.02"]
