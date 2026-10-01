"""Durable tests for the v6 requirements ledger (SPEC_V6, V6-00.06/04.01).

Covers the generator's contract (V6-00.03/00.06): every `V6-NN.MM`
definition is registered, duplicate/unknown ids are rejected, spec
references to undefined requirements/scenarios/gates fail, ids outside
the declared F01–F32 / G6-00–G6-08 registries fail, dispositions rows
for unknown requirements fail, ``locally_measured``/``qualified``
require named executed evidence, stale spec hashes and missing code
surfaces are caught, regeneration is deterministic, and the ``carried``
block lists exactly the V5 rows still ``planned`` or
``implemented_unmeasured`` (V6-04.02/03 — read-only, the V5 overlay is
never modified). The real-repo tests assert the honest state: 69
requirements, 32 scenarios, 9 gates, 8 phases, everything
``planned``/``not_run``, and the carried V5 tail equal to the live
unfinished set — 45 rows after the 2026-09-22 V6-04.02/03 closure
pass (42 re-deferred with owner+reason, 3 still honestly
unmeasured), each carrying a written disposition.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

from eval.v6 import ledger as L

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def run_gen(*args, root=REPO_ROOT):
    return subprocess.run(
        [sys.executable, "tools/gen_v6_ledger.py", *args,
         "--root", root],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


SPEC_MINI = """# mini v6 spec

## 01. Kernel

- V6-01.01: Reads are exact.
- V6-01.02: Writes are durable; see V6-01.01 and V6-02.01/02; exercised by F01.

## 02. Delivery

- V6-02.01: Permits bind caller.
- V6-02.02: Permits expire.

| ID | Observable result |
| --- | --- |
| F01 | read path end to end |
| F02 | reopen preserves state |

| Gate | Permits |
| --- | --- |
| G6-00 Authority | F01 plus audit |

| Phase | Content | Exit |
| --- | --- | --- |
| P0 | contracts freeze | ledger parses |
| P1 | delivery work | F01–F02 plus G6-00 |
"""

V5_TAIL_FIXTURE = json.dumps({
    "spec_sha256": {"SPEC_V5.md": "0" * 64},
    "requirements": {
        "V5-01.01": {"implementation_status": "implemented",
                     "note": "tests pass; evidence never registered"},
        "V5-01.02": {
            "implementation_status": "implemented",
            "qualification_status": "locally_measured",
            "evidence_type": ["LOCAL"],
            "executed_evidence": ["pytest tests/x.py"],
        },
        "V5-01.03": {"note": "still planned"},
        "V5-01.04": {
            "implementation_status": "implemented",
            "qualification_status": "qualified",
            "evidence_type": ["LOCAL"],
            "executed_evidence": ["pytest tests/y.py"],
        },
        "V5-01.05": {"qualification_status": "failed",
                     "note": "named loss"},
    },
})


def make_repo(tmp_path, spec=SPEC_MINI, dispositions=None, write=False,
              v5_dispositions=V5_TAIL_FIXTURE):
    """Create a miniature repo root; returns (root, idx, disp)."""
    (tmp_path / "SPEC_V6.md").write_text(spec)
    if v5_dispositions is not None:
        v5dir = tmp_path / "eval" / "v5"
        v5dir.mkdir(parents=True, exist_ok=True)
        (v5dir / "dispositions_v5.json").write_text(v5_dispositions)
    root = str(tmp_path)
    if write:
        proc = run_gen("--write", root=root)
        assert proc.returncode == 0, proc.stderr + proc.stdout
    idx = L.load_specs(root)
    dpath = tmp_path / "eval" / "v6" / "dispositions_v6.json"
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
    raw = open(os.path.join(REPO_ROOT, "SPEC_V6.md")).read()
    def_lines = re.findall(r"^- V6-(\d{2}\.\d{2}):", raw, re.M)
    assert len(idx.requirements) == len(def_lines) == 69
    assert len(set(def_lines)) == len(def_lines)
    assert idx.duplicates == []


def test_real_spec_scenarios_and_gates():
    idx = L.load_specs(REPO_ROOT)
    assert sorted(idx.scenarios) == [f"F{n:02d}" for n in range(1, 33)]
    assert sorted(idx.gates) == [f"G6-{n:02d}" for n in range(9)]
    assert sorted(idx.stages) == [f"P{n}" for n in range(8)]


def test_real_spec_honest_start_state():
    """The honesty rule (V6-00.06): every requirement derives to
    ``planned``/``not_run`` — the live seed overlay carries an explicit
    planned/not_run + spec-section note per row, and an empty overlay
    derives the same state with no note."""
    idx = L.load_specs(REPO_ROOT)
    reg = L.build_registry(idx, L.Dispositions())
    for rid, r in reg.items():
        assert r["status"] == "planned", rid
        assert r["implementation_status"] == "planned", rid
        assert r["qualification_status"] == "not_run", rid
        assert r["owner"] == "unset", rid
        assert r["executed_evidence"] == [], rid
        assert r["code_surfaces"] == [], rid
        assert r["stage"] in L.STAGES + ("",), rid
    # the seeded overlay covers every requirement id; rows still
    # ``planned`` carry the per-section note, measured rows carry
    # their executed-evidence note — every row reports SOME note.
    disp = L.load_dispositions(
        os.path.join(REPO_ROOT, L.DEFAULT_DISPOSITIONS_PATH))
    assert set(disp.entries) == set(idx.requirements)
    reg = L.build_registry(idx, disp)
    for rid, r in reg.items():
        assert r["note"].strip(), rid
        if r["status"] == "planned":
            assert f"§{r['section']:02d}" in r["note"], rid


def test_live_overlay_validate_clean():
    idx = L.load_specs(REPO_ROOT)
    disp = L.load_dispositions(
        os.path.join(REPO_ROOT, L.DEFAULT_DISPOSITIONS_PATH))
    issues = L.validate(idx, disp, root=REPO_ROOT)
    assert issues == [], issues[:5]


def test_real_spec_phase_and_anchor_inference():
    """The spec's own planning map: §09 phase exits name F-scenarios;
    requirements bind them through their phase, plus envelopes and
    G6-NN text mentions."""
    idx = L.load_specs(REPO_ROOT)
    sc_stage = L.infer_scenario_stages(idx)
    reqs = idx.requirements
    # phase exits bind their scenarios
    assert sc_stage["F17"] == ("P1", "phase_exit")
    assert sc_stage["F29"] == ("P5", "phase_exit")
    assert sc_stage["F31"] == ("P6", "phase_exit")
    # §02.1 causal-barrier program -> P1 -> F17–F20 -> G6-01
    d = reqs["V6-02.07"]
    stage, src = L.infer_requirement_stage(idx, d, sc_stage)
    assert (stage, src) == ("P1", "fallback")
    sc = L.infer_requirement_scenarios(idx, d, sc_stage, stage)
    assert sc == ["F17", "F18", "F19", "F20"]
    assert L.infer_requirement_gates(idx, d, sc) == ["G6-01"]
    # envelope binding: A0 -> F09 + G6-01; A3 -> G6-07
    d = reqs["V6-02.13"]
    assert "F09" in L.infer_requirement_scenarios(
        idx, d, sc_stage, "P2")
    assert "G6-01" in L.infer_requirement_gates(idx, d, [])
    assert "G6-07" in L.infer_requirement_gates(
        idx, reqs["V6-02.15"], [])
    # §03.1 neural path -> P5 -> F29/F30 -> G6-02
    d = reqs["V6-03.06"]
    stage, _ = L.infer_requirement_stage(idx, d, sc_stage)
    assert stage == "P5"
    assert L.infer_requirement_gates(
        idx, d, L.infer_requirement_scenarios(idx, d, sc_stage, stage)
    ) == ["G6-02"]
    # §04 tail closure -> P6 -> F31/F32 -> G6-08 (+G6-05, same phase)
    d = reqs["V6-04.02"]
    stage, _ = L.infer_requirement_stage(idx, d, sc_stage)
    assert stage == "P6"
    gates = L.infer_requirement_gates(
        idx, d, L.infer_requirement_scenarios(idx, d, sc_stage, stage))
    assert "G6-08" in gates and "G6-05" in gates
    # service surface -> P3 -> G6-04
    d = reqs["V6-05.01"]
    gates = L.infer_requirement_gates(
        idx, d, L.infer_requirement_scenarios(idx, d, sc_stage, "P3"))
    assert "G6-04" in gates
    # V6-08.01 names gates directly; the F01–F08 range does not bind
    d = reqs["V6-08.01"]
    assert L.infer_requirement_scenarios(idx, d, sc_stage, "") == []
    assert L.infer_requirement_gates(idx, d, []) == [
        "G6-00", "G6-01", "G6-03", "G6-04", "G6-05", "G6-06"]


def test_real_repo_check_passes():
    proc = run_gen("--check")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "69 requirements" in proc.stdout
    assert "32 scenarios" in proc.stdout
    assert "9 gates" in proc.stdout
    # the printed carried-tail size must equal the live computed tail —
    # never a frozen constant (the V6-04.02/03 pass legitimately shrinks it)
    tail = L.load_carried(REPO_ROOT)
    assert f"carried V5 tail {len(tail.rows)}" in proc.stdout


# ---------------------------------------------------------------------------
# carried V5 tail (V6-04.02/03)
# ---------------------------------------------------------------------------


def test_carried_lists_only_unfinished_v5_rows(tmp_path):
    root, _idx, _d = make_repo(tmp_path)
    tail = L.load_carried(root)
    assert tail.missing is False
    by_id = {r["id"]: r for r in tail.rows}
    # implemented + not_run -> implemented_unmeasured (carried);
    # empty entry -> planned (carried); measured/qualified/failed -> not
    assert sorted(by_id) == ["V5-01.01", "V5-01.03"]
    assert by_id["V5-01.01"]["status"] == "implemented_unmeasured"
    assert by_id["V5-01.01"]["note"] == "tests pass; evidence never registered"
    assert by_id["V5-01.03"]["status"] == "planned"
    for row in tail.rows:
        assert row["status"] in L.CARRIED_STATUSES


def test_carried_missing_source_is_honest(tmp_path):
    root, _idx, _d = make_repo(tmp_path, v5_dispositions=None)
    tail = L.load_carried(root)
    assert tail.missing is True and tail.rows == []
    led = L.render_ledger(_idx, L.build_registry(_idx, _d), _d,
                          carried=tail)
    assert led["carried"]["missing"] is True
    assert led["carried"]["total"] == 0


def test_carried_real_repo_v5_tail():
    """The real V5 tail is exactly the rows still unfinished after the
    V6-04.02/03 re-disposition pass — carried read-only, and every
    carried row MUST carry a written disposition note (no silent
    drops)."""
    tail = L.load_carried(REPO_ROOT)
    assert tail.missing is False
    counts = tail.by_status()
    # 2026-09-22 V6-04.02/03 closure: the carried tail is exactly the
    # published deferral list — 20 rows `deferred` with owner+reason,
    # kept `planned` so they stay visibly open (never summarized away).
    assert counts == {"planned": 20, "implemented_unmeasured": 0}
    assert len(tail.rows) == 20
    assert all(r["id"].startswith("V5-") for r in tail.rows)
    # every row reports its id + current status + a written
    # disposition (V6-04.02/03)
    for r in tail.rows:
        assert r["status"] in L.CARRIED_STATUSES
        assert r["implementation_status"] in L.IMPLEMENTATION_STATUSES
        assert r["qualification_status"] in L.QUALIFICATION_STATUSES
        assert r["note"].strip(), (
            f"carried row {r['id']} has no disposition note")
    # embedded in the generated ledger
    disp = L.load_dispositions(
        os.path.join(REPO_ROOT, L.DEFAULT_DISPOSITIONS_PATH))
    idx = L.load_specs(REPO_ROOT)
    led = L.render_ledger(idx, L.build_registry(idx, disp), disp,
                          carried=tail)
    assert led["carried"]["total"] == len(tail.rows)
    assert led["carried"]["source"] == L.V5_DISPOSITIONS_PATH
    assert led["carried"]["source_sha256"] == hashlib.sha256(
        open(os.path.join(REPO_ROOT, L.V5_DISPOSITIONS_PATH), "rb")
        .read()).hexdigest()


def test_write_never_touches_v5_overlay(tmp_path):
    """--write regenerates V6 artifacts; the V5 dispositions file is
    read-only input and must come out byte-identical (V6-04.01)."""
    root, _idx, _d = make_repo(tmp_path, write=True)
    v5_path = os.path.join(root, "eval", "v5", "dispositions_v5.json")
    before = open(v5_path).read()
    proc = run_gen("--write", root=root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert open(v5_path).read() == before
    # and the carried block made it into the generated ledger
    led = json.loads(open(os.path.join(
        root, "eval", "v6", "ledger_v6.json")).read())
    assert led["carried"]["total"] == 2
    assert sorted(r["id"] for r in led["carried"]["rows"]) == [
        "V5-01.01", "V5-01.03"]


# ---------------------------------------------------------------------------
# parsing primitives
# ---------------------------------------------------------------------------


def test_anchor_shorthand_expansion():
    assert L.anchors_in("V6-04.02/03") == ["V6-04.02", "V6-04.03"]
    assert L.anchors_in("V6-01.05/06.03") == ["V6-01.05", "V6-06.03"]
    assert L.fids_in("F01, F10–F12, F15") == [
        "F01", "F10", "F11", "F12", "F15"]
    assert L.gates_in("G6-01/03/04") == ["G6-01", "G6-03", "G6-04"]
    # a declared range is a namespace span: reference expansion covers
    # it, per-item binding does not
    assert L.gates_in("G6-00–G6-08") == [
        f"G6-{n:02d}" for n in range(9)]
    assert L._gate_mentions("G6-00–G6-08") == []
    assert L.fids_in("F01–F32") == [f"F{n:02d}" for n in range(1, 33)]
    assert L._f_mentions("F01–F32") == []


def test_envelope_tokens_longest_match():
    assert L.envelopes_in("A0 cache-off and A0-cache") == [
        "A0", "A0-cache"]
    assert L.envelopes_in("A0/A1 hashing, A0-neural, A3") == [
        "A0", "A0-neural", "A1", "A3"]
    assert L.envelopes_in("no envelopes") == []


# ---------------------------------------------------------------------------
# rejection contract (V6-00.03/00.06)
# ---------------------------------------------------------------------------


def test_duplicate_definition_rejected(tmp_path):
    dup = SPEC_MINI + "\n## 09. Later\n\n- V6-01.01: again.\n"
    _root, idx, disp = make_repo(tmp_path, spec=dup)
    issues = L.validate(idx, disp)
    assert any("duplicate spec definition" in i and "V6-01.01" in i
               for i in issues)


def test_undefined_reference_rejected(tmp_path):
    bad = SPEC_MINI + "\n## 09. Later\n\n- V6-09.01: see V6-99.99, F99, G6-09.\n"
    _root, idx, disp = make_repo(tmp_path, spec=bad)
    issues = L.validate(idx, disp)
    assert any("undefined requirement V6-99.99" in i for i in issues)
    assert any("undefined scenario F99" in i for i in issues)
    assert any("undefined gate G6-09" in i for i in issues)


def test_out_of_registry_ids_rejected(tmp_path):
    bad = SPEC_MINI + (
        "\n| F33 | beyond the declared registry |\n"
        "\n| G6-09 Extra | too far |\n"
    )
    _root, idx, disp = make_repo(tmp_path, spec=bad)
    issues = L.validate(idx, disp)
    assert any("F33" in i and "registry" in i for i in issues)
    assert any("G6-09" in i and "registry" in i for i in issues)


def test_section_mismatch_rejected(tmp_path):
    bad = SPEC_MINI + "\n## 09. Later\n\n- V6-07.03: wrong section.\n"
    _root, idx, disp = make_repo(tmp_path, spec=bad)
    issues = L.validate(idx, disp)
    assert any("id/section mismatch" in i for i in issues)


def test_disposition_unknown_requirement_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {"V6-77.01": {"owner": "x"}},
    }))
    issues = L.validate(idx, disp)
    assert any("unknown requirement V6-77.01" in i for i in issues)


def test_disposition_duplicate_key_rejected(tmp_path):
    with pytest.raises(L.LedgerError):
        L.load_dispositions_from_text(
            '{"requirements": {}, "requirements": {}}')


def test_qualified_requires_executed_evidence(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V6-01.01": {
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
            "V6-01.01": {"qualification_status": "locally_measured"},
        },
    }))
    issues = L.validate(idx, disp)
    assert any("locally_measured" in i for i in issues)


def test_implemented_unmeasured_is_honest(tmp_path):
    """V6-00.06: implemented + not_run is a legal, distinct state."""
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V6-01.01": {"implementation_status": "implemented"},
        },
    }))
    issues = L.validate(idx, disp)
    assert issues == []
    reg = L.build_registry(idx, disp)
    r = reg["V6-01.01"]
    assert r["status"] == "implemented_unmeasured"
    assert r["qualification_status"] == "not_run"


def test_measured_but_unqualified_is_honest(tmp_path):
    """The middle state: measured locally, still not qualified."""
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V6-01.01": {
                "implementation_status": "implemented",
                "qualification_status": "locally_measured",
                "executed_evidence": ["eval/v6/exp1.json"],
                "evidence_type": ["LOCAL"],
            },
        },
    }))
    issues = L.validate(idx, disp)
    assert issues == []
    reg = L.build_registry(idx, disp)
    assert reg["V6-01.01"]["status"] == "locally_measured"


def test_overlay_applies_full_path(tmp_path):
    """The overlay merges with spec inference: curated scenarios and
    gates add to the spec-anchored set, never replace it."""
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V6-01.02": {
                "implementation_status": "implemented",
                "qualification_status": "qualified",
                "executed_evidence": ["tests/eval/test_x.py::test_y"],
                "evidence_type": ["LOCAL"],
                "gate": ["G6-00"],
                "scenarios": ["E02"],
                "owner": "builder",
                "note": "curated row",
            },
        },
    }))
    assert L.validate(idx, disp) == []
    r = L.build_registry(idx, disp)["V6-01.02"]
    assert r["status"] == "qualified"
    assert r["owner"] == "builder"
    # spec-anchored F01/F02 (text mention + P1 exit) + curated E02
    assert r["spec_scenarios"] == ["F01", "F02"]
    assert r["scenarios"] == ["E02", "F01", "F02"]
    assert "G6-00" in r["gate"]


def test_stale_spec_hash_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": {"SPEC_V6.md": "0" * 64},
        "requirements": {},
    }))
    issues = L.validate(idx, disp)
    assert any("stale spec hash" in i for i in issues)


def test_missing_code_surface_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V6-01.01": {"code_surfaces": ["verbatim/no_such_file.py"]},
        },
    }))
    issues = L.validate(idx, disp, root=_root)
    assert any("does not exist" in i for i in issues)


def test_unknown_disposition_key_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {"V6-01.01": {"statsu": "planned"}},
    }))
    issues = L.validate(idx, disp)
    assert any("unknown key 'statsu'" in i for i in issues)


def test_unknown_scenario_in_overlay_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {"V6-01.01": {"scenarios": ["F77"]}},
    }))
    issues = L.validate(idx, disp)
    assert any("unknown scenario F77" in i for i in issues)


def test_unknown_gate_in_overlay_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {"V6-01.01": {"gate": ["G6-77"]}},
    }))
    issues = L.validate(idx, disp)
    assert any("unknown gate G6-77" in i for i in issues)


def test_invalid_statuses_rejected(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V6-01.01": {"implementation_status": "done"},
            "V6-01.02": {"qualification_status": "measured"},
            "V6-02.01": {"stage": "P9"},
        },
    }))
    issues = L.validate(idx, disp)
    assert any("invalid implementation_status 'done'" in i for i in issues)
    assert any("invalid qualification_status 'measured'" in i
               for i in issues)
    assert any("invalid stage 'P9'" in i for i in issues)


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
    doc["requirements"]["V6-01.01"]["status"] = "qualified"
    open(jpath, "w").write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    proc = run_gen("--check", root=root)
    assert proc.returncode == 1 and "stale" in proc.stdout


def test_check_detects_spec_drift(tmp_path):
    root, _idx, _d = make_repo(tmp_path, write=True)
    with open(os.path.join(root, "SPEC_V6.md"), "a") as f:
        f.write("\n## 09. Later\n\n- V6-09.01: new work.\n")
    proc = run_gen("--check", root=root)
    assert proc.returncode == 1
    assert "stale spec hash" in proc.stdout


def test_summary_tables_render(tmp_path):
    _root, idx, disp = make_repo(tmp_path)
    reg = L.build_registry(idx, disp)
    carried = L.load_carried(_root)
    md = L.render_summary(idx, reg, disp, carried=carried)
    assert "## By section" in md and "## Gates (G6-00–G6-08)" in md
    assert "## Scenarios (F01–F32)" in md
    assert "## Carried V5 tail" in md
    assert "F01" in md and "G6-00" in md
    assert "V5-01.01" in md and "V5-01.02" not in md
    led = L.render_ledger(idx, reg, disp, carried=carried)
    assert led["summary"]["total"] == 4
    assert led["scenarios"]["F01"]["anchored_requirements"] == ["V6-01.02"]
    assert led["scenarios"]["F01"]["stage"] == "P1"
