"""Durable tests for the v7 requirements ledger + manifest + gate
skeleton (SPEC_V7 V7-00.03/00.06, V7-22.10, V7-26.02).

Covers the generator's contract: every `V7-NN.MM` definition is
registered (349 at spec R1), the H01–H104 / G7-00–G7-14 / SB-01–SB-20 /
B0–B6 / D7-01–D7-40 registries are parsed, duplicates and dangling
references fail, `ledger_v7.json` parses and stays byte-fresh under
``--check``, every requirement seeds ``planned``/``not_run``, all 15
gates evaluate ``not_run`` against an empty artifact index (V7-26.02),
and the run manifest records git sha + dirty flag, seeds,
python/sqlite versions, CPU count, dependency versions, artifact
digests, and the ``provisional/v7-r0`` formula tag under a canonical
digest that changes when any artifact digest changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

from eval.v7 import gates as G
from eval.v7 import ledger as L
from eval.v7 import manifest as M

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def run_ledger(*args, root=REPO_ROOT):
    return subprocess.run(
        [sys.executable, "-m", "eval.v7.ledger", *args,
         "--root", root],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def run_gates(*args, root=REPO_ROOT):
    return subprocess.run(
        [sys.executable, "-m", "eval.v7.gates", *args,
         "--root", root],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


SPEC_MINI = """# mini v7 spec

## 01. Kernel

- V7-01.01: Reads are exact.
- V7-01.02: Writes are durable; see V7-01.01 and V7-02.01/02 and
  V7-02.03–04; exercised by H01.

## 02. Delivery

- V7-02.01: Permits bind caller.
- V7-02.02: Permits expire.
- V7-02.03: Permits renew.
- V7-02.04: Permits audit.

| ID | Observable result |
| --- | --- |
| H01 | read path end to end |
| H02 | reopen preserves state |

| Gate | Permits | Requires |
| --- | --- | --- |
| G7-00 Authority | any change | H01 plus audit |
| G7-01 Parity | claims | SB-01/02 pass |

| ID | Benchmark | Metric | Target | Ref |
| --- | --- | --- | --- | --- |
| SB-01 | bench | any@10 | ≥ 0.9 | x 0.5 |
| SB-02 | bench | all@10 | ≥ 0.5 | x 0.4 |

| Env | Workload | Targets |
| --- | --- | --- |
| B0 | tiny | p95 ≤ 20 |
| B1 | small | p95 ≤ 40 |

| ID | Defect | Location | Consequence | Closed by |
| --- | --- | --- | --- | --- |
| D7-01 | broken fold | x.py:1 | noise | V7-01.01 |

| Defect | Requirement(s) | Scenario(s) |
| --- | --- | --- |
| D7-01 fold bug | V7-01.01 | H02 |

| Phase | Content | Exit |
| --- | --- | --- |
| P0 | contracts freeze | ledger parses |
| P1 | delivery work | H01–H02 plus G7-00 |
"""


def make_repo(tmp_path, spec=SPEC_MINI):
    """Create a miniature repo root; returns (root, idx, disp)."""
    (tmp_path / "SPEC_V7.md").write_text(spec)
    root = str(tmp_path)
    idx = L.load_specs(root)
    return root, idx, L.Dispositions()


# ---------------------------------------------------------------------------
# real-spec coverage
# ---------------------------------------------------------------------------


def test_real_spec_all_requirements_parsed():
    idx = L.load_specs(REPO_ROOT)
    # independent recount: definition lines in the file, unique ids
    raw = open(os.path.join(REPO_ROOT, "SPEC_V7.md")).read()
    def_lines = re.findall(r"^- V7-(\d{2}\.\d{2}):", raw, re.M)
    assert len(idx.requirements) == len(def_lines) == 349
    assert len(set(def_lines)) == len(def_lines)
    assert idx.duplicates == []


def test_real_spec_named_collections():
    idx = L.load_specs(REPO_ROOT)
    assert sorted(idx.scenarios, key=lambda s: int(s[1:])) == [
        f"H{n:02d}" for n in range(1, 105)]
    assert sorted(idx.gates) == [f"G7-{n:02d}" for n in range(15)]
    assert sorted(idx.scoreboard) == [f"SB-{n:02d}" for n in range(1, 21)]
    assert sorted(idx.stages) == [
        "P0", "P1", "P10", "P2", "P2b", "P3", "P4", "P5", "P6",
        "P7", "P8", "P9",
    ]
    for base in "B0 B1 B2 B3 B4 B5 B6".split():
        assert base in idx.envelopes
    for v in ("B1-cache", "B1-neural", "B1-CE"):
        assert v in idx.envelopes
    assert sorted(idx.defects) == [f"D7-{n:02d}" for n in range(1, 25)]
    assert sorted(idx.traces) == [f"D7-{n:02d}" for n in range(1, 25)]


def test_real_spec_honest_start_state():
    """V7-00.06: every requirement derives ``planned``/``not_run`` —
    no executed evidence exists yet."""
    idx = L.load_specs(REPO_ROOT)
    reg = L.build_registry(idx, L.Dispositions())
    assert len(reg) == 349
    for rid, r in reg.items():
        assert r["status"] == "planned", rid
        assert r["implementation_status"] == "planned", rid
        assert r["qualification_status"] == "not_run", rid
        assert r["owner"] == "unset", rid
        assert r["executed_evidence"] == [], rid
        assert r["stage"] in L.STAGES + ("",), rid


def test_real_spec_validate_clean():
    idx = L.load_specs(REPO_ROOT)
    issues = L.validate(idx, L.Dispositions(), root=REPO_ROOT)
    assert issues == [], issues[:8]


def test_real_repo_check_passes():
    proc = run_ledger("--check")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "349 requirements" in proc.stdout
    assert "104 scenarios" in proc.stdout
    assert "15 gates" in proc.stdout
    assert "20 scoreboard rows" in proc.stdout


def test_ledger_json_on_disk_parses_and_matches_spec():
    path = os.path.join(REPO_ROOT, L.DEFAULT_JSON_PATH)
    doc = json.loads(open(path).read())
    idx = L.load_specs(REPO_ROOT)
    assert set(doc["requirements"]) == set(idx.requirements)
    assert set(doc["scenarios"]) == set(idx.scenarios)
    assert set(doc["gates"]) == set(idx.gates)
    assert set(doc["scoreboard"]) == set(idx.scoreboard)
    row = doc["requirements"]["V7-22.10"]
    for key in ("qualification_status", "implementation_status",
                "status", "owner", "code_surfaces", "scenarios",
                "evidence_type", "executed_evidence", "gate",
                "scoreboard", "envelopes", "blocker", "note", "text"):
        assert key in row, key
    # the spec sha is pinned
    assert doc["specs"]["SPEC_V7.md"]["sha256"] == hashlib.sha256(
        open(os.path.join(REPO_ROOT, "SPEC_V7.md"), "rb").read()
    ).hexdigest()


def test_id_section_semantics_for_playbook_definitions():
    """The five §25-located definitions keep their id namespace:
    V7-12.15 is a packs requirement written inside §25.5."""
    idx = L.load_specs(REPO_ROOT)
    d = idx.requirements["V7-12.15"]
    assert d.section == 25 and d.id_section == 12
    reg = L.build_registry(idx, L.Dispositions())
    r = reg["V7-12.15"]
    assert r["section"] == 12          # registry rows carry the int
    assert r["spec_section"] == 25
    assert r["capability"] == "packs"
    assert r["stage"] == "P4"
    rendered = L.render_ledger(idx, reg, L.Dispositions())
    assert rendered["requirements"]["V7-12.15"]["section"] == "V7-12"


def test_anchor_shorthand_and_ranges():
    assert L.anchors_in("V7-22.08/09") == ["V7-22.08", "V7-22.09"]
    assert L.anchors_in("V7-01.05/06.03") == ["V7-01.05", "V7-06.03"]
    assert L.anchors_in("V7-05.01–09") == [
        f"V7-05.{n:02d}" for n in range(1, 10)]
    assert L.hids_in("H01, H10–H12, H104") == [
        "H01", "H10", "H11", "H12", "H104"]
    assert L._h_mentions("H01–H96") == []
    assert L.gates_in("G7-01/03") == ["G7-01", "G7-03"]
    assert L.gates_in("G7-10 … G7-13") == [
        "G7-10", "G7-11", "G7-12", "G7-13"]
    assert L._gate_mentions("G7-00–G7-14") == []
    assert L.scoreboard_in("SB-06/07/08/10/11") == [
        "SB-06", "SB-07", "SB-08", "SB-10", "SB-11"]
    assert L.envelopes_in("B0–B2 plus B1-CE") == [
        "B0", "B1", "B1-CE", "B2"]


# ---------------------------------------------------------------------------
# rejection contract
# ---------------------------------------------------------------------------


def test_duplicate_definition_rejected(tmp_path):
    dup = SPEC_MINI + "\n## 09. Later\n\n- V7-01.01: again.\n"
    _root, idx, disp = make_repo(tmp_path, spec=dup)
    issues = L.validate(idx, disp)
    assert any("duplicate spec definition" in i and "V7-01.01" in i
               for i in issues)


def test_undefined_reference_rejected(tmp_path):
    bad = SPEC_MINI + (
        "\n## 09. Later\n\n"
        "- V7-09.01: see V7-99.99, H99-free? no: H105, G7-15, SB-21, "
        "B7, D7-41.\n")
    _root, idx, disp = make_repo(tmp_path, spec=bad)
    issues = L.validate(idx, disp)
    assert any("undefined requirement V7-99.99" in i for i in issues)
    assert any("H105" in i and "registry" in i for i in issues)
    assert any("G7-15" in i and "registry" in i for i in issues)
    assert any("SB-21" in i and "registry" in i for i in issues)
    assert any("B7" in i and "registry" in i for i in issues)
    assert any("D7-41" in i and "registry" in i for i in issues)


def test_reserved_defect_range_is_legal(tmp_path):
    """D7-25..D7-40 are allocated by the audit wave — referencing one
    before its register row exists is legal (V7-00.03)."""
    ok = SPEC_MINI + "\n## 09. Later\n\n- V7-09.01: reserves D7-30.\n"
    _root, idx, disp = make_repo(tmp_path, spec=ok)
    issues = L.validate(idx, disp)
    assert not any("D7-30" in i for i in issues)


def test_disposition_rules(tmp_path):
    _root, idx, _d = make_repo(tmp_path)
    disp = L.load_dispositions_from_text(json.dumps({
        "spec_sha256": idx.spec_sha256,
        "requirements": {
            "V7-01.01": {
                "implementation_status": "implemented",
                "qualification_status": "qualified",
            },
            "V7-01.02": {
                "qualification_status": "deferred",
            },
            "V7-77.01": {"owner": "x"},
        },
    }))
    issues = L.validate(idx, disp)
    assert any("named executed evidence" in i for i in issues)
    assert any("deferred" in i and "owner" in i for i in issues)
    assert any("unknown requirement V7-77.01" in i for i in issues)


def test_check_fails_when_ledger_stale(tmp_path):
    root, _idx, _d = make_repo(tmp_path)
    proc = run_ledger("--check", root=root)
    assert proc.returncode == 1
    assert "missing" in proc.stdout
    proc = run_ledger("--write", root=root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    proc = run_ledger("--check", root=root)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# gates skeleton (V7-26.02)
# ---------------------------------------------------------------------------


def test_gates_all_not_run_on_clean_tree():
    results = G.evaluate(REPO_ROOT)
    assert sorted(results) == [f"G7-{n:02d}" for n in range(15)]
    for gid, g in results.items():
        assert g["status"] == "not_run", gid
        assert g["inputs_ready"] == 0
        assert g["name"], gid          # names come from the ledger
        assert g["requires"], gid
        assert g["inputs_total"] > 0


def test_gate_aggregation_rules():
    ledger = L.render_ledger(
        L.load_specs(REPO_ROOT),
        L.build_registry(L.load_specs(REPO_ROOT), L.Dispositions()),
        L.Dispositions())
    # a single not_run input forces not_run even beside passed inputs
    arts = {
        "sb.SB-17": {"status": "passed", "manifest_digest": "x"},
        "sb.SB-18": {"status": "passed", "manifest_digest": "x"},
        # scale.add_ack_alpha absent -> not_run
    }
    assert G.evaluate_gates(ledger, arts)["G7-06"]["status"] == "not_run"
    arts["scale.add_ack_alpha"] = {"status": "passed"}
    assert G.evaluate_gates(ledger, arts)["G7-06"]["status"] == "passed"
    arts["sb.SB-17"]["status"] = "failed"
    assert G.evaluate_gates(ledger, arts)["G7-06"]["status"] == "missed"
    arts["sb.SB-17"]["status"] = "invalid"
    assert G.evaluate_gates(ledger, arts)["G7-06"]["status"] == "invalid"
    arts["sb.SB-17"]["status"] = "blocked_on_authorization"
    assert (G.evaluate_gates(ledger, arts)["G7-06"]["status"]
            == "blocked_on_authorization")
    # invalid/missed still outrank blocked
    arts["scale.add_ack_alpha"]["status"] = "not_run"
    assert (G.evaluate_gates(ledger, arts)["G7-06"]["status"]
            == "blocked_on_authorization")


def test_gates_cli_prints_table():
    proc = run_gates()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "G7-00" in proc.stdout and "G7-14" in proc.stdout
    assert "not_run=15" in proc.stdout


# ---------------------------------------------------------------------------
# run manifest (V7-22.10)
# ---------------------------------------------------------------------------


def manifest_kwargs(tmp_path, **over):
    art = tmp_path / "artifact-a.json"
    art.write_text('{"a": 1}\n')
    kw = dict(
        root=str(tmp_path),
        run_id="test-run",
        seeds=[42],
        profile="local_memory",
        tiers=["S1"],
        lane_policy_digest="0" * 64,
        reader={"id": "verbatim_reader/v1", "decoding": "temp0",
                "prompt_digest": "1" * 64},
        judge={"id": "verbatim_judge_locomo/v1",
               "prompt_digest": "2" * 64},
        datasets={"owned_lme_like": {"digest": "3" * 64,
                                     "split": "test"}},
        artifacts=["artifact-a.json"],
        formula_tags={"bm25f": "bm25f/v1"},
        started_at="2026-09-22T00:00:00Z",
        ended_at="2026-09-22T00:00:01Z",
    )
    kw.update(over)
    return kw


def test_manifest_records_required_fields(tmp_path):
    m = M.build_manifest(**manifest_kwargs(tmp_path))
    # V7-22.10 field set
    assert m["vcs"]["system"] == "git"
    assert "sha" in m["vcs"] and "dirty" in m["vcs"]
    assert m["seeds"] == [42]
    assert m["environment"]["python_version"]
    assert m["environment"]["sqlite_version"]
    assert m["environment"]["cpu_count"] == os.cpu_count()
    assert "numpy" in m["environment"]["dependencies"]
    assert m["artifacts"][0]["sha256"]
    assert m["artifacts"][0]["missing"] is False
    assert m["reader"]["prompt_digest"] == "1" * 64
    assert m["judge"]["prompt_digest"] == "2" * 64
    # V7-26.03: provisional formula tag until O9
    assert m["formula"]["status"] == "provisional/v7-r0"
    assert m["formula"]["label"] == "formula=unselected"
    assert m["formula"]["selected"] is False
    assert m["formula"]["tags"] == {"bm25f": "bm25f/v1"}
    assert m["stage_profile_schema"] == "stage_profile/v7"
    assert re.fullmatch(r"[0-9a-f]{64}", m["digest"])
    assert M.verify_manifest(m)


def test_manifest_digest_deterministic(tmp_path):
    a = M.build_manifest(**manifest_kwargs(tmp_path))
    b = M.build_manifest(**manifest_kwargs(tmp_path))
    assert a["digest"] == b["digest"]
    assert M._json_dumps(a) == M._json_dumps(b)


def test_manifest_digest_tracks_artifact_digests(tmp_path):
    kw = manifest_kwargs(tmp_path)
    a = M.build_manifest(**kw)
    b = M.build_manifest(**manifest_kwargs(
        tmp_path, artifact_digests={"other.bin": "4" * 64}))
    assert a["digest"] != b["digest"]
    # and a changed file content changes the pin, hence the digest —
    # reuse the same kwargs so only the artifact bytes move
    (tmp_path / "artifact-a.json").write_text('{"a": 2}\n')
    c = M.build_manifest(**kw)
    assert c["digest"] != a["digest"]
    assert (c["artifacts"][0]["sha256"]
            != a["artifacts"][0]["sha256"])


def test_manifest_git_state_real_repo():
    st = M.git_state(REPO_ROOT)
    assert st["system"] == "git"
    if st["status"] == "ok":
        assert re.fullmatch(r"[0-9a-f]{40}", st["sha"])
        assert isinstance(st["dirty"], bool)
    else:
        assert st["sha"] is None and st["dirty"] is None


def test_manifest_git_state_non_repo(tmp_path):
    st = M.git_state(str(tmp_path))
    assert st["status"] == "unavailable"
    assert st["sha"] is None and st["dirty"] is None


def test_manifest_missing_artifact_is_honest(tmp_path):
    m = M.build_manifest(**manifest_kwargs(
        tmp_path, artifacts=["artifact-a.json", "ghost.bin"]))
    ghost = [a for a in m["artifacts"] if a["path"] == "ghost.bin"][0]
    assert ghost["missing"] is True and ghost["sha256"] is None


def test_manifest_write_round_trip(tmp_path):
    m = M.build_manifest(**manifest_kwargs(tmp_path))
    rel = M.write_manifest(m, str(tmp_path))
    assert rel == os.path.join(
        "eval", "v7", "manifests", f"{m['digest']}.json")
    loaded = M.load_manifest(os.path.join(tmp_path, rel))
    assert loaded["digest"] == m["digest"]
    assert M.verify_manifest(loaded)
