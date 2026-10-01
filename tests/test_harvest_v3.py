"""V3 harvester tests (SPEC_V3 §15): per-kind segmentation, references,
language tolerance, and v2 prose parity.

Covers: V3-15.01 (punctuation-light chat parity), V3-15.02 (Unicode/CJK),
V3-15.04 (structure-aware tokenization of diffs/paths/identifiers),
V3-15.05/§15.09 (deterministic reference extraction), V3-15.12
(deterministic versioned output), V3-15.13 (tool-output state facts).
"""

from __future__ import annotations

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v3 import EnvelopeKind
from verbatim.harvest_v3 import (
    HARVESTER_V3_VERSION,
    KIND_HINTS,
    harvest_v3,
)
from verbatim.harvest_v3_refs import REF_EXTRACTOR_VERSION, extract_refs


def texts(cands: list[dict]) -> list[str]:
    return [c["text"] for c in cands]


# ---------------------------------------------------------------------------
# Candidate shape / contract
# ---------------------------------------------------------------------------


def test_candidates_have_required_shape():
    cands = harvest_v3(
        "EditFile(path=\"src/a.py\") → ok",
        envelope_kind=EnvelopeKind.TOOL_CALL,
    )
    assert cands
    for c in cands:
        assert isinstance(c["text"], str) and c["text"]
        assert c["kind_hint"] in KIND_HINTS
        assert isinstance(c["refs"], dict)
        assert "start_byte" in c and "end_byte" in c


def test_kind_hint_vocabulary_is_bounded():
    assert KIND_HINTS == {"command", "path", "error", "test", "prose", "hunk"}


def test_version_constants_exist():
    assert HARVESTER_V3_VERSION
    assert REF_EXTRACTOR_VERSION


def test_deterministic_same_input_same_output():
    text = "EditFile(path=\"src/a.py\") → ok\nBash(\"pytest -x\") → exit 0"
    a = harvest_v3(text, envelope_kind="tool_call")
    b = harvest_v3(text, envelope_kind="tool_call")
    assert a == b  # V3-15.12: replay under same identity → identical output


# ---------------------------------------------------------------------------
# tool_call / tool_result
# ---------------------------------------------------------------------------


def test_tool_call_extracts_path():
    cands = harvest_v3(
        'EditFile(path="src/a.py") → ok',
        envelope_kind=EnvelopeKind.TOOL_CALL,
    )
    assert len(cands) == 1
    c = cands[0]
    assert "src/a.py" in c["text"]
    assert c["kind_hint"] == "command"
    assert "src/a.py" in c["refs"]["paths"]


def test_tool_call_multiple_invocations_split_on_boundaries():
    text = (
        'ReadFile(path="src/a.py") → ok\n'
        'EditFile(path="src/b.py", old="x", new="y") → ok\n'
        'Bash("pytest tests/ -x") → exit 0'
    )
    cands = harvest_v3(text, envelope_kind="tool_call")
    got = texts(cands)
    assert len(got) == 3
    assert all(c["kind_hint"] == "command" for c in cands)
    assert "src/a.py" in got[0] and "src/b.py" in got[1] and "pytest" in got[2]


def test_tool_result_command_error_prose_blocks():
    text = (
        "$ pytest tests/\n"
        "Traceback (most recent call last):\n"
        '  File "src/a.py", line 10, in f\n'
        "    return x\n"
        "ValueError: bad value\n"
        "1 failed, 2 passed in 0.05s"
    )
    cands = harvest_v3(text, envelope_kind="tool_result")
    hints = [c["kind_hint"] for c in cands]
    assert "command" in hints
    assert "error" in hints
    assert "prose" in hints
    err = cands[hints.index("error")]
    assert "ValueError" in err["text"]
    assert "Traceback" in err["text"]
    assert err["refs"]["error_codes"]
    summary = cands[hints.index("prose")]
    assert "1 failed" in summary["text"]


def test_tool_result_path_lines_become_path_candidates():
    text = "$ ls src/\nsrc/a.py\nsrc/b.py\nsrc/sub/c.py"
    cands = harvest_v3(text, envelope_kind="tool_result")
    hints = [c["kind_hint"] for c in cands]
    assert "command" in hints and "path" in hints
    path_cand = cands[hints.index("path")]
    assert "src/a.py" in path_cand["text"]
    assert "src/b.py" in path_cand["refs"]["paths"]
    assert "src/sub/c.py" in path_cand["refs"]["paths"]


def test_tool_result_backslash_continuation_stays_one_command():
    text = "docker run \\\n  --rm \\\n  img:latest"
    cands = harvest_v3(text, envelope_kind="tool_call")
    assert len(cands) == 1
    assert "--rm" in cands[0]["text"]


# ---------------------------------------------------------------------------
# file_diff
# ---------------------------------------------------------------------------

DIFF = (
    "diff --git a/src/a.py b/src/a.py\n"
    "index 111..222 100644\n"
    "--- a/src/a.py\n"
    "+++ b/src/a.py\n"
    "@@ -1,3 +1,4 @@\n"
    " line1\n"
    "-old_line\n"
    "+new_line\n"
    " line2\n"
    "@@ -10,2 +10,2 @@\n"
    " ctx\n"
    "-x\n"
    "+y\n"
    "diff --git a/src/b.py b/src/b.py\n"
    "--- a/src/b.py\n"
    "+++ b/src/b.py\n"
    "@@ -5,1 +5,1 @@\n"
    "-p\n"
    "+q\n"
)


def test_diff_produces_per_hunk_candidates():
    cands = harvest_v3(DIFF, envelope_kind="file_diff")
    hunks = [c for c in cands if c["kind_hint"] == "hunk"]
    assert len(hunks) == 3  # two hunks in a.py + one in b.py


def test_diff_hunks_preserve_file_context():
    cands = harvest_v3(DIFF, envelope_kind=EnvelopeKind.FILE_DIFF)
    a_hunks = [c for c in cands if c.get("file") == "src/a.py"]
    b_hunks = [c for c in cands if c.get("file") == "src/b.py"]
    assert len(a_hunks) == 2 and len(b_hunks) == 1
    for c in a_hunks:
        assert "src/a.py" in c["text"]  # context line preserved
        assert "src/a.py" in c["refs"]["paths"]
    assert "@@ -1,3 +1,4 @@" in a_hunks[0]["text"]
    assert "@@ -10,2 +10,2 @@" in a_hunks[1]["text"]
    assert "-old_line" in a_hunks[0]["text"] and "+new_line" in a_hunks[0]["text"]
    assert "src/b.py" in b_hunks[0]["text"]


def test_diff_plain_unified_without_git_headers():
    text = (
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    cands = harvest_v3(text, envelope_kind="file_diff")
    assert len(cands) == 1
    assert cands[0]["kind_hint"] == "hunk"
    assert cands[0].get("file") == "x.py"
    assert "-old" in cands[0]["text"]


def test_diff_plain_multifile_hunks_attribute_correct_file():
    text = (
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a1\n"
        "+b1\n"
        "--- a/y.py\n"
        "+++ b/y.py\n"
        "@@ -2,1 +2,1 @@\n"
        "-a2\n"
        "+b2\n"
    )
    cands = harvest_v3(text, envelope_kind="file_diff")
    hunks = [c for c in cands if c["kind_hint"] == "hunk"]
    assert len(hunks) == 2
    assert hunks[0].get("file") == "x.py"
    assert hunks[1].get("file") == "y.py"
    assert "x.py" in hunks[0]["text"] and "y.py" in hunks[1]["text"]


def test_diff_header_only_section_kept():
    text = (
        "diff --git a/bin.dat b/bin.dat\n"
        "index abc..def 100644\n"
        "Binary files a/bin.dat and b/bin.dat differ\n"
    )
    cands = harvest_v3(text, envelope_kind="file_diff")
    assert len(cands) >= 1
    assert any("Binary" in c["text"] for c in cands)


def test_diff_hunk_body_is_verbatim_substring():
    cands = harvest_v3(DIFF, envelope_kind="file_diff")
    payload = DIFF.encode("utf-8")
    for c in cands:
        if c["kind_hint"] != "hunk":
            continue
        body = payload[c["start_byte"] : c["end_byte"]].decode("utf-8")
        assert body in DIFF
        assert body in c["text"]


# ---------------------------------------------------------------------------
# test_result / verification
# ---------------------------------------------------------------------------


def test_pytest_failed_line_names_test():
    cands = harvest_v3(
        "FAILED tests/test_x.py::test_y - assert 1 == 2",
        envelope_kind="test_result",
    )
    assert len(cands) == 1
    c = cands[0]
    assert "tests/test_x.py::test_y" in c["text"]
    assert c["kind_hint"] == "test"
    assert "tests/test_x.py" in c["refs"]["paths"]
    assert "tests/test_x.py::test_y" in c["refs"]["tests"]


PYTEST_OUT = (
    "============================= test session starts ==============================\n"
    "collected 2 items\n"
    "\n"
    "tests/test_x.py .F                                                       [100%]\n"
    "\n"
    "=================================== FAILURES ===================================\n"
    "__________________________________ test_y __________________________________\n"
    "    def test_y():\n"
    ">       assert 1 == 2\n"
    "E       assert 1 == 2\n"
    "\n"
    "tests/test_x.py:10: AssertionError\n"
    "=========================== short test summary info ============================\n"
    "FAILED tests/test_x.py::test_y - assert 1 == 2\n"
    "========================= 1 failed, 1 passed in 0.05s ==========================\n"
)


def test_pytest_full_output_segments():
    cands = harvest_v3(PYTEST_OUT, envelope_kind=EnvelopeKind.TEST_RESULT)
    got = texts(cands)
    assert any("tests/test_x.py::test_y" in t for t in got)
    err = [c for c in cands if c["kind_hint"] == "error"]
    assert err, "assertion-failure block must be an error candidate"
    assert any("assert 1 == 2" in c["text"] for c in err)
    assert any("test_y" in c["text"] for c in err)


def test_tap_style_lines():
    text = "1..2\nok 1 - first test\nnot ok 2 - second test"
    cands = harvest_v3(text, envelope_kind="test_result")
    got = texts(cands)
    assert any("not ok 2" in t for t in got)


def test_go_fail_line():
    text = "--- FAIL: TestFoo (0.00s)\nFAIL\nexit status 1"
    cands = harvest_v3(text, envelope_kind="test_result")
    assert any("TestFoo" in c["text"] for c in cands)


# ---------------------------------------------------------------------------
# error / recovery
# ---------------------------------------------------------------------------


def test_error_traceback_is_single_error_candidate():
    text = (
        "Traceback (most recent call last):\n"
        '  File "app/main.py", line 42, in run\n'
        "    do_thing()\n"
        '  File "app/lib.py", line 7, in do_thing\n'
        "    raise RuntimeError('boom')\n"
        "RuntimeError: boom"
    )
    cands = harvest_v3(text, envelope_kind="error")
    assert len(cands) == 1
    c = cands[0]
    assert c["kind_hint"] == "error"
    assert "RuntimeError: boom" in c["text"]
    assert 'File "app/main.py"' in c["text"]
    assert "app/main.py" in c["refs"]["paths"]
    assert "RuntimeError" in c["refs"]["error_codes"]


def test_error_chained_tracebacks_split():
    text = (
        "Traceback (most recent call last):\n"
        '  File "a.py", line 1, in f\n'
        "ValueError: first\n"
        "\n"
        "During handling of the above exception, another exception occurred:\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "b.py", line 2, in g\n'
        "KeyError: 'k'"
    )
    cands = harvest_v3(text, envelope_kind="error")
    assert len(cands) >= 1
    joined = "\n".join(texts(cands))
    assert "ValueError" in joined and "KeyError" in joined


def test_recovery_keeps_error_and_prose():
    text = "Error: connection refused\nretrying with backoff and it worked"
    cands = harvest_v3(text, envelope_kind="recovery")
    hints = [c["kind_hint"] for c in cands]
    assert "error" in hints and "prose" in hints


# ---------------------------------------------------------------------------
# Prose path: v2 parity (V3-15.01 / V3-15.02)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "My name is Bob",
        "my wife is Sarah",
        "use pnpm not npm",
        "moved to Berlin last week",
        "prefer dark mode",
    ],
)
def test_unpunctuated_chat_still_yields_candidates(text):
    cands = harvest_v3(text, envelope_kind="user_message")
    assert texts(cands) == [text]


def test_cjk_text_yields_candidates():
    text = "私は毎日Neovimを使っています。仕事ではVS Codeを使います。"
    cands = harvest_v3(text, envelope_kind="user_message", min_len=8)
    got = texts(cands)
    assert len(got) == 2
    assert got[0].endswith("。")


def test_cjk_without_punctuation_not_skipped():
    text = "毎日ターミナルで作業しています"
    cands = harvest_v3(text, envelope_kind="user_message", min_len=8)
    assert cands


def test_prose_path_matches_v2_sentence_split():
    text = "Dr. Smith uses Neovim daily and writes lots of Rust. He lives in Oslo."
    cands = harvest_v3(text, envelope_kind="user_message", min_len=8)
    assert texts(cands) == [
        "Dr. Smith uses Neovim daily and writes lots of Rust.",
        "He lives in Oslo.",
    ]


def test_prose_path_matches_v2_merged_short_tail():
    # v2 parity: a terminated trailing sentence below min_len merges back
    # into the preceding bounded statement (SPEC §11).
    text = "Dr. Smith uses Neovim daily and writes lots of Rust. He lives in Oslo."
    cands = harvest_v3(text, envelope_kind="user_message")
    assert len(cands) == 1
    assert cands[0]["text"] == text


@pytest.mark.parametrize(
    "kind",
    [
        "user_message",
        "assistant_message",
        "document",
        "import",
        "plan",
        "subgoal",
        "decision",
        "agent_note",
        "lesson",
        "connector_item",
    ],
)
def test_prose_kinds_delegate_to_v2(kind):
    cands = harvest_v3("I use Neovim for all of my editing.", envelope_kind=kind)
    assert len(cands) == 1
    assert cands[0]["kind_hint"] == "prose"
    assert cands[0]["text"] == "I use Neovim for all of my editing."


def test_plan_list_items():
    text = "1. refactor the parser next week\n2. add tests for the harvester"
    cands = harvest_v3(text, envelope_kind="plan")
    assert len(cands) == 2


def test_unknown_kind_falls_back_to_prose():
    cands = harvest_v3("I use Neovim for everything.", envelope_kind="bogus_kind")
    assert len(cands) == 1
    assert cands[0]["kind_hint"] == "prose"


def test_none_kind_uses_prose():
    cands = harvest_v3("I use Neovim for everything.")
    assert len(cands) == 1


def test_chat_noise_still_dropped():
    assert harvest_v3("ok hi", envelope_kind="user_message") == []
    assert harvest_v3("thanks", envelope_kind="assistant_message") == []


# ---------------------------------------------------------------------------
# Pathological input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "\n\n\n", None, 123, b"", b"   ", object()],
)
def test_pathological_input_returns_empty(bad):
    assert harvest_v3(bad, envelope_kind="tool_call") == []
    assert harvest_v3(bad, envelope_kind="file_diff") == []
    assert harvest_v3(bad, envelope_kind="user_message") == []


def test_invalid_utf8_bytes_raise_typed_validation():
    # F4-19 / V4-13.11/13.12: a decode failure is not "no candidates" —
    # the public harvesting surface reports it as a typed VALIDATION
    # failure instead of silently reading malformed bytes as clean-empty.
    for kind in ("tool_call", "file_diff", "user_message", None):
        with pytest.raises(VerbatimError) as ei:
            harvest_v3(b"\xff\xfe invalid \x80", envelope_kind=kind)
        assert ei.value.code is ErrorCode.VALIDATION


def test_invalid_bounds_return_empty():
    assert harvest_v3("some text here", max_candidates=0) == []
    assert harvest_v3("some text here", max_len=2, min_len=10) == []


def test_separator_only_input_returns_empty():
    assert harvest_v3("====\n----\n****", envelope_kind="tool_result") == []


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_max_candidates_respected():
    text = "\n".join(f'EditFile(path="src/f{i}.py") → ok' for i in range(40))
    cands = harvest_v3(text, envelope_kind="tool_call", max_candidates=5)
    assert len(cands) == 5


def test_max_candidates_respected_diff():
    hunks = "".join(
        f"@@ -{i},1 +{i},1 @@\n-x\n+y\n" for i in range(20)
    )
    text = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n" + hunks
    cands = harvest_v3(text, envelope_kind="file_diff", max_candidates=7)
    assert len(cands) == 7


def test_long_error_block_respects_max_len():
    frames = "".join(
        f'  File "app/m{i}.py", line {i}, in f{i}\n    work()\n' for i in range(60)
    )
    text = f"Traceback (most recent call last):\n{frames}RuntimeError: boom"
    cands = harvest_v3(text, envelope_kind="error", max_len=200)
    assert cands
    assert all(len(c["text"]) <= 200 for c in cands)


def test_single_long_line_is_bounded_not_dropped():
    text = "x" * 5000
    cands = harvest_v3(text, envelope_kind="tool_result", max_len=500)
    assert cands
    assert all(len(c["text"]) <= 500 for c in cands)


# ---------------------------------------------------------------------------
# refs extraction
# ---------------------------------------------------------------------------


def test_refs_paths_identifiers_urls():
    refs = extract_refs(
        "EditFile writes src/a.py using snake_case_fn; see https://ex.com/x"
    )
    assert "src/a.py" in refs["paths"]
    assert "snake_case_fn" in refs["identifiers"]
    assert "EditFile" in refs["identifiers"]
    assert "https://ex.com/x" in refs["urls"]


def test_refs_commands():
    refs = extract_refs("$ pytest tests/ -x\n`git status`\nEditFile(path=\"a/b.py\")")
    assert any("pytest" in c for c in refs["commands"])
    assert any("git status" in c for c in refs["commands"])
    assert any("EditFile(" in c for c in refs["commands"])


def test_refs_error_codes():
    refs = extract_refs(
        "ValueError: bad\nerror TS2345 in build\nexit code 1\nHTTP 404"
    )
    assert "ValueError" in refs["error_codes"]
    assert "TS2345" in refs["error_codes"]
    assert any("exit code 1" == e.lower() or "exit code 1" in e.lower() for e in refs["error_codes"])
    assert any("404" in e for e in refs["error_codes"])


def test_refs_versions():
    refs = extract_refs("upgraded to v1.2.3 and python 3.11.15")
    assert "v1.2.3" in refs["versions"]
    assert "3.11.15" in refs["versions"]


def test_refs_empty_input():
    refs = extract_refs("")
    assert all(v == [] for v in refs.values())
    refs = extract_refs(None)  # type: ignore[arg-type]
    assert all(v == [] for v in refs.values())


def test_refs_no_identifiers_inside_paths():
    refs = extract_refs("src/a.py")
    assert "a" not in refs["identifiers"]
    assert "py" not in refs["identifiers"]


def test_refs_cjk_no_crash():
    refs = extract_refs("私は毎日Neovimを使っています")
    assert isinstance(refs, dict)


def test_refs_are_bounded_on_adversarial_input():
    text = "\n".join(f"path{i}/file{i}.py" for i in range(200))
    refs = extract_refs(text)
    assert len(refs["paths"]) <= 64


# ---------------------------------------------------------------------------
# v2-write-path compatibility
# ---------------------------------------------------------------------------


def test_candidates_feed_text_dict_shape():
    """The v2 context_groups write path consumes dicts carrying 'text' —
    every candidate must supply it."""
    for kind, text in [
        ("tool_call", 'EditFile(path="a/b.py") → ok'),
        ("file_diff", DIFF),
        ("test_result", "FAILED t/x.py::test_y - assert 1 == 2"),
        ("error", "Error: broke"),
        ("user_message", "I use Neovim for all of my editing."),
    ]:
        cands = harvest_v3(text, envelope_kind=kind)
        assert cands, kind
        assert all(isinstance(c.get("text"), str) and c["text"] for c in cands)


def test_byte_offsets_reproduce_body_verbatim():
    text = (
        "EditFile(path=\"src/a.py\") → ok\n"
        "some prose output here for padding\n"
        "src/a.py\n"
    )
    payload = text.encode("utf-8")
    cands = harvest_v3(text, envelope_kind="tool_result")
    for c in cands:
        body = payload[c["start_byte"] : c["end_byte"]].decode("utf-8")
        assert body in text
        # non-composed candidates reproduce text exactly
        if c["kind_hint"] != "hunk":
            assert body == c["text"]
