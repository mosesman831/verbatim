"""Deterministic tool-call → OperationClass mapping (SPEC_V3 §22, coding_rules_v1).

``coding_rules_v1`` abstracts host tool calls into the bounded coding-domain
taxonomy (V3-22, OperationClass): ``inspect_file``, ``search_repo``,
``apply_patch``, ``run_check``. Anything the rules cannot name —
arbitrary shell scripts, command substitution, browser actions,
deployments, privileged or irreversible commands, unknown tools — is
``opaque`` evidence. Opaque operations are never abstracted and they block
automatic compilation of the whole episode (§22 producer table), because a
procedure template must preserve only what the compiler actually
understood.

The rules here are a versioned, deterministic table: exact tool-name match
after normalization (case-folded, namespace/tool-prefix stripped), plus one
argument-shape rule for ``run_check`` — structured ``argv`` against an
allowlisted checker (pytest, npm/pnpm test, or named project checkers)
keeps the ``run_check`` class; a raw shell command string does not.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..core.types_v3 import OperationClass

# ---------------------------------------------------------------------------
# tool-name tables (versioned by the compiler manifest)
# ---------------------------------------------------------------------------

#: File-inspection tools: read/cat/open/view a single file's content.
_INSPECT_FILE_TOOLS = frozenset({
    "read", "read_file", "readfile", "cat", "open", "open_file", "view",
    "view_file", "show", "show_file", "file_read", "file_view", "inspect",
    "inspect_file", "head", "tail", "get_file", "load_file", "print_file",
    "file_contents", "get_contents", "sed_print",
})

#: Repository-search tools: locate files or content without modifying.
_SEARCH_REPO_TOOLS = frozenset({
    "grep", "rg", "ripgrep", "find", "fd", "ag", "ack", "search",
    "search_repo", "search_files", "search_code", "find_file",
    "find_files", "find_in_files", "glob", "ls", "list", "list_dir",
    "list_files", "list_directory", "tree", "locate", "whereis",
    "file_search", "code_search", "grep_files", "scan", "scan_repo",
})

#: File-modification tools: write/edit/patch file contents.
_APPLY_PATCH_TOOLS = frozenset({
    "edit", "edit_file", "write", "write_file", "apply_patch", "patch",
    "sed", "replace", "replace_in_file", "insert", "create_file",
    "update_file", "modify", "modify_file", "rename_file", "move_file",
    "delete_file", "remove_file", "append_file", "patch_file", "tee",
    "str_replace_editor", "apply_diff", "fs_write", "fs_edit",
})

#: Checker/test-runner tool names. These still need structured argv in the
#: args to keep the ``run_check`` class — a free-form shell string is opaque.
_RUN_CHECK_TOOLS = frozenset({
    "run_check", "run_checks", "check", "run_test", "run_tests", "test",
    "pytest", "run_pytest", "npm_test", "pnpm_test", "yarn_test",
    "run_build", "build", "lint", "run_lint", "typecheck", "type_check",
    "make", "tox", "nox", "verify", "run_verify", "exec_check",
    "run_command", "execute", "exec_command", "terminal", "shell_exec",
})

#: Checker executables accepted at ``argv[0]`` for a ``run_check`` op
#: (allowlisted pytest / npm-family test runners / named project checkers —
#: §22 "structured argv only for allowlisted pytest, npm/pnpm test, or
#: named project checkers").
CHECKER_ARGV0_ALLOWLIST = frozenset({
    "pytest", "py.test", "python", "python3", "npm", "pnpm", "yarn",
    "make", "tox", "nox", "ruff", "mypy", "flake8", "black", "isort",
    "cargo", "ctest", "go", "check", "lint", "test", "build",
    "npx", "uvx", "uv", "poetry", "hatch", "pre-commit",
})

#: ``python -m <module>`` checker modules accepted in argv form.
_CHECKER_PYMODULES = frozenset({
    "pytest", "mypy", "ruff", "flake8", "black", "isort", "tox", "nox",
    "coverage", "unittest", "build", "twine", "pip", "pre_commit",
})

#: Arg keys that carry a *shell command string* rather than structured
#: argv. Presence of one of these on a run-check tool call demotes the op
#: to opaque: arbitrary command substitution is never a reusable template.
_SHELL_ARG_KEYS = frozenset({
    "command", "cmd", "script", "shell", "sh", "bash", "exec", "code",
    "program", "command_line", "cmdline",
})

_NAME_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _normalize_tool_name(tool_name: Any) -> str:
    """Case-fold and strip host/tool namespaces (``fs.read`` → ``read``,
    ``mcp__filesystem__read_file`` → ``read_file``)."""
    if not isinstance(tool_name, str) or not tool_name.strip():
        return ""
    name = tool_name.strip().lower()
    # Dotted or namespaced ids: keep the most specific final segment, then
    # also try the trailing two segments joined (``fs.read_file``).
    for sep in ("::", "/", "."):
        if sep in name:
            name = name.split(sep)[-1]
    name = _NAME_SPLIT_RE.sub("_", name).strip("_")
    # MCP-style prefixes like ``mcp__filesystem__read_file`` already end in
    # the real tool name after splitting on ``/``; handle ``__`` ids too.
    if "__" in name:
        name = name.split("__")[-1]
    return name


def checker_from_argv(argv: Any) -> Optional[str]:
    """Return the checker name for a structured argv list, or ``None``.

    ``argv`` must be a list/tuple of strings whose first element is an
    allowlisted checker executable (§22): ``pytest``, ``npm test``,
    ``pnpm test``, ``make <target>``, ``python -m pytest`` … Anything else
    — a bare string, an empty list, a non-allowlisted executable — is not a
    checker the compiler can name, so the op stays opaque.
    """
    if not isinstance(argv, (list, tuple)) or not argv:
        return None
    head = argv[0]
    if not isinstance(head, str) or not head:
        return None
    exe = head.rsplit("/", 1)[-1].lower()
    if exe not in CHECKER_ARGV0_ALLOWLIST:
        return None
    if exe in ("python", "python3"):
        # Only ``python -m <allowlisted module>`` is a named checker; an
        # arbitrary ``python script.py`` is an opaque command.
        if len(argv) < 3 or argv[1] != "-m":
            return None
        module = argv[2]
        if not isinstance(module, str) or module.lower() not in _CHECKER_PYMODULES:
            return None
        return module.lower()
    if exe in ("npm", "pnpm", "yarn", "npx"):
        # ``npm test`` / ``pnpm test`` / ``npm run <script>`` are checker
        # invocations; ``npm install`` is a setup action, not a check.
        if len(argv) < 2 or not isinstance(argv[1], str):
            return None
        sub = argv[1].lower()
        if exe == "npx":
            return argv[1].lower()
        if sub in ("test", "t", "run", "run-script", "exec", "lint", "check"):
            return f"{exe} {sub}"
        return None
    if exe in ("uv", "uvx", "poetry", "hatch"):
        if len(argv) < 2 or not isinstance(argv[1], str):
            return None
        sub = argv[1].lower()
        if sub in ("run", "test", "pytest"):
            # ``uv run pytest`` / ``poetry run pytest``: re-dispatch on the
            # wrapped executable when present.
            if sub == "run" and len(argv) >= 3 and isinstance(argv[2], str):
                inner = argv[2].rsplit("/", 1)[-1].lower()
                if inner in CHECKER_ARGV0_ALLOWLIST or inner == "pytest":
                    return inner
                return None
            return f"{exe} {sub}"
        return None
    if exe == "cargo":
        if len(argv) < 2 or not isinstance(argv[1], str):
            return None
        sub = argv[1].lower()
        if sub in ("test", "check", "clippy", "build", "fmt"):
            return f"cargo {sub}"
        return None
    if exe == "go":
        if len(argv) < 2 or argv[1] != "test":
            return None
        return "go test"
    return exe


def _argv_from_args(args: Any) -> Optional[list]:
    """Pull a structured argv out of a tool-call argument mapping."""
    if not isinstance(args, dict):
        return None
    for key in ("argv", "args_list", "command_argv", "exec_argv"):
        v = args.get(key)
        if isinstance(v, (list, tuple)):
            return list(v)
    return None


def classify(tool_name: Any, args: Any = None) -> OperationClass:
    """Map one host tool call to an :class:`OperationClass`.

    Rules (deterministic, versioned by ``COMPILER_MANIFEST``):

    * inspect/search/patch tool names map directly to their class.
    * run-check tool names keep ``run_check`` only when the args carry a
      structured ``argv`` whose executable is an allowlisted checker — a
      ``command``/``cmd``/``script`` shell string demotes to ``opaque``.
    * Unknown names, empty names, and unparseable calls are ``opaque`` —
      never silently dropped or guessed (§22: "unrecognizable → opaque").
    """
    name = _normalize_tool_name(tool_name)
    if not name:
        return OperationClass.OPAQUE
    if name in _INSPECT_FILE_TOOLS:
        return OperationClass.INSPECT_FILE
    if name in _SEARCH_REPO_TOOLS:
        return OperationClass.SEARCH_REPO
    if name in _APPLY_PATCH_TOOLS:
        return OperationClass.APPLY_PATCH
    if name in _RUN_CHECK_TOOLS:
        if isinstance(args, dict) and any(
            k in args and isinstance(args[k], str) for k in _SHELL_ARG_KEYS
        ):
            return OperationClass.OPAQUE
        if checker_from_argv(_argv_from_args(args)) is not None:
            return OperationClass.RUN_CHECK
        return OperationClass.OPAQUE
    return OperationClass.OPAQUE
