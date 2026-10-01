"""Binding extraction for ``coding_rules_v1`` (SPEC_V3 §21.02, §22).

Instance-specific values inside a tool call's arguments become
:class:`Binding` slots — values the host must re-acquire at reuse time
(V3-21.02). Only *schema-declared* parameter fields are substituted into
slots (§22 binding-extraction row): a root-relative file reference becomes
a ``repo_path`` slot, check targets become ``test_target`` slots, the
recorded source revision becomes a ``revision`` slot. Unknown field roles
stay literal evidence inside the episode's envelopes — the compiler never
string-matches arbitrary argument values into parameters, and never
treats secrets or patch bodies as template content (V3-22.14).

``param_template`` therefore contains only binding placeholders
(``{"$binding": <name>}``); the observed instance value lives on the
Binding record itself, not inside the executable template. This is what
keeps ``operations_json`` free of raw shell strings.
"""

from __future__ import annotations

from typing import Any, Optional

from ..core.types import json_dumps
from ..core.types_v3 import Binding, OperationClass
from .operations import _argv_from_args, _normalize_tool_name

# ---------------------------------------------------------------------------
# declared parameter fields per operation class → binding kind
# ---------------------------------------------------------------------------

_REPO_PATH_FIELDS = frozenset({
    "path", "file", "file_path", "filename", "filepath", "target",
    "root", "dir", "directory", "base", "cwd", "folder", "repo",
})

#: SEARCH_REPO pattern-ish fields — task-intrinsic slots, kind ``other``.
_SEARCH_OTHER_FIELDS = frozenset({
    "pattern", "query", "regex", "text", "glob", "symbol", "name",
    "include", "filter",
})

#: INSPECT_FILE selector fields beyond the path itself.
_INSPECT_OTHER_FIELDS = frozenset({"symbol", "function", "section", "query"})

_TEST_TARGET_FIELDS = frozenset({
    "target", "targets", "tests", "test", "files", "paths", "mark",
    "select", "suite",
})

#: APPLY_PATCH arg keys that are *instance evidence*, never template slots
#: (V3-22.14: a patch body is not a reusable patch). Listed so the extractor
#: documents the fence in one place; anything undeclared is dropped anyway.
_PATCH_BODY_KEYS = frozenset({
    "patch", "diff", "content", "body", "edits", "hunks", "hunk",
    "new_string", "old_string", "replacement", "text", "changes",
})

_DECLARED: dict[OperationClass, dict[str, str]] = {
    OperationClass.INSPECT_FILE: {
        **{f: "repo_path" for f in _REPO_PATH_FIELDS},
        **{f: "other" for f in _INSPECT_OTHER_FIELDS},
    },
    OperationClass.SEARCH_REPO: {
        **{f: "repo_path" for f in _REPO_PATH_FIELDS},
        **{f: "other" for f in _SEARCH_OTHER_FIELDS},
    },
    OperationClass.APPLY_PATCH: {
        **{f: "repo_path" for f in _REPO_PATH_FIELDS},
    },
    OperationClass.RUN_CHECK: {
        **{f: "test_target" for f in _TEST_TARGET_FIELDS},
    },
}

#: Arg values that look like test/check targets inside a checker argv tail:
#: path-ish tokens (contain ``/`` or a known test extension) and node ids.
_TARGET_EXTS = (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java",
                ".rb", ".cpp", ".cc", ".c", ".h", ".cs")


def _looks_like_target(token: str) -> bool:
    t = token.strip()
    if not t or t.startswith("-"):
        return False
    base = t.split("::", 1)[0]
    return "/" in base or base.endswith(_TARGET_EXTS)


_CHECKER_LIKE = frozenset({
    "pytest", "mypy", "ruff", "flake8", "black", "isort", "tox", "nox",
    "unittest", "coverage", "npm", "pnpm", "yarn", "cargo", "go",
})

_SUBCOMMAND_TOKENS = frozenset({"test", "run", "exec", "check", "lint", "t"})


def argv_targets(argv: Any) -> list[str]:
    """Test/check targets inside a structured checker argv.

    Drops ``argv[0]`` (checker identity), option flags, ``-m <module>``
    pairs, wrapper subcommands (``npm test``, ``uv run``), and the wrapped
    checker name itself — what remains is the check's target set.
    """
    if not isinstance(argv, (list, tuple)):
        return []
    items = [a for a in argv if isinstance(a, str)]
    tail = items[1:] if items else []
    targets: list[str] = []
    seen: set[str] = set()
    skip_next = False
    for i, tok in enumerate(tail):
        if skip_next:
            skip_next = False
            continue
        if tok == "-m":
            skip_next = True
            continue
        if tok.startswith("-"):
            continue
        if i == 0 and tok.lower() in _SUBCOMMAND_TOKENS:
            continue
        if i <= 1 and tok.lower() in _CHECKER_LIKE:
            continue
        if _looks_like_target(tok) and tok not in seen:
            seen.add(tok)
            targets.append(tok)
    return targets


def extract_bindings(
    op_class: OperationClass,
    args: Any,
    *,
    ord: int = 0,
) -> tuple[dict[str, Any], list[Binding]]:
    """Return ``(param_template, bindings)`` for one classified operation.

    ``param_template`` maps each declared field to a ``{"$binding": name}``
    placeholder; every placeholder resolves to a returned Binding carrying
    the observed instance value. Undeclared fields (patch bodies, opaque
    payloads, unrecognized roles) are omitted from the template entirely —
    they remain evidence on the episode's envelopes.
    """
    declared = _DECLARED.get(op_class, {})
    template: dict[str, Any] = {}
    bindings: list[Binding] = []
    if not isinstance(args, dict) or not declared:
        return template, bindings

    used: set[str] = set()

    def _name(field: str) -> str:
        base = f"op{ord}_{field}"
        name, i = base, 1
        while name in used:
            i += 1
            name = f"{base}_{i}"
        used.add(name)
        return name

    for field in sorted(args):
        value = args[field]
        if field == "argv" and op_class == OperationClass.RUN_CHECK:
            # Structured checker argv binds only its target tail; the
            # executable itself is checker identity (verification/signature),
            # never a template value.
            targets = argv_targets(value)
            if not targets:
                continue
            name = _name(field)
            template[field] = {"$binding": name}
            bindings.append(Binding(
                name=name, kind="test_target",
                observed_value=json_dumps(targets), required=True,
            ))
            continue
        kind = declared.get(field)
        if kind is None:
            continue
        if isinstance(value, (list, tuple)):
            scalars = [v for v in value if isinstance(v, (str, int, float))
                       and not isinstance(v, bool)]
            if not scalars:
                continue
            name = _name(field)
            template[field] = {"$binding": name}
            bindings.append(Binding(
                name=name, kind=kind,
                observed_value=json_dumps([str(v) for v in scalars]),
                required=True,
            ))
            continue
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            continue
        name = _name(field)
        template[field] = {"$binding": name}
        bindings.append(Binding(
            name=name, kind=kind, observed_value=str(value), required=True,
        ))
    return template, bindings


def revision_binding(observed_value: Optional[str]) -> Optional[Binding]:
    """The recorded source revision becomes a ``revision`` slot (§22).

    ``required=False``: the slot records what was observed; absence is
    unknown, not a hard precondition (V3-21.07).
    """
    if not observed_value:
        return None
    return Binding(
        name="source_revision", kind="revision",
        observed_value=str(observed_value), required=False,
    )


def tool_name_of(tool_name: Any) -> str:
    """Normalized tool name for OperationTemplate.tool."""
    return _normalize_tool_name(tool_name) or "unknown"
