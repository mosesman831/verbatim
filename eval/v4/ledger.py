"""Executable-traceability ledger for SPEC_V4 / SPEC_V4_5 (§61, V4-02.*).

The ledger answers one question honestly: which `V4-NN.MM` / `V45-NN.MM`
requirement has *named executable checks* behind it, and which does not.
It is generated, never handwritten (V4-61.01/61.02):

* ``load_specs`` parses every ``- V4-NN.MM:`` definition line plus every
  in-text reference. Duplicate definitions and references to ids that
  were never defined are hard errors (V4-61.03, V4-67.08, V45-19.02).
* ``discover_evidence`` scans ``tests/`` for anchors. A requirement id
  inside a test function's own docstring or body is a *named check*
  (strong). A ``test_cNN_*``/``test_dNN_*`` name binds the test to the
  §58/D acceptance scenario and through it to the anchored requirements
  (strong). Anchors in section-banner comments or class docstrings are
  attributed to the tests they head, but as *section markers* they are
  weak (V4-61.02). Anchors in a module docstring are recorded as file
  *claims* only — a docstring is not an executable check.
* ``requirement_map.yaml`` is the human-curated overlay: per id a
  ``status`` (V4-02.02), ``tests`` (pytest node ids), ``scenarios``,
  ``evidence`` labels (V4-02.07 DOC/CODE/LOCAL/BENCH/INDEPENDENT) and a
  ``note``. The generator *adds* discovered tests on ``--write`` but
  never rewrites a status a human set.
* ``validate`` enforces the §61 contract: no unknown or duplicate ids,
  no mapped test that pytest cannot collect, no stale spec hash, no
  ``verified`` without a named check, no ``unimplemented`` row carrying
  tests, and no discovered check left unmapped.

Honesty rule (V4-02.02/02.03): nothing here infers ``verified`` from
counts; a seeded entry is ``verified`` only when named checks cite the
requirement directly or implement its spec-anchored scenario.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass, field
from typing import Iterator, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a dev dependency
    yaml = None

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Requirement statuses, V4-02.02. Order is the registry sort order.
STATUSES = (
    "verified",
    "partial",
    "implemented_unverified",
    "deferred",
    "not_applicable",
    "unimplemented",
)

#: Evidence labels, V4-02.07.
EVIDENCE_LABELS = ("DOC", "CODE", "LOCAL", "BENCH", "INDEPENDENT")

#: Specs covered by the ledger, in registry order.
SPEC_FILES = ("SPEC_V4.md", "SPEC_V4_5.md")

#: Default locations (relative to repo root).
DEFAULT_MAP_PATH = os.path.join("eval", "v4", "requirement_map.yaml")
DEFAULT_JSON_PATH = os.path.join("eval", "v4", "ledger.json")
DEFAULT_MD_PATH = "REQUIREMENTS_V4.md"
TESTS_DIR = "tests"
PROD_DIR = "verbatim"

#: `- V4-NN.MM: text` (also V45-). Definition lines only.
_DEF_RE = re.compile(r"^-\s+(V45|V4)-(\d{2})\.(\d{2}):\s*(.*)$")
#: `## NN. Title`
_SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$")
#: Any anchor mention, incl. `/MM` and `/NN.MM` shorthand continuations
#: (`V4-38.01/02`, `V4-10.05/11.04`, `V4-09.06/07/08`).
_ANCHOR_RE = re.compile(
    r"\b(V45|V4)-(\d{2})\.(\d{2})((?:/\d{2}(?:\.\d{2})?)*)"
)
_ANCHOR_PART_RE = re.compile(r"/(\d{2})(?:\.(\d{2}))?")
#: Acceptance-scenario mentions (`C18`, `D07`) — letter boundary required.
_SCENARIO_RE = re.compile(r"\b([CD])(\d{2})\b")
#: `test_c55_...` / `test_d07...` function names.
_TESTNAME_SCENARIO_RE = re.compile(r"^test_([cCdD])(\d{2})(?:_|$)")
#: Table row whose first cell is a scenario id (`| C01 | ... |`).
_SCENARIO_ROW_RE = re.compile(r"^\|\s*(C\d{2}|D\d{2})\s*\|(.*)\|\s*$")


def _expand_anchor(m: re.Match) -> list[str]:
    """Expand one regex match into every requirement id it denotes.

    `V4-10.05/11.04` -> [`V4-10.05`, `V4-11.04`]; `V4-38.01/02` ->
    [`V4-38.01`, `V4-38.02`]. A bare `/MM` keeps the base section.
    """
    prefix, sec, sub, tail = m.groups()
    ids = [f"{prefix}-{sec}.{sub}"]
    for seg, sub2 in _ANCHOR_PART_RE.findall(tail or ""):
        if sub2:
            ids.append(f"{prefix}-{seg}.{sub2}")
        else:
            ids.append(f"{prefix}-{sec}.{seg}")
    return ids


def anchors_in(text: str) -> list[str]:
    """All requirement ids mentioned in ``text`` (shorthand expanded)."""
    out: list[str] = []
    for m in _ANCHOR_RE.finditer(text):
        out.extend(_expand_anchor(m))
    return out


def scenarios_in(text: str) -> list[str]:
    """All acceptance-scenario ids (`C07`, `D12`) mentioned in ``text``."""
    return [f"{letter}{num}" for letter, num in _SCENARIO_RE.findall(text)]


class LedgerError(Exception):
    """Raised for unparseable inputs; validation failures are reported
    as issue lists instead (see :func:`validate`)."""


# ---------------------------------------------------------------------------
# spec parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequirementDef:
    """One `- V4-NN.MM:` definition line."""

    req_id: str
    spec: str          # spec file name, e.g. "SPEC_V4.md"
    section: int       # numeric section, e.g. 38
    section_title: str
    text: str
    line: int


@dataclass(frozen=True)
class ScenarioDef:
    """One §58 C-row or SPEC_V4_5 §15 D-row."""

    sc_id: str         # "C07" / "D12"
    spec: str
    text: str
    anchors: tuple     # requirement ids the spec binds to the scenario
    tier: str          # C / E / R / M5...
    line: int


@dataclass
class SpecIndex:
    """Merged parse of SPEC_V4.md + SPEC_V4_5.md."""

    requirements: dict = field(default_factory=dict)   # id -> RequirementDef
    scenarios: dict = field(default_factory=dict)      # id -> ScenarioDef
    section_titles: dict = field(default_factory=dict)  # "V4-38" -> title
    references: dict = field(default_factory=dict)     # id -> [spec:line]
    duplicates: list = field(default_factory=list)     # [(id, spec, line)]
    spec_sha256: dict = field(default_factory=dict)    # file -> hex

    def scenario_anchors(self, sc_id: str) -> tuple:
        sc = self.scenarios.get(sc_id)
        return sc.anchors if sc else ()


def _parse_spec_file(root: str, fname: str, idx: SpecIndex) -> None:
    path = os.path.join(root, fname)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        raise LedgerError(f"cannot read {path}: {e}")
    idx.spec_sha256[fname] = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    section = 0
    section_title = ""
    prefix = "V45" if fname.endswith("_5.md") else "V4"
    for n, line in enumerate(raw.splitlines(), 1):
        m = _SECTION_RE.match(line)
        if m:
            section = int(m.group(1))
            section_title = m.group(2).strip()
            idx.section_titles[f"{prefix}-{section:02d}"] = section_title
            continue
        m = _DEF_RE.match(line)
        if m:
            rid = f"{m.group(1)}-{m.group(2)}.{m.group(3)}"
            if rid in idx.requirements:
                prev = idx.requirements[rid]
                idx.duplicates.append(
                    f"{rid} defined twice: {prev.spec}:{prev.line} "
                    f"and {fname}:{n}"
                )
                continue
            idx.requirements[rid] = RequirementDef(
                req_id=rid,
                spec=fname,
                section=section,
                section_title=section_title,
                text=m.group(4).strip(),
                line=n,
            )
            # the requirement text itself may reference other ids
            for ref in anchors_in(m.group(4)):
                if ref != rid:
                    idx.references.setdefault(ref, []).append(f"{fname}:{n}")
            continue
        sm = _SCENARIO_ROW_RE.match(line)
        if sm:
            sc_id, rest = sm.group(1), sm.group(2)
            cells = [c.strip() for c in rest.split("|")]
            # | desc | anchors | tier | — anchors live in cell 1 by
            # contract; scan every cell so a reordered table still parses.
            anchors: list[str] = []
            desc = cells[0] if cells else ""
            tier = cells[-1] if len(cells) > 1 else ""
            for cell in cells[1:]:
                for am in _ANCHOR_RE.finditer(cell):
                    anchors.extend(_expand_anchor(am))
            if sc_id in idx.scenarios:
                prev = idx.scenarios[sc_id]
                idx.duplicates.append(
                    f"scenario {sc_id} defined twice: "
                    f"{prev.spec}:{prev.line} and {fname}:{n}"
                )
            else:
                idx.scenarios[sc_id] = ScenarioDef(
                    sc_id=sc_id,
                    spec=fname,
                    text=desc,
                    anchors=tuple(anchors),
                    tier=tier,
                    line=n,
                )
            # the row's anchors are also *references*
            for rid in anchors:
                idx.references.setdefault(rid, []).append(f"{fname}:{n}")
            continue
        for rid in anchors_in(line):
            idx.references.setdefault(rid, []).append(f"{fname}:{n}")
        for sc in scenarios_in(line):
            idx.references.setdefault(sc, []).append(f"{fname}:{n}")


def load_specs(root: str, spec_files=SPEC_FILES) -> SpecIndex:
    """Parse every spec file into one :class:`SpecIndex`."""
    idx = SpecIndex()
    for fname in spec_files:
        _parse_spec_file(root, fname, idx)
    return idx


# ---------------------------------------------------------------------------
# evidence discovery — scan tests/ for anchors and scenario tests
# ---------------------------------------------------------------------------

#: Attribution kinds for a (requirement, test) pair, strongest first.
#: ``verified`` may only be seeded from STRONG_KINDS; the rest registers
#: coverage but keeps the requirement ``implemented_unverified``
#: (V4-61.02: section markers and file docstrings alone are not checks).
K_DIRECT = "direct"          # req id inside the test's body/docstring
K_SCENARIO = "scenario"      # test named test_cNN_*/test_dNN_* (spec-anchored)
K_BANNER = "banner"          # section-banner comment or docstring scenario cite
K_CLASS_DOC = "class_doc"    # class docstring anchor
STRONG_KINDS = frozenset({K_DIRECT, K_SCENARIO})
_KIND_RANK = {K_DIRECT: 3, K_SCENARIO: 2, K_BANNER: 1, K_CLASS_DOC: 1}
CLAIM = "claim"              # module docstring anchor — file claim, not a test


@dataclass
class Discovery:
    """Evidence found by scanning the test tree (and production cites)."""

    # req_id -> {test node id -> attribution kind}  (claims land in
    # `claims` — a module docstring is not an executable check)
    tests: dict = field(default_factory=dict)
    # req_id -> {scenario id} covered by at least one discovered test
    scenarios: dict = field(default_factory=dict)
    # scenario id -> {test node id}
    scenario_tests: dict = field(default_factory=dict)
    # req_id -> {test file} whose module docstring cites it
    claims: dict = field(default_factory=dict)
    # req_id -> {verbatim/ file} citing it (implementation exists)
    code_refs: dict = field(default_factory=dict)
    # scenario id -> {test file} docstring/name claims (informational)
    scenario_claims: dict = field(default_factory=dict)

    def add_test(self, req_id: str, node_id: str, kind: str) -> None:
        cur = self.tests.setdefault(req_id, {})
        # keep the strongest attribution when several apply
        if _KIND_RANK.get(kind, 0) >= _KIND_RANK.get(cur.get(node_id, K_BANNER), 0):
            cur[node_id] = kind

    def add_scenario(self, req_id: str, sc_id: str) -> None:
        self.scenarios.setdefault(req_id, set()).add(sc_id)


@dataclass
class _FuncSpan:
    node_id: str          # pytest node id tail, e.g. "test_x" or "TestA::test_y"
    start: int
    end: int
    is_test: bool
    node: object = None   # ast FunctionDef (for docstring extraction)


@dataclass
class _ClassSpan:
    node_id: str          # nested class node id, e.g. "TestOuter::TestInner"
    start: int
    end: int
    testable: bool        # pytest-collectable (Test* name, no __init__)
    node: object = None


def _collect_spans(tree: ast.AST):
    """Walk a test module's AST; return (funcs, classes, module_doc)."""
    funcs: list[_FuncSpan] = []
    classes: list[_ClassSpan] = []

    def visit_class(node: ast.ClassDef, prefix: str) -> None:
        has_init = any(
            isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
            and b.name == "__init__"
            for b in node.body
        )
        testable = node.name.startswith("Test") and not has_init
        node_id = f"{prefix}{node.name}"
        cls = _ClassSpan(
            node_id=node_id,
            start=node.lineno,
            end=node.end_lineno or node.lineno,
            testable=testable,
            node=node,
        )
        classes.append(cls)
        for b in node.body:
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append(
                    _FuncSpan(
                        f"{node_id}::{b.name}",
                        b.lineno,
                        b.end_lineno or b.lineno,
                        testable and b.name.startswith("test_"),
                        b,
                    )
                )
            elif isinstance(b, ast.ClassDef):
                visit_class(b, f"{node_id}::")

    for item in tree.body:  # type: ignore[attr-defined]
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(
                _FuncSpan(
                    item.name,
                    item.lineno,
                    item.end_lineno or item.lineno,
                    item.name.startswith("test_"),
                    item,
                )
            )
        elif isinstance(item, ast.ClassDef):
            visit_class(item, "")
    return funcs, classes, _docstring_span(tree)


def _docstring_span(node) -> tuple:
    """(start, end) lines of a module/class docstring, or ()."""
    if not getattr(node, "body", None):
        return ()
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return (first.lineno, first.end_lineno or first.lineno)
    return ()


def _enclosing(spans, line: int):
    """Innermost span containing ``line`` (spans have .start/.end)."""
    best = None
    for s in spans:
        if s.start <= line <= s.end:
            if best is None or s.start >= best.start:
                best = s
    return best


def _scan_test_file(path: str, rel: str, idx: SpecIndex, disc: Discovery) -> None:
    """Attribute every anchor in one test file to pytest node ids.

    * anchor inside a ``test_*`` body or docstring -> that node, DIRECT
      (a named check citing the requirement);
    * ``test_cNN_*`` / ``test_dNN_*`` name -> the spec-anchored
      requirements of scenario CNN/DNN, SCENARIO (the named scenario is
      the requirement's acceptance check);
    * ``CNN``/``DNN`` inside a test body -> anchored reqs, BANNER
      (incidental citation — the test may cover only part);
    * anchors in a class docstring -> every testable method, CLASS_DOC;
    * anchors in a standalone comment run -> every following test until
      the next standalone comment run at that scope, BANNER (V4-61.02:
      a section marker alone is not a check — it registers coverage but
      cannot seed ``verified``);
    * anchors in the module docstring -> file claims only (CLAIM).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return

    funcs, classes, module_doc = _collect_spans(tree)
    test_funcs = [f for f in funcs if f.is_test]
    all_defs = list(funcs) + [
        _FuncSpan("", n.lineno, n.end_lineno or n.lineno, False, n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    class_of_method = {f.node_id: f.node_id.rsplit("::", 1)[0]
                       for f in funcs if "::" in f.node_id}

    def apply_pending(pend, fspan):
        if not pend:
            return
        node_id = f"{rel}::{fspan.node_id}"
        for rid in pend["reqs"]:
            disc.add_test(rid, node_id, K_BANNER)
        for sc in pend["scenarios"]:
            disc.scenario_tests.setdefault(sc, set()).add(node_id)
            for rid in idx.scenario_anchors(sc):
                disc.add_test(rid, node_id, K_BANNER)
                disc.add_scenario(rid, sc)

    # --- docstrings -------------------------------------------------
    if module_doc:
        dstart, dend = module_doc
        text = "\n".join(src.splitlines()[dstart - 1:dend])
        for rid in anchors_in(text):
            disc.claims.setdefault(rid, set()).add(rel)
        for sc in scenarios_in(text):
            disc.scenario_claims.setdefault(sc, set()).add(rel)

    for c in classes:
        doc = ast.get_docstring(c.node) if c.node is not None else None
        if not doc or not c.testable:
            continue
        rids = anchors_in(doc)
        scs = scenarios_in(doc)
        if not rids and not scs:
            continue
        for f in test_funcs:
            if f.node_id.startswith(c.node_id + "::"):
                node_id = f"{rel}::{f.node_id}"
                for rid in rids:
                    disc.add_test(rid, node_id, K_CLASS_DOC)
                for sc in scs:
                    disc.scenario_tests.setdefault(sc, set()).add(node_id)
                    for rid in idx.scenario_anchors(sc):
                        disc.add_test(rid, node_id, K_CLASS_DOC)
                        disc.add_scenario(rid, sc)

    for f in test_funcs:
        doc = ast.get_docstring(f.node) if f.node is not None else None
        node_id = f"{rel}::{f.node_id}"
        if doc:
            for rid in anchors_in(doc):
                disc.add_test(rid, node_id, K_DIRECT)
            for sc in scenarios_in(doc):
                disc.scenario_tests.setdefault(sc, set()).add(node_id)
                for rid in idx.scenario_anchors(sc):
                    disc.add_test(rid, node_id, K_BANNER)
                    disc.add_scenario(rid, sc)
        nm = _TESTNAME_SCENARIO_RE.match(f.node_id.rsplit("::", 1)[-1])
        if nm:
            sc = f"{nm.group(1).upper()}{nm.group(2)}"
            if sc in idx.scenarios:
                disc.scenario_tests.setdefault(sc, set()).add(node_id)
                for rid in idx.scenario_anchors(sc):
                    disc.add_test(rid, node_id, K_SCENARIO)
                    disc.add_scenario(rid, sc)

    # --- comments + pending banners ---------------------------------
    # pending[scope] holds the anchors of the most recent standalone
    # comment run at scope ("<module>" or a class node id); those anchors
    # apply to every test def that follows until the next standalone run
    # at the same scope — including an anchor-free run, which is a real
    # section break. *Inline* comments (code before the #) never touch
    # pending: they annotate one expression, not a section. Inside a
    # test body any comment — inline or standalone — is a named check.
    pending: dict = {}
    prev_standalone = -10
    try:
        tokens = tokenize.generate_tokens(io.StringIO(src).readline)
        comments = [
            (tok.start[0], tok.start[1], tok.string)
            for tok in tokens
            if tok.type == tokenize.COMMENT
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        comments = []

    lines = src.splitlines()
    comment_lines = {}
    for row, col, text in comments:
        standalone = not lines[row - 1][:col].strip()
        comment_lines.setdefault(row, []).append((text, standalone))

    class_starts = {c.start: c for c in classes}
    func_starts = {f.start: f for f in test_funcs}
    for i in range(1, len(lines) + 1):
        if i in comment_lines:
            ftest = _enclosing(test_funcs, i)
            text = "\n".join(t for t, _ in comment_lines[i])
            if ftest is not None:
                # comment inside a test body: named check (direct);
                # scenario mentions stay weak for anchored reqs.
                node_id = f"{rel}::{ftest.node_id}"
                for rid in anchors_in(text):
                    disc.add_test(rid, node_id, K_DIRECT)
                for sc in scenarios_in(text):
                    disc.scenario_tests.setdefault(sc, set()).add(node_id)
                    for rid in idx.scenario_anchors(sc):
                        disc.add_test(rid, node_id, K_BANNER)
                        disc.add_scenario(rid, sc)
                continue
            if _enclosing(all_defs, i) is not None:
                continue  # inside a helper/fixture body — not a check
            if not all(s for _, s in comment_lines[i]):
                continue  # inline remark — not a section marker
            cls = _enclosing(classes, i)
            scope = cls.node_id if cls is not None else "<module>"
            if i != prev_standalone + 1:
                pending[scope] = {"reqs": set(), "scenarios": set()}
            pend = pending.setdefault(scope, {"reqs": set(), "scenarios": set()})
            pend["reqs"].update(anchors_in(text))
            pend["scenarios"].update(scenarios_in(text))
            prev_standalone = i
            continue
        # non-comment line: a class def inherits the module-scope banner
        # for its methods; a test def consumes its own scope's banner.
        c = class_starts.get(i)
        if c is not None:
            mod = pending.get("<module>")
            pending[c.node_id] = {
                "reqs": set(mod["reqs"]) if mod else set(),
                "scenarios": set(mod["scenarios"]) if mod else set(),
            }
        f = func_starts.get(i)
        if f is not None:
            scope = class_of_method.get(f.node_id, "<module>")
            apply_pending(pending.get(scope), f)




def _iter_test_files(root: str, tests_dir: str) -> Iterator[str]:
    base = os.path.join(root, tests_dir)
    for dirpath, _dirs, files in os.walk(base):
        for fn in sorted(files):
            if fn.startswith("test_") and fn.endswith(".py"):
                yield os.path.join(dirpath, fn)
            elif fn.endswith("_test.py"):
                yield os.path.join(dirpath, fn)


def _scan_production_refs(root: str, idx: SpecIndex, disc: Discovery) -> None:
    """Record which requirements production modules cite (code exists)."""
    base = os.path.join(root, PROD_DIR)
    if not os.path.isdir(base):
        return
    for dirpath, _dirs, files in os.walk(base):
        if "__pycache__" in dirpath:
            continue
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue
            for rid in set(anchors_in(text)):
                disc.code_refs.setdefault(rid, set()).add(rel)


def discover_evidence(root: str, idx: SpecIndex, tests_dir: str = TESTS_DIR) -> Discovery:
    """Scan the test tree + production cites into a :class:`Discovery`."""
    disc = Discovery()
    for path in _iter_test_files(root, tests_dir):
        rel = os.path.relpath(path, root)
        _scan_test_file(path, rel, idx, disc)
    _scan_production_refs(root, idx, disc)
    return disc


def collect_pytest_nodes(root: str, tests_dir: str = TESTS_DIR) -> set:
    """Base node ids pytest actually collects (params stripped).

    Runs ``python -m pytest <tests_dir> --collect-only``; each emitted
    line looks like ``tests/x.py::TestA::test_y[param]``.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", tests_dir, "--collect-only"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    nodes = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith(("=", "<")):
            nodes.add(line.split("[", 1)[0])
    if not nodes and proc.returncode not in (0, 5):
        raise LedgerError(
            f"pytest collection failed (rc={proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return nodes


# ---------------------------------------------------------------------------
# requirement map (eval/v4/requirement_map.yaml)
# ---------------------------------------------------------------------------


@dataclass
class MapEntry:
    """One curated row of requirement_map.yaml."""

    status: str = ""
    tests: list = field(default_factory=list)
    scenarios: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "evidence": list(self.evidence),
            "scenarios": list(self.scenarios),
            "tests": list(self.tests),
            "note": self.note,
        }


@dataclass
class RequirementMap:
    spec_sha256: dict = field(default_factory=dict)
    entries: dict = field(default_factory=dict)   # req_id -> MapEntry
    source_path: str = DEFAULT_MAP_PATH


if yaml is not None:
    class _DupCheckLoader(yaml.SafeLoader):
        """SafeLoader that rejects duplicate mapping keys (V4-61.03)."""

    def _no_dup_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if key in mapping:
                raise LedgerError(
                    f"duplicate key {key!r} in requirement map "
                    f"(line {key_node.start_mark.line + 1})"
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _DupCheckLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup_mapping
    )
else:
    _DupCheckLoader = None


def _require_yaml() -> None:
    if yaml is None:
        raise LedgerError(
            "pyyaml is required to read eval/v4/requirement_map.yaml "
            "(dev extra: pip install -e '.[dev]')"
        )


def load_map_from_text(text: str, source: str = "<map>") -> RequirementMap:
    """Parse requirement-map YAML from a string."""
    _require_yaml()
    try:
        raw = yaml.load(text, Loader=_DupCheckLoader)
    except LedgerError:
        raise
    except yaml.YAMLError as e:
        raise LedgerError(f"cannot parse {source}: {e}")
    return _map_from_raw(raw, source)


def load_map(path: str) -> RequirementMap:
    """Load requirement_map.yaml; structural errors raise LedgerError."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return load_map_from_text(f.read(), path)
    except LedgerError:
        raise
    except OSError as e:
        raise LedgerError(f"cannot read {path}: {e}")


def _map_from_raw(raw, source: str) -> RequirementMap:
    raw = raw or {}
    rm = RequirementMap(source_path=source)
    rm.spec_sha256 = dict(raw.get("spec_sha256") or {})
    reqs = raw.get("requirements") or {}
    if not isinstance(reqs, dict):
        raise LedgerError(f"{source}: 'requirements' must be a mapping")
    for rid, body in reqs.items():
        body = body or {}
        if not isinstance(body, dict):
            raise LedgerError(f"{source}: entry {rid} must be a mapping")
        rm.entries[rid] = MapEntry(
            status=str(body.get("status") or ""),
            tests=[str(t) for t in (body.get("tests") or [])],
            scenarios=[str(s) for s in (body.get("scenarios") or [])],
            evidence=[str(e) for e in (body.get("evidence") or [])],
            note=str(body.get("note") or ""),
        )
    return rm


def _yaml_scalar(s: str) -> str:
    """Quote a scalar only when plain form would misparse."""
    if s == "" or s != s.strip() or any(
        c in s for c in ":#{}[]&,*?|-<>=!%@`\"' \t\n"
    ) or s.lower() in ("yes", "no", "true", "false", "null", "none", "on", "off"):
        return json.dumps(s, ensure_ascii=False)
    return s


def render_map_yaml(rm: RequirementMap) -> str:
    """Serialize the map with a stable, human-diffable field order."""
    out = io.StringIO()
    out.write(
        "# V4/V45 requirement -> executable-evidence map (SPEC_V4 §61).\n"
        "#\n"
        "# `tests`/`scenarios` are regenerated from test-file anchors by\n"
        "# `python tools/gen_v4_ledger.py --write`; `status`, `evidence`,\n"
        "# and `note` are the reviewed judgments — the generator never\n"
        "# rewrites a status once a row exists. Statuses per V4-02.02:\n"
        "#   verified | partial | implemented_unverified | deferred |\n"
        "#   not_applicable | unimplemented\n"
        "# `verified` requires at least one mapped test node that pytest\n"
        "# collects; evidence labels per V4-02.07: DOC CODE LOCAL BENCH\n"
        "# INDEPENDENT.\n"
    )
    out.write("schema: 1\n")
    out.write("spec_sha256:\n")
    for name in sorted(rm.spec_sha256):
        out.write(f"  {name}: {rm.spec_sha256[name]}\n")
    out.write("requirements:\n")
    for rid in sorted(rm.entries):
        e = rm.entries[rid]
        out.write(f"  {rid}:\n")
        out.write(f"    status: {e.status}\n")
        if e.evidence:
            out.write("    evidence:\n")
            for ev in e.evidence:
                out.write(f"      - {ev}\n")
        if e.scenarios:
            out.write("    scenarios:\n")
            for sc in e.scenarios:
                out.write(f"      - {sc}\n")
        if e.tests:
            out.write("    tests:\n")
            for t in e.tests:
                out.write(f"      - {_yaml_scalar(t)}\n")
        if e.note:
            out.write(f"    note: {_yaml_scalar(e.note)}\n")
    return out.getvalue()


# ---------------------------------------------------------------------------
# registry build — merge spec definitions, map overlay, discovery
# ---------------------------------------------------------------------------

#: Statuses a human may set that ``--write`` must preserve verbatim.
_REVIEWED_STATUSES = ("verified", "partial", "deferred", "not_applicable")


def seed_status(req_id: str, disc: Discovery) -> str:
    """Auto status for a newly-seeded map row.

    `verified` only when a *named* check exists (direct in-test anchor
    or a spec-anchored scenario-named test); banner/class-doc coverage
    registers but seeds ``implemented_unverified`` (V4-61.02).
    """
    kinds = (disc.tests.get(req_id) or {}).values()
    if any(k in STRONG_KINDS for k in kinds):
        return "verified"
    return "implemented_unverified"


def default_status(req_id: str, disc: Discovery) -> tuple:
    """(status, note) for a requirement with no map row."""
    code = sorted(disc.code_refs.get(req_id) or ())
    claims = sorted(disc.claims.get(req_id) or ())
    if code:
        return (
            "implemented_unverified",
            "implementation cites this id in "
            + ", ".join(code[:3])
            + ("..." if len(code) > 3 else "")
            + "; no mapped executable check",
        )
    if claims:
        return (
            "implemented_unverified",
            "test module docstring claims coverage in "
            + ", ".join(claims[:3])
            + ("..." if len(claims) > 3 else "")
            + "; no named check (V4-61.02)",
        )
    return ("unimplemented", "no implementation or test evidence registered")


def build_registry(idx: SpecIndex, rm: RequirementMap, disc: Discovery) -> dict:
    """Merge spec defs + map overlay + discovery into per-id records."""
    registry = {}
    for rid in sorted(idx.requirements):
        d = idx.requirements[rid]
        entry = rm.entries.get(rid)
        if entry is not None:
            status = entry.status
            note = entry.note
            evidence = list(entry.evidence)
            tests = list(entry.tests)
            scenarios = list(entry.scenarios)
        else:
            status, note = default_status(rid, disc)
            evidence, tests, scenarios = [], [], []
        if tests and not evidence:
            evidence = ["LOCAL"]  # a mapped pytest node is a LOCAL check
        registry[rid] = {
            "id": rid,
            "spec": d.spec,
            "section": d.section,
            "section_title": d.section_title,
            "text": d.text,
            "line": d.line,
            "status": status,
            "evidence": evidence,
            "tests": sorted(set(tests)),
            "scenarios": sorted(set(scenarios)),
            "spec_scenarios": sorted(
                s for s, sc in idx.scenarios.items() if rid in sc.anchors
            ),
            "claims": sorted(disc.claims.get(rid) or ()),
            "code_refs": sorted(disc.code_refs.get(rid) or ()),
            "note": note,
        }
    return registry


def merge_discovery(rm: RequirementMap, idx: SpecIndex, disc: Discovery) -> RequirementMap:
    """Return a new map with discovered evidence folded in.

    * requirements with discovered checks get a row (status seeded by
      :func:`seed_status`) — new rows only; existing ``status``,
      ``evidence`` and ``note`` are preserved verbatim;
    * ``tests``/``scenarios`` become the union of curated and discovered
      sets (curated rows may list checks that cite no anchor);
    * ``spec_sha256`` is refreshed to the files on disk.

    The merge is deterministic — ``--check`` compares its rendering
    against the file on disk to detect a stale map.
    """
    out = RequirementMap(
        spec_sha256=dict(idx.spec_sha256),
        entries={rid: MapEntry(
            status=e.status,
            tests=list(e.tests),
            scenarios=list(e.scenarios),
            evidence=list(e.evidence),
            note=e.note,
        ) for rid, e in rm.entries.items()},
        source_path=rm.source_path,
    )
    for rid in sorted(disc.tests):
        if rid not in idx.requirements:
            continue  # dangling cite — validate() reports it
        entry = out.entries.get(rid)
        if entry is None:
            entry = out.entries[rid] = MapEntry(status=seed_status(rid, disc))
            if not entry.evidence:
                entry.evidence = ["LOCAL"]
        entry.tests = sorted(set(entry.tests) | set(disc.tests[rid]))
        entry.scenarios = sorted(
            set(entry.scenarios) | set(disc.scenarios.get(rid) or ())
        )
    return out


# ---------------------------------------------------------------------------
# validation (V4-61.03)
# ---------------------------------------------------------------------------


def validate(
    idx: SpecIndex,
    rm: RequirementMap,
    disc: Discovery,
    collected: Optional[set],
) -> list:
    """Return the list of ledger violations; empty means the map is valid.

    ``collected`` is the set of pytest base node ids (see
    :func:`collect_pytest_nodes`); pass ``None`` to skip the collection
    check (e.g. unit tests of the parser).
    """
    issues: list[str] = []

    for dup in idx.duplicates:
        issues.append(f"duplicate spec definition: {dup}")

    defined = set(idx.requirements)
    for rid, refs in sorted(idx.references.items()):
        if rid[0] in "CD" and rid[1:].isdigit():
            continue  # scenario refs checked separately
        if rid not in defined:
            issues.append(
                f"spec reference to undefined requirement {rid} "
                f"({refs[0]}, +{len(refs) - 1} more)"
            )
    for sc_id, refs in sorted(idx.references.items()):
        if sc_id[0] in "CD" and sc_id[1:].isdigit() and sc_id not in idx.scenarios:
            issues.append(
                f"spec reference to undefined scenario {sc_id} ({refs[0]})"
            )

    # anchors in the test/production trees must also resolve — a test
    # citing V4-99.99 is a dangling reference, not evidence
    for rid in sorted(disc.tests):
        if rid not in defined:
            t = sorted(disc.tests[rid])[0]
            issues.append(f"undefined requirement {rid} cited by {t}")
    for rid in sorted(disc.claims):
        if rid not in defined:
            issues.append(
                f"undefined requirement {rid} cited in docstring of "
                f"{sorted(disc.claims[rid])[0]}"
            )
    for rid in sorted(disc.code_refs):
        if rid not in defined:
            issues.append(
                f"undefined requirement {rid} cited in "
                f"{sorted(disc.code_refs[rid])[0]}"
            )
    cited_scenarios = (
        set(disc.scenario_tests)
        | set(disc.scenario_claims)
        | {s for scs in disc.scenarios.values() for s in scs}
    )
    for sc in sorted(cited_scenarios - set(idx.scenarios)):
        where = sorted(disc.scenario_tests.get(sc)
                       or disc.scenario_claims.get(sc) or ("?",))[0]
        issues.append(f"undefined scenario {sc} cited by {where}")

    # pinned spec hashes must match the files on disk
    for fname, pinned in sorted(rm.spec_sha256.items()):
        actual = idx.spec_sha256.get(fname)
        if actual is None:
            issues.append(f"map pins hash for unknown spec {fname}")
        elif pinned != actual:
            issues.append(
                f"stale spec hash for {fname}: map has "
                f"{pinned[:12]}..., file is {actual[:12]}... — "
                f"run `tools/gen_v4_ledger.py --write` after reviewing "
                f"the spec diff"
            )
    for fname in idx.spec_sha256:
        if fname not in rm.spec_sha256:
            issues.append(f"map does not pin a spec hash for {fname}")

    for rid in sorted(set(rm.entries) - defined):
        issues.append(f"map row for unknown requirement {rid}")

    for rid, entry in sorted(rm.entries.items()):
        where = f"map row {rid}"
        if entry.status not in STATUSES:
            issues.append(f"{where}: invalid status {entry.status!r}")
        for ev in entry.evidence:
            if ev not in EVIDENCE_LABELS:
                issues.append(f"{where}: invalid evidence label {ev!r}")
        if len(set(entry.tests)) != len(entry.tests):
            issues.append(f"{where}: duplicate test node ids")
        if entry.status == "verified" and not entry.tests:
            issues.append(
                f"{where}: 'verified' with no mapped test (V4-02.03)"
            )
        if entry.status == "unimplemented" and entry.tests:
            issues.append(
                f"{where}: 'unimplemented' cannot list tests — either "
                f"the checks exist (implemented_unverified/verified) or "
                f"the row is wrong"
            )
        if entry.status in ("deferred", "not_applicable") and not entry.note:
            issues.append(
                f"{where}: '{entry.status}' requires a note naming "
                f"reason/prerequisite (V4-02.06)"
            )
        for sc in entry.scenarios:
            scdef = idx.scenarios.get(sc)
            if scdef is None:
                issues.append(f"{where}: unknown scenario {sc}")
            elif rid not in scdef.anchors:
                issues.append(
                    f"{where}: scenario {sc} is not anchored to {rid} "
                    f"by the spec (anchors: {', '.join(scdef.anchors)})"
                )
        if collected is not None:
            for t in entry.tests:
                base = t.split("[", 1)[0]
                if base not in collected:
                    issues.append(
                        f"{where}: mapped test {t} is not collected by "
                        f"pytest (removed/renamed?)"
                    )
        # every discovered check must be registered in the map —
        # otherwise the map is stale (V4-61.03 'missing mappings')
        if rid in defined:
            missing = sorted(
                t for t in (disc.tests.get(rid) or {}) if t not in entry.tests
            )
            for t in missing:
                issues.append(
                    f"{where}: discovered check {t} cites {rid} but is "
                    f"not mapped — regenerate with --write"
                )
            for sc in sorted(disc.scenarios.get(rid) or ()):
                if sc not in entry.scenarios:
                    issues.append(
                        f"{where}: discovered scenario coverage {sc} is "
                        f"not mapped"
                    )

    # a requirement with discovered checks but no map row at all is
    # unmapped-but-should-be-covered
    for rid in sorted(disc.tests):
        if rid in defined and rid not in rm.entries:
            t = sorted(disc.tests[rid])[0]
            issues.append(
                f"{rid} has discovered checks (e.g. {t}) but no map row"
            )
    return issues


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_ledger(idx: SpecIndex, registry: dict, disc: Discovery) -> dict:
    """The machine-readable ledger.json document."""
    reqs = {}
    for rid, r in registry.items():
        reqs[rid] = {
            "section": f"{rid.split('-')[0]}-{r['section']:02d}",
            "section_title": r["section_title"],
            "spec": r["spec"],
            "spec_line": r["line"],
            "status": r["status"],
            "evidence": r["evidence"],
            "tests": r["tests"],
            "test_attribution": dict(
                sorted((disc.tests.get(rid) or {}).items())
            ),
            "scenarios": r["scenarios"],
            "spec_scenarios": r["spec_scenarios"],
            "file_claims": r["claims"],
            "code_refs": r["code_refs"],
            "text": r["text"],
            "note": r["note"],
        }
    summary = {s: 0 for s in STATUSES}
    for r in registry.values():
        summary[r["status"]] += 1
    scenarios = {}
    for sc_id, sc in sorted(idx.scenarios.items()):
        scenarios[sc_id] = {
            "spec": sc.spec,
            "text": sc.text,
            "anchors": list(sc.anchors),
            "tier": sc.tier,
            "tests": sorted(disc.scenario_tests.get(sc_id) or ()),
            "file_claims": sorted(disc.scenario_claims.get(sc_id) or ()),
        }
    return {
        "schema": 1,
        "generated_by": "tools/gen_v4_ledger.py",
        "specs": {
            f: {"sha256": idx.spec_sha256.get(f, "")} for f in SPEC_FILES
        },
        "statuses": list(STATUSES),
        "evidence_labels": list(EVIDENCE_LABELS),
        "summary": {
            "total": len(registry),
            "by_status": summary,
            "with_mapped_tests": sum(1 for r in registry.values() if r["tests"]),
            "scenarios_total": len(idx.scenarios),
            "scenarios_with_tests": sum(
                1 for s in scenarios.values() if s["tests"]
            ),
        },
        "requirements": reqs,
        "scenarios": scenarios,
    }


_MARK = {
    "verified": "x",
    "partial": "~",
    "implemented_unverified": "u",
    "deferred": "-",
    "not_applicable": "n",
    "unimplemented": " ",
}


def render_markdown(idx: SpecIndex, registry: dict) -> str:
    """REQUIREMENTS_V4.md — the human-facing registry."""
    out = [
        "# Verbatim v4 Requirements Registry",
        "",
        "Generated from `SPEC_V4.md` + `SPEC_V4_5.md` by",
        "`tools/gen_v4_ledger.py`. Do not hand-edit: curate",
        "`eval/v4/requirement_map.yaml` and re-run with `--write`.",
        "",
        "Statuses (V4-02.02): `[x]` verified — named executable checks",
        "cover the requirement; `[u]` implemented_unverified — code or",
        "coverage exists but no named check pins the whole requirement;",
        "`[~]` partial — note names the gap; `[-]` deferred — note keeps",
        "reason/prerequisite (V4-02.06); `[n]` not_applicable; `[ ]`",
        "unimplemented — no registered evidence. Status is never inferred",
        "from test counts (V4-02.02). Evidence labels (V4-02.07): DOC,",
        "CODE, LOCAL, BENCH, INDEPENDENT.",
        "",
    ]
    last_key = None
    counts = {s: 0 for s in STATUSES}
    for rid in sorted(registry):
        r = registry[rid]
        key = (r["spec"], r["section"])
        if key != last_key:
            prefix = rid.split("-")[0]
            out.append(
                f"## {prefix}-{r['section']:02d} — {r['section_title']}"
            )
            out.append("")
            last_key = key
        counts[r["status"]] += 1
        mark = _MARK[r["status"]]
        out.append(f"- [{mark}] **{rid}** ({r['status']}): {r['text']}")
        bits = []
        if r["evidence"]:
            bits.append("evidence: " + "/".join(r["evidence"]))
        if r["tests"]:
            show = r["tests"][:4]
            more = f" (+{len(r['tests']) - 4} more)" if len(r["tests"]) > 4 else ""
            bits.append("tests: " + ", ".join(show) + more)
        if r["scenarios"]:
            bits.append("scenarios: " + ", ".join(r["scenarios"]))
        elif r["spec_scenarios"]:
            bits.append(
                "spec scenarios (uncovered): " + ", ".join(r["spec_scenarios"])
            )
        if r["claims"]:
            bits.append("file claims: " + ", ".join(r["claims"][:3]))
        if r["code_refs"]:
            bits.append("code: " + ", ".join(r["code_refs"][:3]))
        if bits:
            out.append(f"  - {'; '.join(bits)}")
        if r["note"]:
            out.append(f"  - note: {r['note']}")
    out += [
        "",
        "## Summary",
        "",
        f"- verified: {counts['verified']}",
        f"- partial: {counts['partial']}",
        f"- implemented_unverified: {counts['implemented_unverified']}",
        f"- deferred: {counts['deferred']}",
        f"- not_applicable: {counts['not_applicable']}",
        f"- unimplemented: {counts['unimplemented']}",
        f"- total: {len(registry)}",
        "",
    ]
    return "\n".join(out)
