"""Executable-traceability ledger for SPEC_V6 (§00/§04, V6-00.06, V6-04.01).

The v6 ledger answers the same honesty question as v5 — which
requirement has *named executed evidence* behind it — under the same
two-axis status model (V6-00.06): implementation and qualification are
separate axes. An item may be implemented and locally measured yet
still unqualified; the rollup ``status`` is *derived*, never set
directly. It also carries the V5 tail (V6-04.02/03): the rows of
``eval/v5/dispositions_v5.json`` still ``planned`` or
``implemented_unmeasured``, read-only — the V5 file is never edited.

Pipeline (``tools/gen_v6_ledger.py`` drives it):

* ``load_specs`` parses SPEC_V6.md: every ``- V6-NN.MM:`` definition,
  every ``| FNN |`` scenario row (§07 — 32 scenarios), every
  ``| G6-NN Name |`` gate row (§08 — 9 gates), and every
  ``| PN |`` phase row (§09 — P0..P7). Duplicate definitions and
  references to ids that were never defined are hard errors
  (V6-00.03/00.06). ``### NN.M`` subsections are tracked so the §02.1
  causal-barrier program and the §03.1 neural path can bind their own
  phases.
* ``infer_*`` derives the planning map the spec itself declares:
  scenario stages come from §09 phase exit cells (``F17–F20`` -> P1);
  requirement stages come from explicit text directives or the
  curated ``SECTION_STAGE_FALLBACK`` keyed by ``(section, subsection)``;
  requirements anchor scenarios through their phase, through F-ids
  named in their own text, or through shared envelope tokens
  (``A0``/``A0-cache``/``A0-neural``/``A1``/``A3``); gates bind through
  ``G6-NN`` text mentions, through the F-scenarios their permits cells
  name, or through shared envelopes. Inference provenance is recorded
  per row (``stage_source``).
* ``dispositions_v6.json`` is the curated overlay: per id it may set
  ``owner``, ``profile``, ``stage``, ``code_surfaces``, extra
  ``scenarios``/``gate``, evidence labels, named ``executed_evidence``,
  ``implementation_status``, ``qualification_status``, ``blocker`` and
  ``note``. The generator merges it but never rewrites human judgments
  (V6-04.01 — same overlay semantics as V5).
* ``load_carried`` reads ``eval/v5/dispositions_v5.json`` and lists the
  V5 rows whose derived status is ``planned`` or
  ``implemented_unmeasured`` — the 121-row V5 tail V6-04.02/03 must
  re-disposition. It is a read-only overlay; nothing writes back.
* ``validate`` enforces the V6-00.03/00.06 contract: no
  duplicate/unknown ids, no references to undefined
  requirements/scenarios/gates, no out-of-registry ids (F33+, G6-09+),
  no stale spec hash, no ``qualified``/``locally_measured`` without
  named executed evidence, no unknown overlay keys or stale code
  surfaces.

Honesty rule (V6-00.06): everything seeds ``planned``/``not_run``.
``locally_measured`` and ``qualified`` both require executed evidence
named in the dispositions file; a spec citation is never evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Implementation axis (V6-00.06): has the code been written?
IMPLEMENTATION_STATUSES = (
    "planned",
    "in_progress",
    "implemented",
    "not_applicable",
)

#: Qualification axis (V6-00.06): has executed evidence qualified it?
#: ``locally_measured`` = ran here, unqualified — distinct from both
#: ``not_run`` and ``qualified`` (measurement, qualification, and
#: recommendation stay separate).
QUALIFICATION_STATUSES = (
    "not_run",
    "locally_measured",
    "qualified",
    "failed",
    "not_applicable",
    # V6-04.02/04.03 disposition for inherited tail rows: an explicit
    # re-deferral with owner+reason. Derives no new rollup — the row
    # stays visibly open under its implementation axis.
    "deferred",
)

#: Derived rollup statuses (never set by hand).
STATUSES = (
    "planned",
    "in_progress",
    "implemented_unmeasured",
    "locally_measured",
    "qualified",
    "failed",
    "not_applicable",
)

#: Evidence labels (same vocabulary as V5).
EVIDENCE_LABELS = ("DOC", "CODE", "LOCAL", "BENCH", "INDEPENDENT")

#: Implementation phases, §09 (P0..P7 — one more than V5).
STAGES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")

#: Consumer performance envelopes named by §02.2 / §08. Longest-match
#: tokenization: ``A0-neural`` never counts as plain ``A0``.
ENVELOPES = ("A0", "A0-cache", "A0-neural", "A1", "A3")

#: Declared registries (V6-00.03): F01–F32 scenarios, G6-00–G6-08 gates.
SCENARIO_RANGE = (1, 32)
GATE_RANGE = (0, 8)

#: The spec this ledger covers, plus inherited specs consulted only to
#: resolve curated E/C/D scenario ids and G4/G5 gate ids (read-only;
#: never generated from).
SPEC_FILE = "SPEC_V6.md"
INHERITED_SPEC_FILES = ("SPEC_V5.md", "SPEC_V4.md", "SPEC_V4_5.md")

#: Default locations (relative to repo root).
DEFAULT_DISPOSITIONS_PATH = os.path.join("eval", "v6", "dispositions_v6.json")
DEFAULT_JSON_PATH = os.path.join("eval", "v6", "ledger_v6.json")
DEFAULT_SUMMARY_PATH = os.path.join("eval", "v6", "summary.md")

#: The carried V5 tail (V6-04.02/03): this file is read, never edited.
V5_DISPOSITIONS_PATH = os.path.join("eval", "v5", "dispositions_v5.json")

#: V5 rollup statuses that count as unfinished tail rows.
CARRIED_STATUSES = ("planned", "implemented_unmeasured")

#: V6-00.01 authorizes implementation, so rows start unblocked; a
#: blocker appears only when a disposition names one.
DEFAULT_BLOCKER = ""

#: Requirement -> phase when the spec's own structure cannot decide.
#: Keyed by ``(section, subsection)`` — ``subsection`` is the M in the
#: enclosing ``### NN.M`` heading, or -1 for requirements directly under
#: the ``##`` heading. Provenance is ``stage_source="fallback"`` (or
#: ``"unbound"`` when no entry matches — §07/§08/§10 are registries and
#: stop conditions, not work phases).
SECTION_STAGE_FALLBACK = {
    (0, -1): "P0",   # authority + ledger contract — P0 freezes contracts
    (1, -1): "P3",   # adoption thesis — P3 ships the adoption surface
    (1, 1): "P3",    # §01.1 distribution — P3 packaging/name/entry points
    (2, -1): "P2",   # §02 hot path — P2 typed-memory ranking + cache
    (2, 1): "P1",    # §02.1 causal-barrier program — P1 names it verbatim
    (2, 2): "P2",    # §02.2 envelopes — P2 executes A0/A0-cache
    (3, -1): "P4",   # §03 trust defaults — the G6-03 trust program
    (3, 1): "P5",    # §03.1 neural artifact path — P5 names it verbatim
    (3, 2): "P4",    # §03.2 consolidation — P4 names it verbatim
    (3, 3): "P4",    # §03.3 feedback/controller — P4 names it verbatim
    (4, -1): "P6",   # §04 V5-tail closure — P6 names it verbatim
    (5, -1): "P3",   # §05 service surface — P3 names it verbatim
    (6, -1): "P6",   # §06 comparator ladder — P6 names it verbatim
    (7, -1): "",     # scenario registry — bound per release level, §08.01
    (8, -1): "",     # gate registry — bound per release level, §08.01
    (9, -1): "P0",   # the plan itself executes from contract freeze
    (10, -1): "",    # stop conditions — discipline, not a phase
}

#: One capability label per spec section (deterministic section map;
#: unknown sections fall back to a slug of the section title).
SECTION_CAPABILITY = {
    0: "authority", 1: "adoption", 2: "performance",
    3: "trust-defaults", 4: "qualification-closure",
    5: "service-surface", 6: "comparators", 7: "scenarios",
    8: "gates", 9: "build-phases", 10: "stop-conditions",
}

# ---------------------------------------------------------------------------
# spec grammar
# ---------------------------------------------------------------------------

#: `- V6-NN.MM: text` — definition lines only.
_DEF_RE = re.compile(r"^-\s+V6-(\d{2})\.(\d{2}):\s*(.*)$")
#: `## NN. Title` (top-level sections; `###` subsections do not match).
_SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$")
#: `### NN.M Title` — tracked so §02.1/§03.1-style programs bind their
#: own phases; the subsection number resets at every `##` heading.
_SUBSECTION_RE = re.compile(r"^###\s+(\d+)\.(\d+)\s*[.:—-]?\s*(.*)$")
#: Any `V6-NN.MM` mention incl. `/MM` and `/NN.MM` shorthand
#: continuations (`V6-04.02/03`).
_ANCHOR_RE = re.compile(r"\bV6-(\d{2})\.(\d{2})((?:/\d{2}(?:\.\d{2})?)*)")
_ANCHOR_PART_RE = re.compile(r"/(\d{2})(?:\.(\d{2}))?")
#: `| F01 | desc |` scenario rows (§07 — two cells, no stage/section).
_SCENARIO_ROW_RE = re.compile(r"^\|\s*(F\d{2})\s*\|(.*)\|\s*$")
#: `| G6-00 Authority | permits |` gate rows (§08 — name inside cell 1).
_GATE_ROW_RE = re.compile(r"^\|\s*(G6-\d{2})\s*([^|]*)\|(.*)\|\s*$")
#: `| P0 | content | exit |` phase rows (§09).
_STAGE_ROW_RE = re.compile(r"^\|\s*(P[0-7])\s*\|(.*)\|\s*$")
#: `§NN`, `§NN.MM`, `§NN–§MM`, `§NN.MM–MM.MM` (en dash). A range whose
#: endpoints carry subsections stays inside the first section.
_SECTION_REF_RE = re.compile(
    r"§(\d{2})(?:\.(\d{2}))?(?:\s*–\s*§?(\d{2})(?:\.(\d{2}))?)?"
)
#: Bare scenario ids and explicit `FNN–FNN` ranges (for references).
_FID_RE = re.compile(r"\bF(\d{2})\b")
_FRANGE_RE = re.compile(r"\bF(\d{2})`?\s*–\s*`?F(\d{2})")
#: `G6-NN` incl. `/NN` continuations (`G6-01/03/04`) and `–` ranges
#: (`G6-00`–`G6-08` denotes the whole registry).
_GATE_REF_RE = re.compile(r"\bG6-(\d{2})((?:/\d{2})*)")
_GATE_RANGE_RE = re.compile(r"\bG6-(\d{2})`?\s*–\s*`?G6-(\d{2})")
#: Envelope tokens, longest match first (`A0-neural` != `A0`).
_ENVELOPE_RE = re.compile(r"\bA0(?:-cache|-neural)?\b|\bA1\b|\bA3\b")
#: Explicit phase directives inside requirement text (`P0 MUST define`).
_STAGE_DIRECTIVE_RE = re.compile(r"\bP([0-7])(?=\s+MUST)")
#: Profile names the spec registers.
_PROFILE_RE = re.compile(r"\b(local_memory_neural|local_memory|local_rules|embedded)\b")
#: Inherited ids — recorded per row for context, never satisfied here.
#: `V4-28.03`, `V5-30`, `V5-11` (section refs) all land in
#: ``inherited_requirements`` as raw mentions.
_INHERITED_REQ_RE = re.compile(r"\bV[45]-\d{2}(?:\.\d{2})?\b")
_INHERITED_GATE_RE = re.compile(r"\bG([45])-(\d{2})((?:/\d{2})*)")
_INHERITED_SC_RE = re.compile(r"\b([ECD])(\d{2})\b|\bF4-(\d{2})\b")
#: First-cell `| E01 |`/`| C01 |`/`| D01 |` rows and `| G4-NN |`/
#: `| G5-NN |` gate rows in the inherited specs (overlay validation).
_INH_SCENARIO_ROW_RE = re.compile(r"^\|\s*([ECD]\d{2})\s*\|", re.M)
_INH_GATE_ROW_RE = re.compile(r"^\|\s*(G[45]-\d{2})\s", re.M)


def _expand_anchor(m: re.Match) -> list:
    """Expand one V6 anchor match into every id it denotes.

    `V6-04.02/03` -> [`V6-04.02`, `V6-04.03`]; a bare `/MM` keeps the
    base section.
    """
    sec, sub, tail = m.groups()
    ids = [f"V6-{sec}.{sub}"]
    for seg, sub2 in _ANCHOR_PART_RE.findall(tail or ""):
        ids.append(f"V6-{seg}.{sub2}" if sub2 else f"V6-{sec}.{seg}")
    return ids


def anchors_in(text: str) -> list:
    """All `V6-NN.MM` ids mentioned in ``text`` (shorthand expanded)."""
    out = []
    for m in _ANCHOR_RE.finditer(text):
        out.extend(_expand_anchor(m))
    return out


def _f_mentions(text: str) -> list:
    """`FNN` singles, excluding members of `FNN–FNN` ranges.

    `F01, F10–F12` -> [F01, F10, F11, F12] is NOT what this returns —
    a range is a namespace span, not a per-scenario binding, so only
    the explicit `F01` binds. Ranges expand in :func:`fids_in` for
    reference validation and gate-cell parsing instead.
    """
    spans = [m.span() for m in _FRANGE_RE.finditer(text)]
    out = set()
    for m in _FID_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        out.add(f"F{m.group(1)}")
    return sorted(out)


def fids_in(text: str) -> list:
    """All `FNN` ids denoted by ``text`` — singles plus en-dash ranges
    (`F17`–`F21`) fully expanded."""
    out = set(_f_mentions(text))
    for a, b in _FRANGE_RE.findall(text):
        out.update(f"F{n:02d}" for n in range(int(a), int(b) + 1))
    return sorted(out)


def _gate_mentions(text: str) -> list:
    """`G6-NN` singles + `/NN` continuations, excluding range members.

    `G6-01/03/04` -> all three; `G6-00`–`G6-08` -> [] (a range is a
    namespace reference, not a per-gate binding).
    """
    spans = [m.span() for m in _GATE_RANGE_RE.finditer(text)]
    out = set()
    for m in _GATE_REF_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        out.add(f"G6-{m.group(1)}")
        out.update(f"G6-{n}" for n in re.findall(r"/(\d{2})", m.group(2) or ""))
    return sorted(out)


def gates_in(text: str) -> list:
    """All `G6-NN` ids denoted by ``text`` — singles, continuations,
    and en-dash ranges (`G6-00`–`G6-08`) fully expanded."""
    out = set(_gate_mentions(text))
    for a, b in _GATE_RANGE_RE.findall(text):
        out.update(f"G6-{n:02d}" for n in range(int(a), int(b) + 1))
    return sorted(out)


def envelopes_in(text: str) -> list:
    """Envelope tokens in ``text`` — longest match wins so
    ``A0-neural`` never double-counts as ``A0``."""
    return sorted(set(_ENVELOPE_RE.findall(text)))


def sections_in(text: str) -> list:
    """Top-level spec sections a `§` reference list denotes.

    `§02` -> [2]; `§05.12–05.13` -> [5] (a subsection range stays in
    its section).
    """
    out = set()
    for m in _SECTION_REF_RE.finditer(text):
        a, sub_a, b, sub_b = m.groups()
        if b and not sub_a and not sub_b:
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(a))
    return sorted(out)


def _stages_in(text: str) -> list:
    """Phase tokens in a cell: `P2/P3` -> both; `All` -> none."""
    return sorted(
        {f"P{d}" for d in re.findall(r"\bP([0-7])\b", text)},
        key=STAGES.index,
    )


def _inherited_mentions(text: str) -> dict:
    """V4/V5 ids a text mentions — context only, never validated."""
    gates = set()
    for gen, num, tail in _INHERITED_GATE_RE.findall(text):
        gates.add(f"G{gen}-{num}")
        gates.update(f"G{gen}-{n}" for n in re.findall(r"/(\d{2})", tail or ""))
    scens = set()
    for letter, num, f4 in _INHERITED_SC_RE.findall(text):
        scens.add(f"{letter}{num}" if letter else f"F4-{f4}")
    return {
        "inherited_requirements": sorted(set(_INHERITED_REQ_RE.findall(text))),
        "inherited_gates": sorted(gates),
        "inherited_scenarios": sorted(scens),
    }


class LedgerError(Exception):
    """Raised for unparseable inputs; validation failures are reported
    as issue lists instead (see :func:`validate`)."""


# ---------------------------------------------------------------------------
# spec parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequirementDef:
    """One `- V6-NN.MM:` definition line."""

    req_id: str
    spec: str           # spec file name, e.g. "SPEC_V6.md"
    section: int        # enclosing `## NN.` section number
    subsection: int     # M in enclosing `### NN.M`, -1 when directly under `##`
    id_section: int     # the NN inside the id itself
    section_title: str
    text: str
    line: int


@dataclass(frozen=True)
class ScenarioDef:
    """One §07 `| FNN |` acceptance-scenario row."""

    sc_id: str          # "F17"
    spec: str
    text: str
    envelopes: tuple    # envelope tokens named in the row (F09 -> A0)
    line: int


@dataclass(frozen=True)
class GateDef:
    """One §08 `| G6-NN Name |` release-gate row."""

    gate_id: str        # "G6-00"
    spec: str
    name: str
    permits: str        # the "Permits" cell verbatim
    scenarios: tuple    # F-ids the permits cell names (ranges expanded)
    envelopes: tuple    # envelope tokens the permits cell names
    sections: tuple     # §-refs inside the permits cell
    line: int


@dataclass(frozen=True)
class StageDef:
    """One §09 `| PN |` phase row."""

    stage_id: str       # "P1"
    spec: str
    title: str          # V6 phase rows have no title cell — ""
    main_work: str      # the "Content" cell verbatim
    exit_text: str      # the "Exit" cell verbatim
    exit_gates: tuple   # G6-ids named in the exit cell
    exit_scenarios: tuple  # F-ids named in the exit cell (ranges expanded)
    exit_envelopes: tuple  # envelope tokens named in the exit cell
    line: int


@dataclass
class SpecIndex:
    """Merged parse of SPEC_V6.md (+ inherited E/C/D/G4/G5 id sets)."""

    requirements: dict = field(default_factory=dict)   # id -> RequirementDef
    scenarios: dict = field(default_factory=dict)      # F-id -> ScenarioDef
    gates: dict = field(default_factory=dict)          # G6-id -> GateDef
    stages: dict = field(default_factory=dict)         # P-id -> StageDef
    section_titles: dict = field(default_factory=dict)  # int -> title
    references: dict = field(default_factory=dict)     # id -> [spec:line]
    duplicates: list = field(default_factory=list)     # [str]
    spec_sha256: dict = field(default_factory=dict)    # file -> hex
    inherited_scenarios: set = field(default_factory=set)  # E/C/D ids
    inherited_gates: set = field(default_factory=set)      # G4/G5 ids

    def scenario_envelopes(self, sc_id: str) -> tuple:
        sc = self.scenarios.get(sc_id)
        return sc.envelopes if sc else ()


def _record_refs(idx: SpecIndex, text: str, where: str,
                 skip_req: Optional[str] = None) -> None:
    """Register every V6 id ``text`` mentions (for undefined-ref checks).

    V4/V5 identifiers (``V5-30``, ``G4-10``, ``E02``, ``F4-11``) are
    inherited namespaces — they are recorded per row via
    :func:`_inherited_mentions`, never as references to satisfy here.
    """
    for rid in anchors_in(text):
        if rid != skip_req:
            idx.references.setdefault(rid, []).append(where)
    for sc in fids_in(text):
        idx.references.setdefault(sc, []).append(where)
    for g in gates_in(text):
        idx.references.setdefault(g, []).append(where)


def _parse_spec_text(raw: str, fname: str, idx: SpecIndex) -> None:
    """Parse one spec's text into ``idx`` (testable without files)."""
    idx.spec_sha256[fname] = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    section = -1
    subsection = -1
    section_title = ""
    for n, line in enumerate(raw.splitlines(), 1):
        m = _SECTION_RE.match(line)
        if m:
            section = int(m.group(1))
            subsection = -1
            section_title = m.group(2).strip()
            idx.section_titles[section] = section_title
            _record_refs(idx, line, f"{fname}:{n}")
            continue
        sm = _SUBSECTION_RE.match(line)
        if sm and int(sm.group(1)) == section:
            subsection = int(sm.group(2))
            _record_refs(idx, line, f"{fname}:{n}")
            continue
        m = _DEF_RE.match(line)
        if m:
            rid = f"V6-{m.group(1)}.{m.group(2)}"
            where = f"{fname}:{n}"
            if rid in idx.requirements:
                prev = idx.requirements[rid]
                idx.duplicates.append(
                    f"{rid} defined twice: {prev.spec}:{prev.line} "
                    f"and {where}"
                )
                continue
            idx.requirements[rid] = RequirementDef(
                req_id=rid,
                spec=fname,
                section=section,
                subsection=subsection,
                id_section=int(m.group(1)),
                section_title=section_title,
                text=m.group(3).strip(),
                line=n,
            )
            _record_refs(idx, m.group(3), where, skip_req=rid)
            continue
        sm = _SCENARIO_ROW_RE.match(line)
        if sm:
            sc_id, rest = sm.group(1), sm.group(2)
            cells = [c.strip() for c in rest.split("|")]
            desc = cells[0] if len(cells) > 0 else ""
            where = f"{fname}:{n}"
            if sc_id in idx.scenarios:
                prev = idx.scenarios[sc_id]
                idx.duplicates.append(
                    f"scenario {sc_id} defined twice: "
                    f"{prev.spec}:{prev.line} and {where}"
                )
            else:
                idx.scenarios[sc_id] = ScenarioDef(
                    sc_id=sc_id, spec=fname, text=desc,
                    envelopes=tuple(envelopes_in(desc)), line=n,
                )
            _record_refs(idx, rest, where)
            continue
        gm = _GATE_ROW_RE.match(line)
        if gm:
            gate_id, name, rest = gm.group(1), gm.group(2).strip(), gm.group(3)
            cells = [c.strip() for c in rest.split("|")]
            permits = cells[0] if len(cells) > 0 else ""
            where = f"{fname}:{n}"
            if gate_id in idx.gates:
                prev = idx.gates[gate_id]
                idx.duplicates.append(
                    f"gate {gate_id} defined twice: "
                    f"{prev.spec}:{prev.line} and {where}"
                )
            else:
                idx.gates[gate_id] = GateDef(
                    gate_id=gate_id, spec=fname, name=name,
                    permits=permits,
                    scenarios=tuple(fids_in(permits)),
                    envelopes=tuple(envelopes_in(permits)),
                    sections=tuple(sections_in(rest)),
                    line=n,
                )
            _record_refs(idx, rest, where)
            continue
        pm = _STAGE_ROW_RE.match(line)
        if pm:
            stage_id, rest = pm.group(1), pm.group(2)
            cells = [c.strip() for c in rest.split("|")]
            work = cells[0] if len(cells) > 0 else ""
            exit_text = cells[1] if len(cells) > 1 else ""
            where = f"{fname}:{n}"
            if stage_id in idx.stages:
                prev = idx.stages[stage_id]
                idx.duplicates.append(
                    f"stage {stage_id} defined twice: "
                    f"{prev.spec}:{prev.line} and {where}"
                )
            else:
                idx.stages[stage_id] = StageDef(
                    stage_id=stage_id, spec=fname, title="",
                    main_work=work, exit_text=exit_text,
                    exit_gates=tuple(gates_in(exit_text)),
                    exit_scenarios=tuple(fids_in(exit_text)),
                    exit_envelopes=tuple(envelopes_in(exit_text)),
                    line=n,
                )
            _record_refs(idx, rest, where)
            continue
        _record_refs(idx, line, f"{fname}:{n}")


def _scan_inherited(raw: str) -> tuple:
    """(scenario ids, gate ids) an inherited spec defines."""
    scenarios = {m.group(1) for m in _INH_SCENARIO_ROW_RE.finditer(raw)}
    gates = {m.group(1) for m in _INH_GATE_ROW_RE.finditer(raw)}
    return scenarios, gates


def load_specs(root: str, spec_file: str = SPEC_FILE) -> SpecIndex:
    """Parse SPEC_V6.md; also collect inherited E/C/D + G4/G5 ids."""
    idx = SpecIndex()
    path = os.path.join(root, spec_file)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        raise LedgerError(f"cannot read {path}: {e}")
    _parse_spec_text(raw, spec_file, idx)
    for fname in INHERITED_SPEC_FILES:
        ipath = os.path.join(root, fname)
        try:
            with open(ipath, "r", encoding="utf-8") as f:
                sc, gates = _scan_inherited(f.read())
                idx.inherited_scenarios |= sc
                idx.inherited_gates |= gates
        except OSError:
            continue  # inherited specs absent: their validation is skipped
    return idx


# ---------------------------------------------------------------------------
# inference — the planning map the spec itself declares (V6-00.06)
# ---------------------------------------------------------------------------


def infer_scenario_stages(idx: SpecIndex) -> dict:
    """scenario id -> (stage, source) from §09 phase exit cells.

    A phase exit naming `F17–F20` binds those scenarios to the phase
    (`"phase_exit"` provenance). Scenarios no exit names (F02–F14, F16,
    F21) stay unbound — they reach requirements through gate cells and
    envelope tokens instead.
    """
    sc_stage: dict = {}
    for st in idx.stages.values():
        for sc in st.exit_scenarios:
            sc_stage.setdefault(sc, (st.stage_id, "phase_exit"))
    return sc_stage


def infer_requirement_stage(idx: SpecIndex, req: RequirementDef,
                            sc_stage: dict) -> tuple:
    """(stage, source) for one requirement.

    An explicit `PN MUST` directive or a named F-scenario inside the
    requirement text wins; otherwise the ``(section, subsection)``
    fallback applies; `""` means no phase is inferable.
    """
    m = _STAGE_DIRECTIVE_RE.search(req.text)
    if m:
        return f"P{m.group(1)}", "text"
    text_stages = sorted(
        {sc_stage[f][0] for f in _f_mentions(req.text) if f in sc_stage},
        key=STAGES.index,
    )
    if text_stages:
        return text_stages[0], "text"
    key = (req.section, req.subsection)
    if key in SECTION_STAGE_FALLBACK:
        return SECTION_STAGE_FALLBACK[key], "fallback"
    return SECTION_STAGE_FALLBACK.get((req.section, -1), ""), "fallback"


def infer_requirement_scenarios(idx: SpecIndex, req: RequirementDef,
                                sc_stage: dict, stage: str) -> list:
    """F-scenarios a requirement anchors.

    Three spec-declared channels: F-ids the requirement text names
    (ranges excluded — a range declares the registry), scenarios whose
    phase exit matches the requirement's stage, and scenarios sharing
    an envelope token with the requirement text (``A0`` -> F09).
    """
    out = set(_f_mentions(req.text))
    env = set(envelopes_in(req.text))
    for sc in idx.scenarios.values():
        if stage and sc_stage.get(sc.sc_id, ("", ""))[0] == stage:
            out.add(sc.sc_id)
        elif env and set(sc.envelopes) & env:
            out.add(sc.sc_id)
    return sorted(out)


def infer_requirement_gates(idx: SpecIndex, req: RequirementDef,
                            scenarios: list) -> list:
    """G6 gates a requirement binds: explicit `G6-NN` text mentions
    (ranges excluded), gates whose permits cell names one of the
    requirement's anchored scenarios, gates sharing an envelope token
    (`A0`/`A1` -> G6-01, `A0-neural` -> G6-02, `A3` -> G6-07), and
    gates whose permits cell §-references the requirement's section.
    """
    out = set(_gate_mentions(req.text))
    env = set(envelopes_in(req.text))
    sc = set(scenarios)
    for g in idx.gates.values():
        if sc and set(g.scenarios) & sc:
            out.add(g.gate_id)
            continue
        if env and set(g.envelopes) & env:
            out.add(g.gate_id)
            continue
        if req.section in g.sections:
            out.add(g.gate_id)
    return sorted(out)


def capability_for(req: RequirementDef) -> str:
    """One capability label per section (slug of the title as fallback)."""
    if req.section in SECTION_CAPABILITY:
        return SECTION_CAPABILITY[req.section]
    slug = re.sub(r"[^a-z0-9]+", "-", req.section_title.lower()).strip("-")
    return slug or "unsectioned"


def profiles_in(text: str) -> list:
    """Registered profile names a text mentions (`local_memory`, …)."""
    return sorted(set(_PROFILE_RE.findall(text)))


def rollup_status(impl: str, qual: str) -> str:
    """Derived ``status`` — never set directly (V6-00.06)."""
    if qual == "failed":
        return "failed"
    if qual == "not_applicable" or impl == "not_applicable":
        return "not_applicable"
    if qual == "qualified":
        return "qualified"
    if qual == "locally_measured":
        return "locally_measured"
    if impl == "implemented":
        return "implemented_unmeasured"
    if impl == "in_progress":
        return "in_progress"
    return "planned"


# ---------------------------------------------------------------------------
# carried V5 tail (V6-04.02/03) — read-only
# ---------------------------------------------------------------------------


@dataclass
class CarriedTail:
    """The V5 ledger rows this program must re-disposition.

    ``rows`` carries ``id``, derived ``status``,
    ``implementation_status``, ``qualification_status`` and ``note``
    from ``eval/v5/dispositions_v5.json`` — the file is read, never
    written.
    """

    source_path: str = V5_DISPOSITIONS_PATH
    sha256: str = ""
    missing: bool = False
    rows: list = field(default_factory=list)

    def by_status(self) -> dict:
        counts = {s: 0 for s in CARRIED_STATUSES}
        for r in self.rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return counts


def load_carried(root: str,
                 source: str = V5_DISPOSITIONS_PATH) -> CarriedTail:
    """Read the V5 dispositions overlay and list its unfinished rows.

    A V5 row is carried when its *derived* status is ``planned`` or
    ``implemented_unmeasured`` — the same rollup the V5 ledger applies.
    A missing file yields an empty tail flagged ``missing`` (honest
    absence, not an error — a tree without the V5 overlay cannot carry
    a tail).
    """
    tail = CarriedTail(source_path=source)
    path = os.path.join(root, source)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        tail.missing = True
        return tail
    tail.sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LedgerError(f"cannot parse {path}: {e}")
    reqs = (doc or {}).get("requirements") or {}
    for rid in sorted(reqs):
        entry = reqs[rid] or {}
        impl = entry.get("implementation_status") or "planned"
        qual = entry.get("qualification_status") or "not_run"
        status = rollup_status(impl, qual)
        if status not in CARRIED_STATUSES:
            continue
        tail.rows.append({
            "id": rid,
            "status": status,
            "implementation_status": impl,
            "qualification_status": qual,
            "note": entry.get("note") or "",
        })
    return tail


# ---------------------------------------------------------------------------
# dispositions overlay (eval/v6/dispositions_v6.json) — curated, JSON
# ---------------------------------------------------------------------------

#: Keys a disposition entry may set; anything else is a typo and fails
#: --check rather than silently dropping.
DISPOSITION_KEYS = frozenset({
    "owner", "profile", "stage", "code_surfaces", "scenarios", "gate",
    "evidence_type", "executed_evidence", "implementation_status",
    "qualification_status", "blocker", "note",
})

_DISPOSITION_LIST_KEYS = frozenset({
    "profile", "code_surfaces", "scenarios", "gate", "evidence_type",
    "executed_evidence",
})


@dataclass
class Dispositions:
    """Parsed eval/v6/dispositions_v6.json."""

    spec_sha256: dict = field(default_factory=dict)
    entries: dict = field(default_factory=dict)   # req_id -> dict of overrides
    source_path: str = DEFAULT_DISPOSITIONS_PATH


def _no_dup_object(pairs):
    """object_pairs_hook rejecting duplicate JSON keys (V6-00.03)."""
    out = {}
    for k, v in pairs:
        if k in out:
            raise LedgerError(f"duplicate key {k!r} in dispositions file")
        out[k] = v
    return out


def load_dispositions_from_text(text: str, source: str = "<dispositions>") -> Dispositions:
    """Parse dispositions JSON from a string."""
    try:
        raw = json.loads(text, object_pairs_hook=_no_dup_object)
    except LedgerError:
        raise
    except json.JSONDecodeError as e:
        raise LedgerError(f"cannot parse {source}: {e}")
    raw = raw or {}
    if not isinstance(raw, dict):
        raise LedgerError(f"{source}: top level must be a JSON object")
    disp = Dispositions(source_path=source)
    disp.spec_sha256 = {
        str(k): str(v) for k, v in (raw.get("spec_sha256") or {}).items()
    }
    reqs = raw.get("requirements") or {}
    if not isinstance(reqs, dict):
        raise LedgerError(f"{source}: 'requirements' must be an object")
    for rid, body in reqs.items():
        if not isinstance(body, dict):
            raise LedgerError(f"{source}: entry {rid} must be an object")
        disp.entries[rid] = dict(body)
    return disp


def load_dispositions(path: str) -> Dispositions:
    """Load dispositions_v6.json; structural errors raise LedgerError."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return load_dispositions_from_text(f.read(), path)
    except LedgerError:
        raise
    except OSError as e:
        raise LedgerError(f"cannot read {path}: {e}")


def render_dispositions(disp: Dispositions) -> str:
    """Canonical serialization — `--write` rewrites the file in exactly
    this form so freshness checks are byte-exact."""
    doc = {
        "schema": 1,
        "_doc": (
            "Curated overlay for the V6 requirements ledger "
            "(SPEC_V6 V6-00.06/04.01). Keys per requirement: owner, "
            "profile, stage, code_surfaces, scenarios (F ids plus "
            "inherited E/C/D ids added beyond the spec's phase "
            "anchors), gate (G6 ids; G4/G5 inherited), evidence_type "
            "(DOC/CODE/LOCAL/BENCH/INDEPENDENT), executed_evidence "
            "(named executed checks/runs — required for "
            "locally_measured/qualified), implementation_status, "
            "qualification_status, blocker, note. spec_sha256 is "
            "refreshed by `tools/gen_v6_ledger.py --write`."
        ),
        "spec_sha256": dict(sorted(disp.spec_sha256.items())),
        "requirements": {
            rid: disp.entries[rid] for rid in sorted(disp.entries)
        },
    }
    return json.dumps(doc, indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# registry build — spec definitions + inference + overlay
# ---------------------------------------------------------------------------


def build_registry(idx: SpecIndex, disp: Dispositions) -> dict:
    """Merge spec defs + inference + dispositions into per-id records."""
    sc_stage = infer_scenario_stages(idx)
    registry = {}
    for rid in sorted(idx.requirements):
        d = idx.requirements[rid]
        entry = disp.entries.get(rid) or {}
        stage, stage_source = infer_requirement_stage(idx, d, sc_stage)
        if entry.get("stage"):
            stage, stage_source = entry["stage"], "disposition"
        spec_scenarios = infer_requirement_scenarios(idx, d, sc_stage, stage)
        scenarios = sorted(
            set(spec_scenarios) | set(entry.get("scenarios") or [])
        )
        gates = sorted(
            set(infer_requirement_gates(idx, d, scenarios))
            | set(entry.get("gate") or [])
        )
        profiles = entry.get("profile")
        if profiles is None:
            profiles = profiles_in(d.text) or ["*"]
        impl = entry.get("implementation_status") or "planned"
        qual = entry.get("qualification_status") or "not_run"
        inherited = _inherited_mentions(d.text)
        registry[rid] = {
            "id": rid,
            "spec": d.spec,
            "section": d.section,
            "subsection": d.subsection,
            "id_section": d.id_section,
            "section_title": d.section_title,
            "line": d.line,
            "text": d.text,
            "capability": capability_for(d),
            "profile": list(profiles),
            "owner": entry.get("owner") or "unset",
            "stage": stage,
            "stage_source": stage_source,
            "code_surfaces": list(entry.get("code_surfaces") or []),
            "scenarios": scenarios,
            "spec_scenarios": spec_scenarios,
            "gate": gates,
            "envelopes": envelopes_in(d.text),
            "inherited_requirements": inherited["inherited_requirements"],
            "inherited_gates": inherited["inherited_gates"],
            "inherited_scenarios": inherited["inherited_scenarios"],
            "evidence_type": list(entry.get("evidence_type") or []),
            "executed_evidence": list(entry.get("executed_evidence") or []),
            "implementation_status": impl,
            "qualification_status": qual,
            "status": rollup_status(impl, qual),
            "blocker": entry.get("blocker") or DEFAULT_BLOCKER,
            "note": entry.get("note") or "",
        }
    return registry


# ---------------------------------------------------------------------------
# validation (V6-00.03/00.06)
# ---------------------------------------------------------------------------


def _is_scenario_id(s: str) -> bool:
    return bool(re.match(r"^[FECD]\d{2}$", s))


def _is_gate_id(s: str) -> bool:
    return bool(re.match(r"^G[456]-\d{2}$", s))


def validate(idx: SpecIndex, disp: Dispositions, root: Optional[str] = None) -> list:
    """Return ledger violations; empty means the ledger is valid.

    ``root`` enables the code-surface existence check (a surface that
    disappears is a stale mapping).
    """
    issues: list = []

    for dup in idx.duplicates:
        issues.append(f"duplicate spec definition: {dup}")

    defined = set(idx.requirements)
    for rid, refs in sorted(idx.references.items()):
        if _is_scenario_id(rid):
            if rid.startswith("F") and rid not in idx.scenarios:
                issues.append(
                    f"spec reference to undefined scenario {rid} ({refs[0]})"
                )
            continue
        if _is_gate_id(rid):
            if rid.startswith("G6") and rid not in idx.gates:
                issues.append(
                    f"spec reference to undefined gate {rid} ({refs[0]})"
                )
            continue
        if rid not in defined:
            issues.append(
                f"spec reference to undefined requirement {rid} "
                f"({refs[0]}, +{len(refs) - 1} more)"
            )

    # requirement id section must match its enclosing ## section —
    # a V6-05.* row under §06 is a spec defect, not a parse difference
    for rid, d in sorted(idx.requirements.items()):
        if d.id_section != d.section:
            issues.append(
                f"{rid} is defined under §{d.section:02d} "
                f"({d.spec}:{d.line}) — id/section mismatch"
            )

    # declared registries (V6-00.03): F01–F32, G6-00–G6-08 — a row
    # outside the declared span is a spec defect
    for sc in sorted(idx.scenarios.values(), key=lambda s: s.sc_id):
        n = int(sc.sc_id[1:])
        if not (SCENARIO_RANGE[0] <= n <= SCENARIO_RANGE[1]):
            issues.append(
                f"scenario {sc.sc_id} outside the declared "
                f"F{SCENARIO_RANGE[0]:02d}–F{SCENARIO_RANGE[1]:02d} "
                f"registry ({sc.spec}:{sc.line})"
            )
    for g in sorted(idx.gates.values(), key=lambda g: g.gate_id):
        n = int(g.gate_id.split("-")[1])
        if not (GATE_RANGE[0] <= n <= GATE_RANGE[1]):
            issues.append(
                f"gate {g.gate_id} outside the declared "
                f"G6-{GATE_RANGE[0]:02d}–G6-{GATE_RANGE[1]:02d} "
                f"registry ({g.spec}:{g.line})"
            )

    # pinned spec hashes must match the file on disk
    for fname, pinned in sorted(disp.spec_sha256.items()):
        actual = idx.spec_sha256.get(fname)
        if actual is None:
            issues.append(f"dispositions pin hash for unknown spec {fname}")
        elif pinned != actual:
            issues.append(
                f"stale spec hash for {fname}: dispositions pin "
                f"{pinned[:12]}..., file is {actual[:12]}... — run "
                f"`tools/gen_v6_ledger.py --write` after reviewing the "
                f"spec diff"
            )
    for fname in idx.spec_sha256:
        if fname not in disp.spec_sha256:
            issues.append(f"dispositions do not pin a spec hash for {fname}")

    inh_sc = idx.inherited_scenarios
    inh_gates = idx.inherited_gates
    for rid in sorted(set(disp.entries) - defined):
        issues.append(f"disposition row for unknown requirement {rid}")

    for rid, entry in sorted(disp.entries.items()):
        where = f"disposition {rid}"
        for key in sorted(set(entry) - DISPOSITION_KEYS):
            issues.append(f"{where}: unknown key {key!r}")
        for key in sorted(set(entry) & _DISPOSITION_LIST_KEYS):
            if not isinstance(entry[key], list) or not all(
                isinstance(v, str) and v for v in entry[key]
            ):
                issues.append(f"{where}: {key} must be a list of strings")
        for key in ("owner", "stage", "implementation_status",
                    "qualification_status", "blocker", "note"):
            if key in entry and not isinstance(entry[key], str):
                issues.append(f"{where}: {key} must be a string")

        impl = entry.get("implementation_status")
        qual = entry.get("qualification_status")
        if impl is not None and impl not in IMPLEMENTATION_STATUSES:
            issues.append(f"{where}: invalid implementation_status {impl!r}")
        if qual is not None and qual not in QUALIFICATION_STATUSES:
            issues.append(f"{where}: invalid qualification_status {qual!r}")
        stage = entry.get("stage")
        if stage and stage not in STAGES:
            issues.append(f"{where}: invalid stage {stage!r}")

        for ev in entry.get("evidence_type") or []:
            if ev not in EVIDENCE_LABELS:
                issues.append(f"{where}: invalid evidence label {ev!r}")
        executed = entry.get("executed_evidence") or []
        evidence_type = entry.get("evidence_type") or []
        if executed and not evidence_type:
            issues.append(
                f"{where}: executed_evidence requires evidence_type "
                f"labels (V6-00.06)"
            )
        if qual in ("locally_measured", "qualified") and not executed:
            issues.append(
                f"{where}: {qual!r} requires named executed evidence — "
                f"a citation or plan is not execution (V6-00.06)"
            )
        if qual == "qualified" and impl != "implemented":
            issues.append(
                f"{where}: 'qualified' requires implementation_status "
                f"'implemented'"
            )
        if qual == "failed" and not (
            entry.get("blocker") or entry.get("note")
        ):
            issues.append(
                f"{where}: 'failed' requires a blocker or note naming "
                f"cause/next verification (V6-00.06)"
            )

        for sc in entry.get("scenarios") or []:
            if not _is_scenario_id(sc):
                issues.append(f"{where}: invalid scenario id {sc!r}")
            elif sc.startswith("F") and sc not in idx.scenarios:
                issues.append(f"{where}: unknown scenario {sc}")
            elif sc[0] in "ECD" and inh_sc and sc not in inh_sc:
                issues.append(f"{where}: unknown inherited scenario {sc}")
        for g in entry.get("gate") or []:
            if not _is_gate_id(g):
                issues.append(f"{where}: invalid gate id {g!r}")
            elif g.startswith("G6") and g not in idx.gates:
                issues.append(f"{where}: unknown gate {g}")
            elif g.startswith(("G4", "G5")) and inh_gates and g not in inh_gates:
                issues.append(f"{where}: unknown inherited gate {g}")

        if root:
            for surf in entry.get("code_surfaces") or []:
                if not os.path.exists(os.path.join(root, surf)):
                    issues.append(
                        f"{where}: code surface {surf} does not exist — "
                        f"stale mapping (V6-00.06)"
                    )
    return issues


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _render_carried(carried: Optional[CarriedTail]) -> dict:
    """The carried V5-tail block embedded in ledger_v6.json."""
    if carried is None:
        carried = CarriedTail(missing=True)
    return {
        "source": carried.source_path,
        "source_sha256": carried.sha256,
        "missing": carried.missing,
        "statuses": list(CARRIED_STATUSES),
        "total": len(carried.rows),
        "by_status": carried.by_status(),
        "rows": list(carried.rows),
    }


def render_ledger(idx: SpecIndex, registry: dict, disp: Dispositions,
                  carried: Optional[CarriedTail] = None,
                  sc_stage: Optional[dict] = None) -> dict:
    """The machine-readable ledger_v6.json document."""
    if sc_stage is None:
        sc_stage = infer_scenario_stages(idx)
    reqs = {}
    for rid, r in registry.items():
        reqs[rid] = {
            "section": f"V6-{r['section']:02d}",
            "subsection": (
                f"{r['section']:02d}.{r['subsection']}"
                if r["subsection"] >= 0 else ""
            ),
            "section_title": r["section_title"],
            "spec": r["spec"],
            "spec_line": r["line"],
            "capability": r["capability"],
            "profile": r["profile"],
            "owner": r["owner"],
            "stage": r["stage"],
            "stage_source": r["stage_source"],
            "code_surfaces": r["code_surfaces"],
            "scenarios": r["scenarios"],
            "spec_scenarios": r["spec_scenarios"],
            "gate": r["gate"],
            "envelopes": r["envelopes"],
            "inherited_requirements": r["inherited_requirements"],
            "inherited_gates": r["inherited_gates"],
            "inherited_scenarios": r["inherited_scenarios"],
            "evidence_type": r["evidence_type"],
            "executed_evidence": r["executed_evidence"],
            "implementation_status": r["implementation_status"],
            "qualification_status": r["qualification_status"],
            "status": r["status"],
            "blocker": r["blocker"],
            "note": r["note"],
            "text": r["text"],
        }
    summary_status = {s: 0 for s in STATUSES}
    summary_stage = {s: 0 for s in STAGES}
    summary_stage[""] = 0
    summary_impl = {s: 0 for s in IMPLEMENTATION_STATUSES}
    summary_qual = {s: 0 for s in QUALIFICATION_STATUSES}
    for r in registry.values():
        summary_status[r["status"]] += 1
        summary_stage[r["stage"]] += 1
        summary_impl[r["implementation_status"]] += 1
        summary_qual[r["qualification_status"]] += 1

    scenarios = {}
    for sc_id, sc in sorted(idx.scenarios.items()):
        stage, stage_source = sc_stage.get(sc_id, ("", "unbound"))
        scenarios[sc_id] = {
            "spec": sc.spec,
            "spec_line": sc.line,
            "text": sc.text,
            "stage": stage,
            "stage_source": stage_source,
            "envelopes": list(sc.envelopes),
            "bound_gates": sorted(
                g.gate_id for g in idx.gates.values()
                if sc_id in g.scenarios
            ),
            "status": "not_run",
            "anchored_requirements": sorted(
                rid for rid, r in registry.items()
                if sc_id in r["scenarios"]
            ),
        }
    gates = {}
    for gid, g in sorted(idx.gates.items()):
        gates[gid] = {
            "spec": g.spec,
            "spec_line": g.line,
            "name": g.name,
            "permits": g.permits,
            "scenarios": list(g.scenarios),
            "envelopes": list(g.envelopes),
            "status": "not_run",
            "bound_requirements": sorted(
                rid for rid, r in registry.items() if gid in r["gate"]
            ),
        }
    stages = {}
    for sid, st in sorted(idx.stages.items()):
        stages[sid] = {
            "spec": st.spec,
            "spec_line": st.line,
            "main_work": st.main_work,
            "exit": st.exit_text,
            "exit_gates": list(st.exit_gates),
            "exit_scenarios": list(st.exit_scenarios),
            "exit_envelopes": list(st.exit_envelopes),
        }
    carried_block = _render_carried(carried)
    return {
        "schema": 1,
        "generated_by": "tools/gen_v6_ledger.py",
        "specs": {SPEC_FILE: {"sha256": idx.spec_sha256.get(SPEC_FILE, "")}},
        "carried": carried_block,
        "statuses": list(STATUSES),
        "implementation_statuses": list(IMPLEMENTATION_STATUSES),
        "qualification_statuses": list(QUALIFICATION_STATUSES),
        "evidence_labels": list(EVIDENCE_LABELS),
        "stages": list(STAGES),
        "summary": {
            "total": len(registry),
            "by_status": summary_status,
            "by_implementation_status": summary_impl,
            "by_qualification_status": summary_qual,
            "by_stage": {k: v for k, v in summary_stage.items()},
            "requirements_with_scenarios": sum(
                1 for r in registry.values() if r["scenarios"]
            ),
            "requirements_with_gates": sum(
                1 for r in registry.values() if r["gate"]
            ),
            "disposition_overrides": len(disp.entries),
            "scenarios_total": len(idx.scenarios),
            "gates_total": len(idx.gates),
            "carried_total": carried_block["total"],
            "carried_by_status": carried_block["by_status"],
        },
        "requirements": reqs,
        "scenarios": scenarios,
        "gates": gates,
        "stages_defined": stages,
    }


_MARK = {
    "planned": " ",
    "in_progress": "~",
    "implemented_unmeasured": "u",
    "locally_measured": "m",
    "qualified": "x",
    "failed": "!",
    "not_applicable": "n",
}


def render_summary(idx: SpecIndex, registry: dict, disp: Dispositions,
                   carried: Optional[CarriedTail] = None,
                   sc_stage: Optional[dict] = None) -> str:
    """eval/v6/summary.md — the human-facing rollup (not a root doc)."""
    led = render_ledger(idx, registry, disp, carried=carried,
                        sc_stage=sc_stage)
    s = led["summary"]
    c = led["carried"]
    out = [
        "# V6 Requirements Ledger — Summary",
        "",
        "Generated from `SPEC_V6.md` by `tools/gen_v6_ledger.py`.",
        "Do not hand-edit: curate `eval/v6/dispositions_v6.json` and",
        "re-run with `--write`. Sibling machine ledger:",
        "`eval/v6/ledger_v6.json`.",
        "",
        "Status model (V6-00.06): `implementation_status` and",
        "`qualification_status` are separate axes; `status` is the",
        "derived rollup — `[x]` qualified, `[m]` locally_measured",
        "(implemented and measured here, still unqualified), `[u]`",
        "implemented_unmeasured, `[~]` in_progress, `[ ]` planned,",
        "`[!]` failed, `[n]` not_applicable. `locally_measured` and",
        "`qualified` require named executed evidence; nothing here",
        "infers status from counts.",
        "",
        "## Totals",
        "",
        f"- requirements: {s['total']}",
    ]
    for st in STATUSES:
        out.append(f"- status `{st}`: {s['by_status'][st]}")
    for st in QUALIFICATION_STATUSES:
        out.append(
            f"- qualification `{st}`: {s['by_qualification_status'][st]}"
        )
    carried_note = ""
    if c["missing"]:
        carried_note = " — source missing"
    out += [
        f"- scenarios: {s['scenarios_total']} (F01–F32), all `not_run`",
        f"- gates: {s['gates_total']} (G6-00–G6-08), all `not_run`",
        f"- carried V5 tail (V6-04.02/03): {c['total']} rows"
        f" (planned={c['by_status']['planned']}"
        f" implemented_unmeasured={c['by_status']['implemented_unmeasured']})"
        f"{carried_note}",
        f"- requirements with ≥1 applicable scenario: "
        f"{s['requirements_with_scenarios']}",
        f"- requirements with a bound gate: {s['requirements_with_gates']}",
        f"- disposition overrides: {s['disposition_overrides']}",
        "",
        "## By section",
        "",
        "| Section | Title | Reqs | Stage | Statuses |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_sec: dict = {}
    for rid, r in registry.items():
        by_sec.setdefault(r["section"], []).append(r)
    for sec in sorted(by_sec):
        rows = by_sec[sec]
        counts: dict = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        statuses = " ".join(
            f"{k}={v}" for k, v in sorted(counts.items())
        )
        sec_stages = sorted(
            {r["stage"] for r in rows if r["stage"]}, key=STAGES.index
        )
        stage = "/".join(sec_stages) if sec_stages else "—"
        title = idx.section_titles.get(sec, "")
        out.append(
            f"| §{sec:02d} | {title} | {len(rows)} | {stage} | {statuses} |"
        )
    out += [
        "",
        "## By phase",
        "",
        "| Phase | Reqs | Exit scenarios | Exit gates |",
        "| --- | --- | --- | --- |",
    ]
    for st in STAGES:
        rows = [r for r in registry.values() if r["stage"] == st]
        if st in idx.stages:
            sd = idx.stages[st]
            exit_sc = ", ".join(sd.exit_scenarios)
            exit_g = ", ".join(sd.exit_gates)
        else:
            exit_sc = exit_g = ""
        out.append(f"| {st} | {len(rows)} | {exit_sc} | {exit_g} |")
    unbound = sum(1 for r in registry.values() if not r["stage"])
    out.append(f"| (unbound) | {unbound} | | |")
    out += [
        "",
        "## Gates (G6-00–G6-08)",
        "",
        "| Gate | Name | Scenarios named | Status | Bound reqs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for gid, g in sorted(led["gates"].items()):
        out.append(
            f"| {gid} | {g['name']} | {', '.join(g['scenarios'])} | "
            f"{g['status']} | {len(g['bound_requirements'])} |"
        )
    out += [
        "",
        "## Scenarios (F01–F32)",
        "",
        "| Scenario | Phase | Bound gates | Anchored reqs | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for sc_id, sc in sorted(led["scenarios"].items()):
        out.append(
            f"| {sc_id} | {sc['stage'] or '—'} | "
            f"{', '.join(sc['bound_gates'])} | "
            f"{len(sc['anchored_requirements'])} | {sc['status']} |"
        )
    out += [
        "",
        "## Carried V5 tail (V6-04.02/03)",
        "",
    ]
    if c["missing"]:
        out.append(
            f"`{c['source']}` is missing — the V5 tail cannot be "
            "carried (V6-04.02/03)."
        )
    else:
        out += [
            f"{c['total']} unfinished V5 rows carried from",
            f"`{c['source']}` (read-only; the V5 file is never",
            "edited). Each MUST be re-evidenced or explicitly",
            "re-deferred in this wave.",
            "",
            "| V5 id | Status | Qualification | Note |",
            "| --- | --- | --- | --- |",
        ]
        for row in c["rows"]:
            note = row["note"].replace("|", "\\|")
            out.append(
                f"| {row['id']} | {row['status']} | "
                f"{row['qualification_status']} | {note} |"
            )
    out += [
        "",
        "Phase-level anchors are the planning map (V6-00.06);",
        "qualification requires named assertions per requirement,",
        "recorded as `executed_evidence` in `dispositions_v6.json`.",
        "",
    ]
    return "\n".join(out)
