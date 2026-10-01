"""Executable-traceability ledger for SPEC_V5 (§27.2, V5-27.*).

The v5 ledger answers the same honesty question as v4 — which
requirement has *named executed evidence* behind it — under the V5
status model (V5-00.05, V5-27.05): implementation and qualification
are separate axes. An item may be implemented and locally measured
yet still unqualified; the rollup ``status`` is *derived*, never set
directly.

Pipeline (``tools/gen_v5_ledger.py`` drives it):

* ``load_specs`` parses SPEC_V5.md: every ``- V5-NN.MM:`` definition
  (``/MM`` shorthand continuations expanded), every ``| ENN |`` E-
  scenario row (§24, §36 — 96 scenarios), every ``| G5-NN ... |`` gate
  row (§25 — 15 gates), and every ``| PN — ... |`` stage row (§26 —
  P0..P6). Duplicate definitions and references to ids that were never
  defined are hard errors (V5-27.03).
* ``infer_*`` derives the planning map the spec itself declares:
  section-level scenario anchors (V5-27.02), the §26 stage a section is
  first exercised at, and the §25 gates a requirement's section/text
  binds. Inference provenance is recorded per row (``stage_source``).
* ``dispositions_v5.json`` is the curated overlay: per id it may set
  ``owner``, ``code_surfaces``, extra ``scenarios``/``gate``, evidence
  labels, named ``executed_evidence``, ``implementation_status``,
  ``qualification_status``, ``blocker`` and ``note``. The generator
  merges it but never rewrites human judgments (V5-27.04).
* ``validate`` enforces the V5-27.03 contract: no duplicate/unknown
  ids, no references to undefined requirements/scenarios/gates, no
  stale spec hash, no ``qualified``/``locally_measured`` without named
  executed evidence, no unknown overlay keys or stale code surfaces.

Honesty rule (V5-00.05): everything seeds ``planned``/``not_run``.
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

#: Implementation axis (V5-27.05): has the code been written?
IMPLEMENTATION_STATUSES = (
    "planned",
    "in_progress",
    "implemented",
    "not_applicable",
)

#: Qualification axis (V5-27.05): has executed evidence qualified it?
#: ``locally_measured`` = ran here, unqualified — distinct from both
#: ``not_run`` and ``qualified`` (V5-27.05 separates measurement,
#: qualification, and recommendation). ``deferred`` is the V6-added
#: tail disposition (SPEC_V6-04.02/04.03): an explicit re-deferral with
#: owner+reason — it derives no new rollup (the row stays visibly open
#: under its implementation axis) and needs no executed evidence.
QUALIFICATION_STATUSES = (
    "not_run",
    "locally_measured",
    "qualified",
    "failed",
    "not_applicable",
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

#: Evidence labels, V5-27.06.
EVIDENCE_LABELS = ("DOC", "CODE", "LOCAL", "BENCH", "INDEPENDENT")

#: Implementation stages, §26.
STAGES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6")

#: The spec this ledger covers, plus inherited specs consulted only to
#: resolve curated C/D scenario ids (read-only; never generated from).
SPEC_FILE = "SPEC_V5.md"
INHERITED_SPEC_FILES = ("SPEC_V4.md", "SPEC_V4_5.md")

#: Default locations (relative to repo root).
DEFAULT_DISPOSITIONS_PATH = os.path.join("eval", "v5", "dispositions_v5.json")
DEFAULT_JSON_PATH = os.path.join("eval", "v5", "ledger_v5.json")
DEFAULT_SUMMARY_PATH = os.path.join("eval", "v5", "summary.md")

#: Every row starts blocked on approval (V5-00.01) until a disposition
#: names a different unresolved blocker.
DEFAULT_BLOCKER = "v5 spec not yet approved — implementation unauthorized (V5-00.01)"

#: Sections no scenario or §26 stage row can pin down; resolved here and
#: marked ``stage_source="fallback"`` (or ``""`` when genuinely unbound —
#: §25 release-gate discipline applies per release level, not one stage).
SECTION_STAGE_FALLBACK = {
    0: "P0",   # authority/acceptance language — contract freeze
    1: "P0",   # product thesis — scope freeze
    2: "P0",   # audited starting point — reproduce gaps
    24: "P0",  # test strategy — P0 writes failing E-path tests
    25: "",    # release-gate discipline — bound per release level, §25.1
    26: "P0",  # the plan itself executes from contract freeze
    29: "P4",  # competitive strategy — exercised by §22/P4 comparisons
}

#: One capability label per spec section (deterministic section map;
#: unknown sections fall back to a slug of the section title).
SECTION_CAPABILITY = {
    0: "governance", 1: "product-thesis", 2: "audit",
    3: "contract-amendments", 4: "profiles", 5: "identity-bootstrap",
    6: "public-api", 7: "ingest", 8: "readiness",
    9: "worker-lifecycle", 10: "retrieval-default",
    11: "indexes-statistics", 12: "vector-acceleration",
    13: "packing-delivery", 14: "corrections-supersession",
    15: "inspect-forget", 16: "caching", 17: "neural-extras",
    18: "distribution", 19: "hosts-adapters", 20: "perf-envelopes",
    21: "quality-portfolio", 22: "competitive", 23: "statistics",
    24: "test-strategy", 25: "release-gates", 26: "implementation-plan",
    27: "traceability", 29: "competitive-strategy", 30: "memory-ops",
    31: "ranking", 32: "token-efficiency", 33: "perf-engineering",
    34: "multi-agent", 35: "learning",
}

# ---------------------------------------------------------------------------
# spec grammar
# ---------------------------------------------------------------------------

#: `- V5-NN.MM: text` — definition lines only.
_DEF_RE = re.compile(r"^-\s+V5-(\d{2})\.(\d{2}):\s*(.*)$")
#: `## NN. Title` (top-level sections; `###` subsections do not match).
_SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$")
#: Any `V5-NN.MM` mention incl. `/MM` and `/NN.MM` shorthand
#: continuations (`V5-13.02/13.05`).
_ANCHOR_RE = re.compile(r"\bV5-(\d{2})\.(\d{2})((?:/\d{2}(?:\.\d{2})?)*)")
_ANCHOR_PART_RE = re.compile(r"/(\d{2})(?:\.(\d{2}))?")
#: `| E01 | desc | §NN… | stage |` scenario rows (§24/§36).
_SCENARIO_ROW_RE = re.compile(r"^\|\s*(E\d{2})\s*\|(.*)\|\s*$")
#: `| G5-00 Name | evidence | claims |` gate rows (§25).
_GATE_ROW_RE = re.compile(r"^\|\s*(G5-\d{2})\s*([^|]*)\|(.*)\|\s*$")
#: `| P0 — Title | work | exit |` stage rows (§26).
_STAGE_ROW_RE = re.compile(r"^\|\s*(P[0-6])\s*[—-]\s*([^|]*)\|(.*)\|\s*$")
#: `§NN`, `§NN.MM`, `§NN–§MM`, `§NN.MM–MM.MM` (en dash). A range whose
#: endpoints carry subsections stays inside the first section.
_SECTION_REF_RE = re.compile(
    r"§(\d{2})(?:\.(\d{2}))?(?:\s*–\s*§?(\d{2})(?:\.(\d{2}))?)?"
)
#: Bare scenario ids and explicit `ENN–ENN` ranges (for references).
_EID_RE = re.compile(r"\bE(\d{2})\b")
_ERANGE_RE = re.compile(r"\bE(\d{2})`?\s*–\s*`?E(\d{2})")
#: `G5-NN` incl. `/NN` continuations (`G5-00/01/02`) and `–` ranges
#: (`G5-00`–`G5-14` denotes the whole registry).
_GATE_REF_RE = re.compile(r"\bG5-(\d{2})((?:/\d{2})*)")
_GATE_RANGE_RE = re.compile(r"\bG5-(\d{2})`?\s*–\s*`?G5-(\d{2})")
#: Inherited gate mentions (`G4-10`) — recorded, never satisfied here.
_INHERITED_GATE_RE = re.compile(r"\bG4-(\d{2})\b")
#: Explicit stage directives inside requirement text (`P0 MUST define`).
_STAGE_DIRECTIVE_RE = re.compile(r"\bP([0-6])(?=\s+MUST)")
#: Profile names the spec registers.
_PROFILE_RE = re.compile(r"\b(local_memory|local_rules|embedded)\b")
#: `| C01 |` / `| D07 |` first-cell rows in the inherited specs.
_CD_ROW_RE = re.compile(r"^\|\s*([CD]\d{2})\s*\|", re.M)


def _expand_anchor(m: re.Match) -> list:
    """Expand one V5 anchor match into every id it denotes.

    `V5-13.02/13.05` -> [`V5-13.02`, `V5-13.05`]; a bare `/MM` keeps the
    base section.
    """
    sec, sub, tail = m.groups()
    ids = [f"V5-{sec}.{sub}"]
    for seg, sub2 in _ANCHOR_PART_RE.findall(tail or ""):
        ids.append(f"V5-{seg}.{sub2}" if sub2 else f"V5-{sec}.{seg}")
    return ids


def anchors_in(text: str) -> list:
    """All `V5-NN.MM` ids mentioned in ``text`` (shorthand expanded)."""
    out = []
    for m in _ANCHOR_RE.finditer(text):
        out.extend(_expand_anchor(m))
    return out


def eids_in(text: str) -> list:
    """All E-scenario ids denoted by ``text`` (ranges expanded)."""
    out = {f"E{n}" for n in _EID_RE.findall(text)}
    for a, b in _ERANGE_RE.findall(text):
        out.update(f"E{n:02d}" for n in range(int(a), int(b) + 1))
    return sorted(out)


def _gate_mentions(text: str) -> list:
    """`G5-NN` singles + `/NN` continuations, excluding range members.

    `G5-00/01/02` -> all three; `G5-00`–`G5-14` -> [] (a range is a
    namespace reference, not a per-gate binding).
    """
    spans = [m.span() for m in _GATE_RANGE_RE.finditer(text)]
    out = set()
    for m in _GATE_REF_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        out.add(f"G5-{m.group(1)}")
        out.update(f"G5-{n}" for n in re.findall(r"/(\d{2})", m.group(2) or ""))
    return sorted(out)


def gates_in(text: str) -> list:
    """All `G5-NN` ids denoted by ``text`` — singles, continuations,
    and en-dash ranges (`G5-00`–`G5-14`) fully expanded."""
    out = set(_gate_mentions(text))
    for a, b in _GATE_RANGE_RE.findall(text):
        out.update(f"G5-{n:02d}" for n in range(int(a), int(b) + 1))
    return sorted(out)


def sections_in(text: str) -> list:
    """Top-level spec sections a `§` reference list denotes.

    `§10–§12` -> [10, 11, 12]; `§30.1, §30.6` -> [30];
    `§05.12–05.13` -> [5] (a subsection range stays in its section).
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
    """Stage tokens in a cell: `P2/P3` -> both; `All` -> none."""
    return sorted(
        {f"P{d}" for d in re.findall(r"\bP([0-6])\b", text)},
        key=STAGES.index,
    )


class LedgerError(Exception):
    """Raised for unparseable inputs; validation failures are reported
    as issue lists instead (see :func:`validate`)."""


# ---------------------------------------------------------------------------
# spec parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequirementDef:
    """One `- V5-NN.MM:` definition line."""

    req_id: str
    spec: str           # spec file name, e.g. "SPEC_V5.md"
    section: int        # enclosing `## NN.` section number
    id_section: int     # the NN inside the id itself
    section_title: str
    text: str
    line: int


@dataclass(frozen=True)
class ScenarioDef:
    """One §24/§36 `| ENN |` acceptance-scenario row."""

    sc_id: str          # "E07"
    spec: str
    text: str
    sections: tuple     # primary-section numbers the row names
    stage: str          # raw stage cell: "P1", "P2/P3", "All"
    line: int


@dataclass(frozen=True)
class GateDef:
    """One §25 `| G5-NN name |` release-gate row."""

    gate_id: str        # "G5-00"
    spec: str
    name: str
    required_evidence: str
    claims_permitted: str
    sections: tuple     # §-refs inside the evidence cell (e.g. G5-13 -> §30)
    line: int


@dataclass(frozen=True)
class StageDef:
    """One §26 `| PN — title |` stage row."""

    stage_id: str       # "P0"
    spec: str
    title: str
    main_work: str
    exit_text: str
    sections: tuple     # §-refs inside the work cell
    exit_gates: tuple   # G5-ids named in the exit cell
    line: int


@dataclass
class SpecIndex:
    """Merged parse of SPEC_V5.md (+ inherited C/D id sets)."""

    requirements: dict = field(default_factory=dict)   # id -> RequirementDef
    scenarios: dict = field(default_factory=dict)      # E-id -> ScenarioDef
    gates: dict = field(default_factory=dict)          # G5-id -> GateDef
    stages: dict = field(default_factory=dict)         # P-id -> StageDef
    section_titles: dict = field(default_factory=dict)  # int -> title
    references: dict = field(default_factory=dict)     # id -> [spec:line]
    duplicates: list = field(default_factory=list)     # [str]
    spec_sha256: dict = field(default_factory=dict)    # file -> hex
    cd_scenarios: set = field(default_factory=set)     # C/D ids from v4 specs

    def scenario_sections(self, sc_id: str) -> tuple:
        sc = self.scenarios.get(sc_id)
        return sc.sections if sc else ()


def _record_refs(idx: SpecIndex, text: str, where: str,
                 skip_req: Optional[str] = None) -> None:
    """Register every id ``text`` mentions (for undefined-ref checks)."""
    for rid in anchors_in(text):
        if rid != skip_req:
            idx.references.setdefault(rid, []).append(where)
    for sc in eids_in(text):
        idx.references.setdefault(sc, []).append(where)
    for g in gates_in(text):
        idx.references.setdefault(g, []).append(where)


def _parse_spec_text(raw: str, fname: str, idx: SpecIndex) -> None:
    """Parse one spec's text into ``idx`` (testable without files)."""
    idx.spec_sha256[fname] = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    section = -1
    section_title = ""
    for n, line in enumerate(raw.splitlines(), 1):
        m = _SECTION_RE.match(line)
        if m:
            section = int(m.group(1))
            section_title = m.group(2).strip()
            idx.section_titles[section] = section_title
            continue
        m = _DEF_RE.match(line)
        if m:
            rid = f"V5-{m.group(1)}.{m.group(2)}"
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
            sects = tuple(sections_in(cells[1])) if len(cells) > 1 else ()
            stage = cells[2] if len(cells) > 2 else ""
            where = f"{fname}:{n}"
            if sc_id in idx.scenarios:
                prev = idx.scenarios[sc_id]
                idx.duplicates.append(
                    f"scenario {sc_id} defined twice: "
                    f"{prev.spec}:{prev.line} and {where}"
                )
            else:
                idx.scenarios[sc_id] = ScenarioDef(
                    sc_id=sc_id, spec=fname, text=desc, sections=sects,
                    stage=stage, line=n,
                )
            _record_refs(idx, rest, where)
            continue
        gm = _GATE_ROW_RE.match(line)
        if gm:
            gate_id, name, rest = gm.group(1), gm.group(2).strip(), gm.group(3)
            cells = [c.strip() for c in rest.split("|")]
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
                    required_evidence=cells[0] if len(cells) > 0 else "",
                    claims_permitted=cells[1] if len(cells) > 1 else "",
                    sections=tuple(sections_in(rest)),
                    line=n,
                )
            _record_refs(idx, rest, where)
            continue
        pm = _STAGE_ROW_RE.match(line)
        if pm:
            stage_id, title, rest = pm.group(1), pm.group(2).strip(), pm.group(3)
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
                    stage_id=stage_id, spec=fname, title=title,
                    main_work=work, exit_text=exit_text,
                    sections=tuple(sections_in(work)),
                    exit_gates=tuple(gates_in(exit_text)),
                    line=n,
                )
            _record_refs(idx, rest, where)
            continue
        _record_refs(idx, line, f"{fname}:{n}")


def _scan_cd_scenarios(raw: str) -> set:
    """`C`/`D` scenario ids defined by an inherited spec file."""
    return {m.group(1) for m in _CD_ROW_RE.finditer(raw)}


def load_specs(root: str, spec_file: str = SPEC_FILE) -> SpecIndex:
    """Parse SPEC_V5.md; also collect inherited C/D ids if present."""
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
                idx.cd_scenarios |= _scan_cd_scenarios(f.read())
        except OSError:
            continue  # inherited specs absent: C/D validation is skipped
    return idx


# ---------------------------------------------------------------------------
# inference — the planning map the spec itself declares (V5-27.02)
# ---------------------------------------------------------------------------


def infer_section_stages(idx: SpecIndex) -> dict:
    """section -> earliest stage exercising it, with provenance.

    Provenance precedence (first hit wins per requirement, but the map
    is per section): ``text`` directives are resolved per requirement in
    :func:`infer_requirement_stage`; here we build the section map from

    * ``stage_table``: §-references inside §26 stage rows' work cells
      (e.g. P1 names §30, P2 names §31/§33);
    * ``scenario_anchor``: the minimum stage across E-scenarios whose
      primary-section cell names the section (`P2/P3` counts as P2;
      `All` carries no stage signal);
    * ``fallback``: :data:`SECTION_STAGE_FALLBACK` for sections no
      scenario or stage row pins (§00–§02, §24–§26, §29).
    """
    sec_stage: dict = {}      # section -> (min_stage, source)
    candidates: dict = {}     # section -> {stage -> {source}}

    def offer(sec: int, stage: str, source: str) -> None:
        if stage not in STAGES:
            return
        candidates.setdefault(sec, {}).setdefault(stage, set()).add(source)

    for st in idx.stages.values():
        for sec in st.sections:
            offer(sec, st.stage_id, "stage_table")
    for sc in idx.scenarios.values():
        for st in _stages_in(sc.stage):
            for sec in sc.sections:
                offer(sec, st, "scenario_anchor")
    for sec, stages in candidates.items():
        best = min(stages, key=STAGES.index)
        srcs = stages[best]
        src = "stage_table" if "stage_table" in srcs else "scenario_anchor"
        sec_stage[sec] = (best, src)
    for sec, st in SECTION_STAGE_FALLBACK.items():
        sec_stage.setdefault(sec, (st, "fallback"))
    return sec_stage


def infer_requirement_stage(idx: SpecIndex, req: RequirementDef,
                            sec_stage: dict) -> tuple:
    """(stage, source) for one requirement.

    An explicit `PN MUST` directive inside the requirement text wins
    (V5-04.01 assigns its mapping work to P0); otherwise the section's
    inferred stage applies; `""` means no stage is inferable.
    """
    m = _STAGE_DIRECTIVE_RE.search(req.text)
    if m:
        return f"P{m.group(1)}", "text"
    return sec_stage.get(req.section, ("", "unbound"))


def infer_requirement_gates(idx: SpecIndex, req: RequirementDef) -> list:
    """G5 gates a requirement binds: explicit `G5-NN` text mentions plus
    gates whose §25 evidence cell names the requirement's section
    (G5-13 -> §30, G5-14 -> §32). Range mentions (`G5-00`–`G5-14`
    declaring the namespace) do not bind. Stage exit gates live on the
    stage record itself — they are stage obligations, not per-row
    bindings."""
    out = set(_gate_mentions(req.text))
    for g in idx.gates.values():
        if req.section in g.sections:
            out.add(g.gate_id)
    return sorted(out)


def infer_requirement_scenarios(idx: SpecIndex, req: RequirementDef) -> list:
    """E-scenarios whose primary sections include the requirement's
    section — the spec's section-level planning anchors (V5-27.02)."""
    return sorted(
        sc.sc_id for sc in idx.scenarios.values() if req.section in sc.sections
    )


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
    """Derived ``status`` — never set directly (V5-27.05)."""
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
# dispositions overlay (eval/v5/dispositions_v5.json) — curated, JSON
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
    """Parsed eval/v5/dispositions_v5.json."""

    spec_sha256: dict = field(default_factory=dict)
    entries: dict = field(default_factory=dict)   # req_id -> dict of overrides
    source_path: str = DEFAULT_DISPOSITIONS_PATH


def _no_dup_object(pairs):
    """object_pairs_hook rejecting duplicate JSON keys (V5-27.03)."""
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
    """Load dispositions_v5.json; structural errors raise LedgerError."""
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
            "Curated overlay for the V5 requirements ledger "
            "(SPEC_V5 §27.2). Keys per requirement: owner, profile, "
            "stage, code_surfaces, scenarios (E/C/D ids added beyond the "
            "spec's section anchors), gate, evidence_type (DOC/CODE/"
            "LOCAL/BENCH/INDEPENDENT), executed_evidence (named executed "
            "checks/runs — required for locally_measured/qualified), "
            "implementation_status, qualification_status, blocker, note. "
            "spec_sha256 is refreshed by `tools/gen_v5_ledger.py --write`."
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
    sec_stage = infer_section_stages(idx)
    registry = {}
    for rid in sorted(idx.requirements):
        d = idx.requirements[rid]
        entry = disp.entries.get(rid) or {}
        stage, stage_source = infer_requirement_stage(idx, d, sec_stage)
        if entry.get("stage"):
            stage, stage_source = entry["stage"], "disposition"
        spec_scenarios = infer_requirement_scenarios(idx, d)
        scenarios = sorted(
            set(spec_scenarios) | set(entry.get("scenarios") or [])
        )
        gates = sorted(
            set(infer_requirement_gates(idx, d)) | set(entry.get("gate") or [])
        )
        profiles = entry.get("profile")
        if profiles is None:
            profiles = profiles_in(d.text) or ["*"]
        impl = entry.get("implementation_status") or "planned"
        qual = entry.get("qualification_status") or "not_run"
        registry[rid] = {
            "id": rid,
            "spec": d.spec,
            "section": d.section,
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
            "inherited_gates": sorted(set(
                f"G4-{n}" for n in _INHERITED_GATE_RE.findall(d.text)
            )),
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
# validation (V5-27.03)
# ---------------------------------------------------------------------------


def _is_scenario_id(s: str) -> bool:
    return bool(re.match(r"^[ECD]\d{2}$", s))


def _is_gate_id(s: str) -> bool:
    return bool(re.match(r"^G[45]-\d{2}$", s))


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
            if rid.startswith("E") and rid not in idx.scenarios:
                issues.append(
                    f"spec reference to undefined scenario {rid} ({refs[0]})"
                )
            continue
        if _is_gate_id(rid):
            if rid not in idx.gates:
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
    # a V5-30.* row under §31 is a spec defect, not a parse difference
    for rid, d in sorted(idx.requirements.items()):
        if d.id_section != d.section:
            issues.append(
                f"{rid} is defined under §{d.section:02d} "
                f"({d.spec}:{d.line}) — id/section mismatch"
            )

    # scenario rows must name real sections
    known_sections = set(idx.section_titles)
    for sc in sorted(idx.scenarios.values(), key=lambda s: s.sc_id):
        if not sc.sections:
            issues.append(
                f"scenario {sc.sc_id} names no primary section "
                f"({sc.spec}:{sc.line})"
            )
        for sec in sc.sections:
            if sec not in known_sections:
                issues.append(
                    f"scenario {sc.sc_id} cites unknown section "
                    f"§{sec:02d} ({sc.spec}:{sc.line})"
                )
        if sc.stage and sc.stage != "All" and not _stages_in(sc.stage):
            issues.append(
                f"scenario {sc.sc_id} has unparsable stage {sc.stage!r}"
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
                f"`tools/gen_v5_ledger.py --write` after reviewing the "
                f"spec diff"
            )
    for fname in idx.spec_sha256:
        if fname not in disp.spec_sha256:
            issues.append(f"dispositions do not pin a spec hash for {fname}")

    cd_known = idx.cd_scenarios
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
                f"labels (V5-27.06)"
            )
        if qual in ("locally_measured", "qualified") and not executed:
            issues.append(
                f"{where}: {qual!r} requires named executed evidence — "
                f"a citation or plan is not execution (V5-27.03/27.05)"
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
                f"cause/next verification (V5-25.03)"
            )

        for sc in entry.get("scenarios") or []:
            if not _is_scenario_id(sc):
                issues.append(f"{where}: invalid scenario id {sc!r}")
            elif sc.startswith("E") and sc not in idx.scenarios:
                issues.append(f"{where}: unknown scenario {sc}")
            elif sc[0] in "CD" and cd_known and sc not in cd_known:
                issues.append(f"{where}: unknown inherited scenario {sc}")
        for g in entry.get("gate") or []:
            if not _is_gate_id(g):
                issues.append(f"{where}: invalid gate id {g!r}")
            elif g.startswith("G5") and g not in idx.gates:
                issues.append(f"{where}: unknown gate {g}")

        if root:
            for surf in entry.get("code_surfaces") or []:
                if not os.path.exists(os.path.join(root, surf)):
                    issues.append(
                        f"{where}: code surface {surf} does not exist — "
                        f"stale mapping (V5-27.03)"
                    )
    return issues


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_ledger(idx: SpecIndex, registry: dict, disp: Dispositions) -> dict:
    """The machine-readable ledger_v5.json document."""
    reqs = {}
    for rid, r in registry.items():
        reqs[rid] = {
            "section": f"V5-{r['section']:02d}",
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
            "inherited_gates": r["inherited_gates"],
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
        scenarios[sc_id] = {
            "spec": sc.spec,
            "spec_line": sc.line,
            "text": sc.text,
            "sections": [f"§{s:02d}" for s in sc.sections],
            "stage": sc.stage,
            "status": "not_run",
            "anchored_requirements": sorted(
                rid for rid, d in idx.requirements.items()
                if d.section in sc.sections
            ),
        }
    gates = {}
    for gid, g in sorted(idx.gates.items()):
        gates[gid] = {
            "spec": g.spec,
            "spec_line": g.line,
            "name": g.name,
            "required_evidence": g.required_evidence,
            "claims_permitted": g.claims_permitted,
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
            "title": st.title,
            "main_work": st.main_work,
            "exit": st.exit_text,
            "exit_gates": list(st.exit_gates),
            "sections": [f"§{s:02d}" for s in st.sections],
        }
    return {
        "schema": 1,
        "generated_by": "tools/gen_v5_ledger.py",
        "specs": {SPEC_FILE: {"sha256": idx.spec_sha256.get(SPEC_FILE, "")}},
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


def render_summary(idx: SpecIndex, registry: dict, disp: Dispositions) -> str:
    """eval/v5/summary.md — the human-facing rollup (not a root doc)."""
    led = render_ledger(idx, registry, disp)
    s = led["summary"]
    out = [
        "# V5 Requirements Ledger — Summary",
        "",
        "Generated from `SPEC_V5.md` by `tools/gen_v5_ledger.py`.",
        "Do not hand-edit: curate `eval/v5/dispositions_v5.json` and",
        "re-run with `--write`. Sibling machine ledger:",
        "`eval/v5/ledger_v5.json`.",
        "",
        "Status model (V5-00.05/27.05): `implementation_status` and",
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
    out += [
        f"- scenarios: {s['scenarios_total']} (E01–E96), all `not_run`",
        f"- gates: {s['gates_total']} (G5-00–G5-14), all `not_run`",
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
        "## By stage",
        "",
        "| Stage | Reqs | Exit gates |",
        "| --- | --- | --- |",
    ]
    for st in STAGES:
        rows = [r for r in registry.values() if r["stage"] == st]
        exit_gates = ", ".join(
            idx.stages[st].exit_gates) if st in idx.stages else ""
        out.append(f"| {st} | {len(rows)} | {exit_gates} |")
    unbound = sum(1 for r in registry.values() if not r["stage"])
    out.append(f"| (unbound) | {unbound} | |")
    out += [
        "",
        "## Gates",
        "",
        "| Gate | Name | Status | Bound reqs |",
        "| --- | --- | --- | --- |",
    ]
    for gid, g in sorted(led["gates"].items()):
        out.append(
            f"| {gid} | {g['name']} | {g['status']} | "
            f"{len(g['bound_requirements'])} |"
        )
    out += [
        "",
        "## Scenarios (E01–E96)",
        "",
        "| Scenario | Stage | Primary sections | Anchored reqs | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for sc_id, sc in sorted(led["scenarios"].items()):
        out.append(
            f"| {sc_id} | {sc['stage'] or '—'} | "
            f"{', '.join(sc['sections'])} | "
            f"{len(sc['anchored_requirements'])} | {sc['status']} |"
        )
    out += [
        "",
        "Section-level anchors are the planning map (V5-27.02);",
        "qualification requires named assertions per requirement,",
        "recorded as `executed_evidence` in `dispositions_v5.json`.",
        "",
    ]
    return "\n".join(out)
