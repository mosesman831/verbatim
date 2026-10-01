"""V4 executable-traceability ledger (SPEC_V4 §61 + §02 evidence labels).

The package exposes the library half of ``tools/gen_v4_ledger.py``:
spec parsing (``load_specs``), test-tree anchor discovery
(``discover_evidence``), requirement-map loading/validation
(``load_map`` / ``validate``), and the rendered artifacts
(``render_markdown`` -> ``REQUIREMENTS_V4.md``, ``render_ledger`` ->
``eval/v4/ledger.json``, ``render_map_yaml`` -> the map itself).
"""

from eval.v4.ledger import (
    DEFAULT_JSON_PATH,
    DEFAULT_MAP_PATH,
    DEFAULT_MD_PATH,
    EVIDENCE_LABELS,
    SPEC_FILES,
    STATUSES,
    STRONG_KINDS,
    Discovery,
    LedgerError,
    MapEntry,
    RequirementDef,
    RequirementMap,
    ScenarioDef,
    SpecIndex,
    anchors_in,
    build_registry,
    collect_pytest_nodes,
    default_status,
    discover_evidence,
    load_map,
    load_specs,
    render_ledger,
    render_map_yaml,
    render_markdown,
    scenarios_in,
    seed_status,
    validate,
)

__all__ = [
    "DEFAULT_JSON_PATH",
    "DEFAULT_MAP_PATH",
    "DEFAULT_MD_PATH",
    "EVIDENCE_LABELS",
    "SPEC_FILES",
    "STATUSES",
    "STRONG_KINDS",
    "Discovery",
    "LedgerError",
    "MapEntry",
    "RequirementDef",
    "RequirementMap",
    "ScenarioDef",
    "SpecIndex",
    "anchors_in",
    "build_registry",
    "collect_pytest_nodes",
    "default_status",
    "discover_evidence",
    "load_map",
    "load_specs",
    "render_ledger",
    "render_map_yaml",
    "render_markdown",
    "scenarios_in",
    "seed_status",
    "validate",
]
