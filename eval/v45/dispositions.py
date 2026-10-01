"""M5 disposition table (SPEC_V4_5 §16; V45-16.01, D21).

``dispositions.json`` is generated, never hand-edited — every
I-experiment (I1–I6) and X-mechanism (X1–X8) has exactly one row
carrying:

* ``disposition``: ``adopt`` | ``reject`` | ``defer`` | ``research_only``
  — the §16 stop-condition verdict;
* ``recommended``: whether the mechanism may enter recommended
  routing/defaults — the D21 consumer flag. A failed, deferred, or
  research-only mechanism is NEVER recommended, regardless of what a
  competitor advertises (V45-09.09);
* ``evidence``: pytest node ids / report paths backing the row;
* ``measured``: raw paired outcomes and resource costs where they exist,
  else the honest string ``"unmeasured"`` — file existence is not
  evidence (V45-17.01);
* ``note``: what keeps the row out of recommended routing, or the gate
  it still owes.

Consumers read :func:`is_recommended` / :func:`recommended_mechanisms`
— routing and capability surfaces must consult the flag, never infer
recommendation from the presence of code (D21).

CLI::

    python -m eval.v45.dispositions            # write dispositions.json
    python -m eval.v45.dispositions --check    # verify it is current
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

DISPOSITIONS = ("adopt", "reject", "defer", "research_only")

MECHANISM_IDS = (
    "I1", "I2", "I3", "I4", "I5", "I6",
    "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8",
)

DISPOSITIONS_PATH = os.path.join(os.path.dirname(__file__),
                                 "dispositions.json")


def generate_rows() -> list[dict[str, Any]]:
    """The canonical M5 disposition table (V45-16.01).

    Honesty rules applied row by row:

    * I1–I6 are ``defer`` — their runners belong to parallel owners and
      no paired ablation outcome is published in this table yet; they
      stay visible and out of recommended routing (D21, V45-09.09).
    * X-mechanisms that are *safety/honesty guards* are ``adopt`` — their
      acceptance is regression-verified behavior, not an ablation; the
      rows name the verifying test nodes.
    * X8 stays ``research_only``: the §16 stop condition keeps HRR-class
      composition in research until it beats exact intersection plus
      hybrid retrieval — no such paired outcome exists, so it cannot be
      recommended.
    """
    return [
        # ---------------- I-experiments (§03–§08) ----------------------
        {
            "id": "I1",
            "mechanism": "counterevidence-first manifest packing",
            "disposition": "defer",
            "recommended": False,
            "evidence": ["eval/v45/i1_manifest.py"],
            "measured": "unmeasured",
            "note": "runner exists (parallel worker); paired false-current "
                    "ablation not yet published in this table — "
                    "implementation is not acceptance (V45-17.01). Keep "
                    "extractive packs.",
        },
        {
            "id": "I2",
            "mechanism": "localized repair under impact plans",
            "disposition": "defer",
            "recommended": False,
            "evidence": [],
            "measured": "unmeasured",
            "note": "paired recompute-cost evidence not yet published; "
                    "keep generation rebuilds.",
        },
        {
            "id": "I3",
            "mechanism": "progressive disclosure with sufficiency "
                         "receipts",
            "disposition": "defer",
            "recommended": False,
            "evidence": [],
            "measured": "unmeasured",
            "note": "token-cut-at-non-inferior-utility evidence not yet "
                    "published; keep fixed packs.",
        },
        {
            "id": "I4",
            "mechanism": "failure-aware procedure reuse",
            "disposition": "defer",
            "recommended": False,
            "evidence": [],
            "measured": "unmeasured",
            "note": "held-out negative-transfer evidence not yet "
                    "published; keep non-recommendation of stale "
                    "procedures.",
        },
        {
            "id": "I5",
            "mechanism": "branch previews with governed apply",
            "disposition": "defer",
            "recommended": False,
            "evidence": [],
            "measured": "unmeasured",
            "note": "safe-apply-forecast evidence not yet published; "
                    "keep offline dry-run tools.",
        },
        {
            "id": "I6",
            "mechanism": "utility-budgeted refresh scheduling",
            "disposition": "defer",
            "recommended": False,
            "evidence": [],
            "measured": "unmeasured",
            "note": "lower-maintenance-cost evidence not yet published; "
                    "keep explicit refresh.",
        },
        # ---------------- X-mechanisms (§09) ---------------------------
        {
            "id": "X1",
            "mechanism": "exact-then-semantic bake-off harness",
            "disposition": "adopt",
            "recommended": True,
            "evidence": [
                "tests/eval/test_v45_bakeoff.py::test_registry_names_every_comparator",
                "tests/eval/test_v45_bakeoff.py::test_weakened_comparator_cannot_lose",
                "tests/eval/test_v45_bakeoff.py::test_native_controlled_tracks_separate",
                "tests/eval/test_v45_bakeoff.py::test_total_cost_categories",
            ],
            "measured": {
                "local_arms": 6,
                "registry_rows": "full parent set",
                "cost_categories": 8,
            },
            "note": "instrumentation — pins per V4-54.01, refuses "
                    "weakened-comparator wins, never counts unavailable "
                    "rows as defeated.",
        },
        {
            "id": "X2",
            "mechanism": "entity timeline with explicit invalidation",
            "disposition": "adopt",
            "recommended": True,
            "evidence": [
                "tests/retrieval/v3/test_entity_timeline.py::test_later_mention_does_not_invalidate",
                "tests/retrieval/v3/test_entity_timeline.py::test_explicit_invalidation_known_at_vs_current",
            ],
            "measured": "unmeasured — guard invariant, not an ablation",
            "note": "a later mention never auto-corrects; only an "
                    "explicit transition ends the current view while "
                    "known-at still sees the prior revision.",
        },
        {
            "id": "X3",
            "mechanism": "source-preserving Markdown projection",
            "disposition": "adopt",
            "recommended": True,
            "evidence": [
                "tests/projections/test_edits.py::TestProposeEdit::test_proposal_created_claim_untouched",
                "tests/projections/test_markdown_projection.py::TestPathSafety::test_check_inside_escapes",
                "tests/projections/test_markdown_projection.py::TestPathSafety::test_safe_relpath_rejects_traversal",
            ],
            "measured": "unmeasured — guard invariant, not an ablation",
            "note": "edits are proposal-only reviews rows; original "
                    "evidence is never overwritten; root escapes and "
                    "hook/hidden names are rejected.",
        },
        {
            "id": "X4",
            "mechanism": "peer-perspective packs",
            "disposition": "adopt",
            "recommended": True,
            "evidence": [
                "tests/profiles/test_profiles.py::test_perspective_never_widens",
                "tests/profiles/test_profiles.py::test_group_session_no_merge_no_leak",
                "tests/profiles/test_x45_mechanisms.py::test_d16_session_peers_no_shared_private_packs",
            ],
            "measured": "unmeasured — guard invariant, not an ablation",
            "note": "packs bind their caller and attenuate only; two "
                    "session peers never receive each other's private "
                    "models.",
        },
        {
            "id": "X5",
            "mechanism": "topic-directed extraction policy",
            "disposition": "adopt",
            "recommended": True,
            "evidence": [
                "tests/profiles/test_x45_mechanisms.py::test_d17_topic_cannot_suppress_safety_events",
                "tests/profiles/test_x45_mechanisms.py::test_d17_exclusion_shaped_match_rejected",
            ],
            "measured": "unmeasured — guard invariant, not an ablation",
            "note": "deletion/consent-withdrawal/correction/security "
                    "events bypass topic match filters and are reported; "
                    "exclusion-shaped match specs are rejected.",
        },
        {
            "id": "X6",
            "mechanism": "competitor import honesty",
            "disposition": "adopt",
            "recommended": True,
            "evidence": [
                "tests/connectors/test_holographic.py::test_jsonl_imports_episodes_entities_claims",
                "tests/connectors/test_holographic.py::test_d18_dry_run_reports_missing_provenance_per_item",
            ],
            "measured": "unmeasured — guard invariant, not an ablation",
            "note": "items without original:true persist as imported "
                    "assertions; dry-run and the accepted receipt name "
                    "missing provenance per item.",
        },
        {
            "id": "X7",
            "mechanism": "session-to-durable promotion with explicit "
                         "retention consent",
            "disposition": "adopt",
            "recommended": True,
            "evidence": [
                "tests/observations/test_working_promotion.py::test_promotion_requires_retention_consent",
                "tests/observations/test_working_promotion.py::test_tool_permission_is_not_consent",
                "tests/observations/test_working_promotion.py::test_expired_revoked_consent_denies",
            ],
            "measured": "unmeasured — guard invariant, not an ablation",
            "note": "working items become durable only through "
                    "ingest_envelope under a live capture_authorizations "
                    "row — tool permission is not retention consent.",
        },
        {
            "id": "X8",
            "mechanism": "bounded associative probe (HRR-class "
                         "composition)",
            "disposition": "research_only",
            "recommended": False,
            "evidence": [
                "tests/retrieval/v3/test_entity_timeline.py::test_compose_report_exact_intersection_reference",
                "tests/retrieval/v3/test_graph_lane.py::test_reference_ops_probe_compose_contradictions",
            ],
            "measured": "unmeasured — no paired outcome vs exact "
                       "intersection + hybrid retrieval",
            "note": "§16 stop condition: remains research until it "
                    "beats exact intersection plus hybrid retrieval; "
                    "associative composition is reported AGAINST the "
                    "exact intersection, never instead of it (D14).",
        },
    ]


def load(path: Optional[str] = None) -> list[dict[str, Any]]:
    """Read the published disposition table."""
    with open(path or DISPOSITIONS_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data["rows"])


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Structural validation — every required id exactly once, legal
    dispositions, recommended only under ``adopt``."""
    problems: list[str] = []
    seen: set[str] = set()
    for row in rows:
        rid = row.get("id")
        if rid in seen:
            problems.append(f"duplicate mechanism id {rid!r}")
        seen.add(rid)
        if row.get("disposition") not in DISPOSITIONS:
            problems.append(
                f"{rid}: disposition must be one of {DISPOSITIONS}"
            )
        if row.get("recommended") and row.get("disposition") != "adopt":
            problems.append(
                f"{rid}: recommended requires disposition 'adopt' "
                "(D21 — failed/deferred/research mechanisms never "
                "enter recommended routing)"
            )
        if not isinstance(row.get("evidence"), list):
            problems.append(f"{rid}: evidence must be a list")
        if "measured" not in row:
            problems.append(f"{rid}: missing 'measured' field")
    missing = [m for m in MECHANISM_IDS if m not in seen]
    if missing:
        problems.append(f"missing mechanism rows: {missing}")
    return problems


def is_recommended(mech_id: str, path: Optional[str] = None) -> bool:
    """The D21 consumer check: may this mechanism enter recommended
    routing? Only ``adopt`` rows flagged recommended — a mechanism that
    lost its ablation is never kept because a competitor advertises it
    (V45-09.09)."""
    for row in load(path):
        if row["id"] == mech_id:
            return bool(row["recommended"])
    return False


def recommended_mechanisms(path: Optional[str] = None) -> list[str]:
    """Ids currently permitted in recommended routing."""
    return [
        row["id"] for row in load(path) if row.get("recommended")
    ]


def write(path: Optional[str] = None) -> str:
    """Regenerate ``dispositions.json`` from :func:`generate_rows`."""
    path = path or DISPOSITIONS_PATH
    rows = generate_rows()
    problems = validate_rows(rows)
    if problems:
        raise ValueError(f"generated table invalid: {problems}")
    doc = {
        "schema": 1,
        "kind": "v45_dispositions",
        "spec": "SPEC_V4_5 §16 (V45-16.01, D21)",
        "note": "generated by eval/v45/dispositions.py — edit the "
                "generator, never this file",
        "rows": rows,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


def check(path: Optional[str] = None) -> list[str]:
    """Verify the published file matches the generator output."""
    path = path or DISPOSITIONS_PATH
    if not os.path.isfile(path):
        return [f"{path} does not exist — run write()"]
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    problems = validate_rows(list(data.get("rows", [])))
    want = generate_rows()
    if data.get("rows") != want:
        problems.append(
            f"{path} is stale — regenerate with write()"
        )
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.v45.dispositions")
    ap.add_argument(
        "--check", action="store_true",
        help="verify dispositions.json is current; nonzero on drift",
    )
    args = ap.parse_args(argv)
    if args.check:
        problems = check()
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        if not problems:
            print("dispositions.json is current")
        return 1 if problems else 0
    path = write()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DISPOSITIONS",
    "DISPOSITIONS_PATH",
    "MECHANISM_IDS",
    "check",
    "generate_rows",
    "is_recommended",
    "load",
    "recommended_mechanisms",
    "validate_rows",
    "write",
]
