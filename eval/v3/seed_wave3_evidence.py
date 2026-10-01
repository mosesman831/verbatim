"""Seed the requirement ledger with Wave-1..3 implementation evidence.

Maps every requirement in sections with landed, tested code to the
artifacts that actually exercise it (§56.10: implementation,
integration, and empirical validation are separate fields).

Honesty rules:

- ``impl_test``/``integration`` entries are ``pass`` only where a real
  test file or suite exercises that section's behavior and is green.
- ``empirical`` entries are ``pass`` only for quantities the
  ``report_wave3.md`` run actually measured.
- Sections whose artifacts exist but were not verified per-requirement
  get ``declared`` — visible coverage, never counted as pass.
- Optional/deferred sections get nothing — the ledger must show the
  gap, not hide it (V3-05.06/V3-05.08).

Usage::

    python -m eval.v3.seed_wave3_evidence [--ledger PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

from .ledger import load_ledger

_DEFAULT_LEDGER = os.path.join(
    os.path.dirname(__file__), "registry", "ledger.json"
)

# ---------------------------------------------------------------------------
# Section -> artifacts. Each entry: (impl_test, integration, empirical) lists
# of (artifact, status). Statuses are honest: "pass" only where the artifact
# verifiably exercises the section; "declared" where coverage exists but
# per-requirement validation is not established.
# ---------------------------------------------------------------------------

_IT = "impl_test"
_IG = "integration"
_EM = "empirical"

SECTION_EVIDENCE: dict[int, dict[str, list[tuple[str, str]]]] = {
    # 02 — research conclusions (documented, not implemented behavior)
    2: {
        _IG: [("SPEC_V3.md R1 §2 + source register", "declared")],
    },
    # 03 — verified v2 baseline + mandatory corrections
    3: {
        _IT: [
            ("tests/test_v2_acceptance.py", "pass"),
            ("tests/test_e2e_pipeline.py", "pass"),
        ],
        _IG: [("eval/v3/f27.py + tests/test_v3_integration.py", "pass")],
        _EM: [
            (
                "eval/v3/f27_report.json (F27 reproduced: 0.61/0.88 "
                "vs saved 0.58/0.89; 0 unattributed — mechanism is "
                "rank_shift/packed_out, not generation stranding)",
                "pass",
            ),
            (
                "eval/v3/f_items_audit.json (F13–F27 dispositioned: "
                "10 evidenced, 3 fixed, 1 partially_evidenced, "
                "1 explicitly_deferred — no unregistered gaps)",
                "declared",
            ),
        ],
    },
    # 04 — system invariants (verified across the suite, not per-req)
    4: {
        _IT: [("tests/test_v3_integration.py", "declared")],
        _IG: [("eval/v3/report_wave3.md (0 disclosure violations)", "declared")],
    },
    # 05 — deployment profiles, capability tiers
    5: {
        _IT: [],
        _IG: [
            ("RELEASE_MANIFEST_V3.json + probe_capabilities", "declared")
        ],
        _EM: [],
    },
    # 06 — two planes, module ownership, derivation graph
    6: {
        _IT: [("tests/test_derivations.py", "pass")],
        _IG: [("tests/privacy/test_deletion.py", "pass")],
    },
    # 07 — modes, config contract, defaults
    7: {
        _IT: [("tests/storage/test_v3_foundation.py", "pass")],
        _IG: [("tests/test_v3_integration.py", "declared")],
    },
    # 08 — principals, perspectives, audiences
    8: {
        _IT: [("tests/governance/test_governance.py", "pass")],
        _IG: [("tests/api_v3/test_api_v3.py", "declared")],
    },
    # 09 — authorization, verbs, revocation
    9: {
        _IT: [("tests/governance/test_governance.py", "pass")],
        _IG: [("eval/v3/suite_security.py (governance lane)", "pass")],
    },
    # 10 — multi-agent shared memory, handoff
    10: {
        _IT: [("tests/governance/test_governance.py", "pass")],
        _IG: [("tests/test_v3_integration.py", "declared")],
    },
    # 11 — consent, purpose limitation, egress
    11: {
        _IT: [("tests/governance/test_governance.py", "pass")],
        _IG: [("eval/v3/suite_security.py (consent cycle)", "pass")],
    },
    # 12 — source envelope, trajectory capture
    12: {
        _IT: [
            ("tests/evidence/test_evidence.py", "pass"),
            ("tests/sdk/test_capture.py", "pass"),
        ],
        _IG: [("tests/test_v3_integration.py", "pass")],
    },
    # 13 — integration depths, host adapter contract
    13: {
        _IT: [
            ("tests/sdk/test_capture.py", "pass"),
            ("tests/adapters/test_adapters.py", "pass"),
        ],
        _IG: [("tests/test_v3_integration.py", "pass")],
    },
    # 14 — trust classes, taint, admission ladder
    14: {
        _IT: [("tests/security/test_security.py", "pass")],
        _IG: [
            ("tests/test_v3_integration.py (quarantine cascade)", "pass"),
            ("eval/v3/suite_security.py", "pass"),
        ],
    },
    # 15 — harvest v3 segmentation
    15: {
        _IT: [("tests/test_harvest_v3.py", "pass")],
        _IG: [("tests/test_v3_integration.py", "pass")],
    },
    # 17 — memory kinds, derivation contract
    17: {
        _IT: [
            ("tests/test_derivations.py", "pass"),
            ("tests/privacy/test_deletion.py", "pass"),
        ],
        _IG: [("tests/privacy/test_deletion.py", "pass")],
    },
    # 18 — claims, bitemporal, supersession (v2 machinery inherited)
    18: {
        _IT: [
            ("tests/core/test_claims.py", "pass"),
            ("tests/evidence/test_supersession.py", "pass"),
            ("tests/evidence/test_relations.py", "pass"),
        ],
        _IG: [("tests/test_v3_integration.py", "declared")],
        _EM: [
            (
                "eval/v3/report_wave3.md tasks suite (pr-007 resolved; "
                "verbatim_v3 negative transfer 0.000)",
                "pass",
            ),
            (
                "eval/v3/report_wave3.md supersession metric (5/5 applied "
                "true pairs, 4/4 false-positive twins clean)",
                "pass",
            ),
            (
                "verbatim/evidence/relations.py — F26/V3-18.03: "
                "unstructured relation discovery. Deterministic checks "
                "(negation asymmetry, numeric mismatch, correction "
                "markers) over same-scope entity neighborhoods, optional "
                "encoder corroboration with recorded confidence, then "
                "conflicts_with edge + dispute review. Generative models "
                "never label; detection never applies transitions",
                "pass",
            ),
            (
                "eval/v3/g3_report.json — G3 deep twin corpus: 460 "
                "synthetic near-miss twins over 12 categories through "
                "both real detectors, 0 false positives (upper-95% "
                "0.0065 rule-of-three / 0.0083 Wilson < 0.01 target); "
                "30/30 true pairs detected. Detector fixes the corpus "
                "surfaced: conditional/hearsay/intent/cancellation "
                "guards, object-form quote capture (recall), "
                "status-negation + imperative-negation + successor "
                "endorsement precision guards",
                "pass",
            ),
        ],
    },
    # 19 — entities, aliases, profiles
    19: {
        _IT: [("tests/core/test_claims.py", "declared")],
        _IG: [],
    },
    # 20 — episodes, transitions
    20: {
        _IT: [("tests/experience/test_v3_episodes.py", "pass")],
        _IG: [("tests/experience/v3_seed.py", "declared")],
    },
    # 21 — procedural memory representation
    21: {
        _IT: [("tests/procedures/test_procedures.py", "pass")],
        _IG: [("eval/v3/suite_tasks.py", "declared")],
    },
    # 22 — procedure compilation, promotion, reuse
    22: {
        _IT: [("tests/procedures/test_procedures.py", "pass")],
        _IG: [("eval/v3/suite_tasks.py", "declared")],
    },
    # 23 — observations
    23: {
        _IT: [("tests/observations/test_observations.py", "pass")],
        _IG: [],
    },
    # 24 — prospective, working, environment memory
    24: {
        _IT: [("tests/observations/test_observations.py", "pass")],
        _IG: [],
    },
    # 25 — social/collaborative memory
    25: {
        _IT: [("tests/observations/test_observations.py", "pass")],
        _IG: [],
    },
    # 26 — memory controller
    26: {
        _IT: [
            ("tests/retrieval/v3/test_retrieval_v3.py", "pass"),
            ("tests/retrieval/v3/test_learned.py", "pass"),
        ],
        _IG: [("eval/v3/suite_tasks.py", "pass")],
        _EM: [
            ("eval/v3/report_wave3.md (paired delta +0.750)", "pass"),
            (
                "eval/v3/g8_report.json + eval/v3/g8_policy.json "
                "(learned controller executed via paired sandbox arms: "
                "11 bounded actions × 52 tasks through real recall_v3; "
                "verdict measured — criterion not met, learned converges "
                "to deterministic warm start on this corpus)",
                "pass",
            ),
        ],
    },
    # 27 — query/context contracts
    27: {
        _IT: [("tests/retrieval/v3/test_retrieval_v3.py", "pass")],
        _IG: [("eval/v3/suite_retrieval.py", "declared")],
    },
    # 28 — retrieval lanes
    28: {
        _IT: [
            ("tests/retrieval/v3/test_retrieval_v3.py", "pass"),
            ("tests/embeddings/test_hashing.py", "pass"),
        ],
        _IG: [("eval/v3/suite_retrieval.py", "pass")],
        _EM: [
            (
                "eval/v3/report_wave3.md (vector_rag + dense lane "
                "measured; zero capability_unavailable warnings)",
                "pass",
            )
        ],
    },
    # 29 — fusion, abstention, calibration
    29: {
        _IT: [("tests/retrieval/v3/test_retrieval_v3.py", "pass")],
        _IG: [("eval/v3/suite_retrieval.py", "pass")],
        _EM: [
            (
                "eval/v3/report_wave3.md (abstain P/R 1.000, "
                "spurious 0.000)",
                "pass",
            )
        ],
    },
    # 30 — context assembly, packs
    30: {
        _IT: [("tests/retrieval/v3/test_retrieval_v3.py", "pass")],
        _IG: [("eval/v3/suite_grounding.py", "pass")],
    },
    # 31 — retrieval-time screening, action gating
    31: {
        _IT: [("tests/security/test_security.py", "pass")],
        _IG: [("eval/v3/suite_security.py", "pass")],
    },
    # 32 — influence tracing
    32: {
        _IT: [("tests/retrieval/v3/test_retrieval_v3.py", "pass")],
        _IG: [],
    },
    # 33 — threat model v3
    33: {
        _IT: [],
        _IG: [("THREAT_MODEL.md", "declared")],
    },
    # 34 — poisoning defense lifecycle
    34: {
        _IT: [("tests/security/test_security.py", "pass")],
        _IG: [
            ("tests/test_v3_integration.py", "pass"),
            ("eval/v3/suite_security.py", "pass"),
        ],
        _EM: [
            (
                "eval/v3/report_wave3.md (8/8 poison blocked at "
                "ingest, benign 1.000)",
                "pass",
            )
        ],
    },
    # 35 — vault, sanitized search
    35: {
        _IT: [
            ("tests/privacy/test_v3_vault.py", "pass"),
            ("tests/privacy/test_deletion.py", "pass"),
        ],
        _IG: [("tests/privacy/test_v3_vault.py", "declared")],
        _EM: [],
    },
    # 36 — retention, purge v3, revocation propagation
    36: {
        _IT: [("tests/privacy/test_deletion.py", "pass")],
        _IG: [("tests/privacy/test_deletion.py (verify_closure)", "pass")],
    },
    # 37 — encryption at rest, keys, recovery
    37: {
        _IT: [("tests/privacy/test_v3_vault.py", "pass")],
        _IG: [],
    },
    # 38 — storage engines, profile bindings
    38: {
        _IT: [("tests/storage/test_v3_foundation.py", "pass")],
        _IG: [],
    },
    # 39 — schema v3
    39: {
        _IT: [("tests/storage/test_v3_foundation.py", "pass")],
        _IG: [("tests/test_v3_integration.py", "declared")],
    },
    # 40 — transactions, durable jobs, workers
    40: {
        _IT: [
            ("tests/storage/test_v3_foundation.py", "pass"),
            ("tests/jobs/test_v2_jobs.py", "pass"),
        ],
        _IG: [("tests/test_v3_integration.py", "pass")],
    },
    # 41 — index generations, artifacts
    41: {
        _IT: [("tests/retrieval/test_v2_retrieval.py", "declared")],
        _IG: [("eval/v3/f27.py", "declared")],
        _EM: [],
    },
    # 42 — typed decision protocol
    42: {
        _IT: [("tests/governance/test_governance.py", "declared")],
        _IG: [],
    },
    # 43 — replay laboratory
    43: {
        _IT: [
            ("tests/evidence/test_evidence.py", "pass"),
            ("tests/replay/test_replay_lab.py", "pass"),
        ],
        _IG: [],
        _EM: [
            (
                "verbatim/replay/lab.py — replay_routing re-executes "
                "recorded routing_decisions in a sandbox under variant "
                "policy tables (state_json pins inputs; unpinned rows "
                "report not_replayable). paired_execution runs full "
                "delivery-path arms (V3-43.05): same snapshot, isolated "
                "writes, identical replay authorization, real recall_v3 "
                "deliveries compared on item sets. paired_admission runs "
                "write-side arms: identical payloads through the real "
                "Ingester+run_pending drain — harvesting, screening, "
                "admission, relation discovery, consolidation. Variant "
                "surface: class_routes/class_lanes/tier_budgets/"
                "lane_weights/abstain thresholds/admission_context. "
                "model_calls=0 everywhere (V3-43.06)",
                "pass",
            )
        ],
    },
    # 45 — observability (journal, event log)
    45: {
        _IT: [("tests/storage/test_v3_foundation.py", "declared")],
        _IG: [],
    },
    # 46 — error taxonomy
    46: {
        _IT: [("tests/api_v3/test_api_v3.py", "declared")],
        _IG: [],
    },
    # 47 — host-neutral API v3
    47: {
        _IT: [("tests/api_v3/test_api_v3.py", "pass")],
        _IG: [("tests/test_v3_integration.py", "pass")],
    },
    # 48 — MCP surface
    48: {
        _IT: [
            ("tests/api_v3/test_mcp_v3.py", "pass"),
            ("tests/api_v3/test_api_v3.py", "pass"),
        ],
        _IG: [("tests/api_v3/test_mcp_v3.py (fresh-store e2e)", "pass")],
    },
    # 49 — framework adapters
    49: {
        _IT: [
            ("tests/adapters/test_adapters.py", "pass"),
            ("tests/sdk/test_capture.py", "pass"),
        ],
        _IG: [],
    },
    # 51 — CLI/operator workflows (v2 CLI inherited)
    51: {
        _IT: [("tests/storage/test_storage.py", "declared")],
        _IG: [("tests/ops/test_operational.py", "pass")],
    },
    # 44 — workload tiers, systems tests (T1/T2 measured)
    44: {
        _IT: [("tests/ops/test_operational.py", "pass")],
        _IG: [],
        _EM: [("tests/ops/test_operational.py (T1 p95 24.4ms ≤25ms, "
               "T2 p95 44.9ms ≤60ms)", "pass")],
    },
    # 53 — evaluation program
    53: {
        _IT: [
            ("tests/eval_v3/test_eval_v3.py", "pass"),
            ("tests/eval/test_v3_harness.py", "pass"),
        ],
        _IG: [
            ("eval/v3/suites.py", "pass"),
            (
                "eval/v3/suite_conformance.py + run.py --suite conformance "
                "(fourth suite executable: 132/132 cases pass)",
                "pass",
            ),
        ],
        _EM: [("eval/v3/report_wave3.md", "pass")],
    },
    # 54 — metrics, gates
    54: {
        _IT: [("tests/eval_v3/test_eval_v3.py", "pass")],
        _IG: [("eval/v3/report_wave3.md", "pass")],
        _EM: [("eval/v3/report_wave3.md", "pass")],
    },
    # 55 — acceptance scenarios (subset verified by integration tests)
    55: {
        _IT: [("tests/test_v3_integration.py", "pass")],
        _IG: [
            ("tests/api_v3/test_mcp_v3.py (fresh-store capture→recall)",
             "pass")
        ],
    },
    # 56 — verification machinery
    56: {
        _IT: [
            ("tests/eval_v3/test_eval_v3.py", "pass"),
            ("tests/eval/test_v3_harness.py (conformance runner cases)", "pass"),
        ],
        _IG: [
            ("eval/v3/registry/ledger.json", "declared"),
            (
                "eval/v3/suite_conformance.py (conformance suite — the "
                "declared §53 fourth suite now executes: 132 cases across "
                "core-api/cli/mcp-stdio/api-v3/mcp-v3/capture-sdk/adapters "
                "surfaces; hermes-live + adk-live declared unavailable)",
                "pass",
            ),
        ],
        _EM: [
            (
                "conformance run: 132/132 cases pass, 0 errors "
                "(7 surfaces measured, 2 live-host lanes declared "
                "unavailable)",
                "pass",
            )
        ],
    },
    # 57 — v2→v3 migration
    57: {
        _IT: [("tests/storage/test_v3_foundation.py", "pass")],
        _IG: [],
    },
    # 58 — phases/ownership (artifact sections: integration-only)
    58: {
        _IG: [("SPEC_V3.md R1 + AGENTS.md build status", "declared")],
    },
    # 59 — packaging, supply chain
    59: {
        _IT: [],
        _IG: [("pyproject.toml", "declared")],
    },
    # 60 — trade-offs, decision owners
    60: {
        _IG: [("SPEC_V3.md R1 §60 review disposition", "declared")],
    },
    # 61 — source register
    61: {
        _IG: [("SPEC_V3.md R1 §61 source register", "declared")],
    },
    # 62 — v2/v3 compatibility
    62: {
        _IG: [("SPEC_V3.md R1 §62 mapping", "declared")],
    },
    # 63 — release
    63: {
        _IG: [("RELEASE_MANIFEST_V3.json", "declared")],
    },
}

RUN_ID = "wave3-impl"


def seed(ledger_path: str, *, dry_run: bool = False) -> dict:
    ledger = load_ledger(ledger_path)
    counts = {"requirements": 0, "entries": 0, "skipped": 0}
    for req_id, rec in ledger.requirements.items():
        ev = SECTION_EVIDENCE.get(rec.section)
        if not ev:
            counts["skipped"] += 1
            continue
        registered = False
        for kind in (_IT, _IG, _EM):
            for artifact, status in ev.get(kind, ()):
                ledger.register(
                    req_id, kind, artifact, status, run_id=RUN_ID
                )
                counts["entries"] += 1
                registered = True
        if registered:
            counts["requirements"] += 1
    if not dry_run:
        ledger.write(ledger_path)
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", default=_DEFAULT_LEDGER)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    counts = seed(args.ledger, dry_run=args.dry_run)
    print(
        f"{'[dry-run] ' if args.dry_run else ''}"
        f"requirements touched: {counts['requirements']}, "
        f"evidence entries: {counts['entries']}, "
        f"requirements with no registered evidence (gap visible): "
        f"{counts['skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
