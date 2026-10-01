"""Requirement→evidence ledger (SPEC_V3 §56, V3-56.01, V3-56.10, V3-56.11).

The ledger is the frozen evidence matrix behind every ``[x]`` claim:

* Each requirement declares ``required_evidence`` — the set of evidence
  *kinds* it needs (``impl_test`` behavior-level tests, ``integration``
  public-entry-point runs or reviewed artifacts, ``empirical`` measured
  artifacts). The required set is frozen per section before
  implementation; removing a failing case is a contract change, not a
  generator override (§56.11).
* ``compute_completion`` returns ``pass`` only when EVERY required kind
  has at least one passing evidence entry at the same commit/run
  manifest. One marked test is never completion (§56.01): missing,
  failed, skipped, xfailed, stale, or excluded required evidence
  prevents it.
* Statuses follow §03.04: requirements report ``tested``, ``partial``,
  ``deferred``, ``unverified``, or ``not_applicable``; the ledger adds
  the honest run-level verdicts ``pass``, ``fail``, and ``inconclusive``
  (§54.11 missing-support rule).
* §56.11 evidence typing: governance, architecture, license, and
  benchmark requirements accept a signed decision, reviewed artifact, or
  reproducible result — registered here as ``integration`` evidence
  rather than a fake unit test.

The JSON file layout is stable so the §63 release checklist and
RELEASE_MANIFEST_V3.json can point at it:

.. code-block:: json

    {
      "schema": 1,
      "spec": "SPEC_V3.md R1",
      "requirements": {
        "V3-53.01": {
          "section": 53,
          "title": "...",
          "status": "pending",
          "required_evidence": ["impl_test", "integration", "empirical"],
          "evidence": [
            {"kind": "impl_test", "artifact": "tests/...py::test_x",
             "status": "pass", "run_id": "...", "note": ""}
          ]
        }
      }
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

#: Evidence kinds (§56.10: implementation, integration validation, and
#: empirical validation are separate fields; ``[x]`` is their required
#: intersection, not a coverage count).
EVIDENCE_KINDS = ("impl_test", "integration", "empirical")

#: Evidence-entry statuses. ``declared`` means an artifact exists but has
#: not been validated; ``inconclusive`` means the run lacked statistical
#: support (§54.11) — neither counts as pass.
EVIDENCE_STATUSES = ("pass", "fail", "pending", "declared", "inconclusive")

#: Requirement-level statuses (§03.04 plus the seed-time ``pending``,
#: which is ``unverified`` before any run). ``pass`` is emitted only by
#: ``compute_completion`` — never assigned by hand.
REQUIREMENT_STATUSES = (
    "pending",
    "tested",
    "partial",
    "deferred",
    "unverified",
    "not_applicable",
    "fail",
    "inconclusive",
)

#: Completion verdicts from ``compute_completion``.
VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_PARTIAL = "partial"
VERDICT_UNVERIFIED = "unverified"

#: Sections whose requirements demand measured/empirical artifacts in
#: addition to behavior-level and integration evidence (§53/§54 gates,
#: §44 envelopes, §34/§35 attack/privacy rates, §02 commitments needing
#: measured effect per V3-02.01, §05 capability measurement, §26 learned
#: controller G8, §29 calibration, §41 artifact measurement, §43 replay).
EMPIRICAL_SECTIONS = frozenset(
    {2, 5, 26, 29, 34, 35, 41, 43, 44, 53, 54}
)

#: Governance/process sections where §56.11 reviewed artifacts or signed
#: decisions are the evidence (phases, decisions, source register,
#: compatibility mapping, release checklist). A fake unit test would not
#: be honest evidence here; the reviewed-artifact registration is
#: recorded as ``integration``.
ARTIFACT_SECTIONS = frozenset({58, 60, 61, 62, 63})

DEFAULT_REQUIRED_EVIDENCE = ("impl_test", "integration")


def required_evidence_for_section(section: int) -> tuple:
    """The frozen required-evidence set for a spec section (§56.01).

    The mapping is declared *before* implementation evidence exists —
    it describes what the section demands, not what currently passes.
    """
    if section in ARTIFACT_SECTIONS:
        return ("integration",)
    if section in EMPIRICAL_SECTIONS:
        return DEFAULT_REQUIRED_EVIDENCE + ("empirical",)
    return DEFAULT_REQUIRED_EVIDENCE


@dataclass(frozen=True)
class EvidenceEntry:
    """One registered evidence artifact for a requirement."""

    kind: str
    artifact: str
    status: str = "declared"
    run_id: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in EVIDENCE_KINDS:
            raise ValueError(
                f"evidence kind must be one of {EVIDENCE_KINDS}, got {self.kind!r}"
            )
        if self.status not in EVIDENCE_STATUSES:
            raise ValueError(
                f"evidence status must be one of {EVIDENCE_STATUSES}, "
                f"got {self.status!r}"
            )
        if not self.artifact:
            raise ValueError("artifact pointer required")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "artifact": self.artifact,
            "status": self.status,
            "run_id": self.run_id,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceEntry":
        return cls(
            kind=d["kind"],
            artifact=d["artifact"],
            status=d.get("status", "declared"),
            run_id=d.get("run_id", ""),
            note=d.get("note", ""),
        )


@dataclass
class RequirementRecord:
    """One requirement's ledger row."""

    req_id: str
    section: int
    title: str = ""
    status: str = "pending"
    required_evidence: tuple = DEFAULT_REQUIRED_EVIDENCE
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "title": self.title,
            "status": self.status,
            "required_evidence": list(self.required_evidence),
            "evidence": [e.to_dict() for e in self.evidence],
        }

    @classmethod
    def from_dict(cls, req_id: str, d: dict) -> "RequirementRecord":
        return cls(
            req_id=req_id,
            section=int(d.get("section", 0)),
            title=d.get("title", ""),
            status=d.get("status", "pending"),
            required_evidence=tuple(
                d.get("required_evidence", DEFAULT_REQUIRED_EVIDENCE)
            ),
            evidence=[EvidenceEntry.from_dict(e) for e in d.get("evidence", [])],
        )


class Ledger:
    """The requirement→evidence ledger.

    Construct from ``data`` (the parsed ledger.json dict) or
    :func:`load_ledger`. Mutations go through :meth:`register`, which
    validates kinds/statuses and dedupes on (req, kind, artifact, run_id)
    so repeated registrations are idempotent.
    """

    def __init__(self, data: Optional[dict] = None) -> None:
        data = data or {}
        self.schema = int(data.get("schema", 1))
        self.spec = data.get("spec", "")
        self.meta = dict(data.get("meta", {}))
        self.requirements: dict[str, RequirementRecord] = {
            rid: RequirementRecord.from_dict(rid, d)
            for rid, d in (data.get("requirements") or {}).items()
        }

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def add_requirement(
        self,
        req_id: str,
        *,
        section: int,
        title: str = "",
        required_evidence: Optional[Iterable[str]] = None,
        status: str = "pending",
    ) -> RequirementRecord:
        """Declare a requirement row (idempotent on req_id)."""
        kinds = (
            tuple(required_evidence)
            if required_evidence is not None
            else required_evidence_for_section(section)
        )
        for k in kinds:
            if k not in EVIDENCE_KINDS:
                raise ValueError(f"unknown evidence kind {k!r}")
        if status not in REQUIREMENT_STATUSES:
            raise ValueError(f"unknown requirement status {status!r}")
        rec = self.requirements.get(req_id)
        if rec is None:
            rec = RequirementRecord(
                req_id=req_id,
                section=section,
                title=title,
                status=status,
                required_evidence=kinds,
            )
            self.requirements[req_id] = rec
        else:
            # Idempotent re-declaration keeps registered evidence but
            # refreshes the frozen contract fields.
            rec.section = section
            rec.title = title or rec.title
            rec.required_evidence = kinds
            rec.status = status
        return rec

    def register(
        self,
        req_id: str,
        evidence_kind: str,
        artifact: str,
        status: str = "declared",
        *,
        run_id: str = "",
        note: str = "",
    ) -> EvidenceEntry:
        """Register one evidence artifact for a requirement.

        Idempotent: re-registering the same (req_id, kind, artifact,
        run_id) updates the status/note rather than duplicating the row —
        a rerun replaces its earlier result instead of inflating counts.
        """
        if req_id not in self.requirements:
            raise KeyError(f"unknown requirement {req_id!r}")
        entry = EvidenceEntry(
            kind=evidence_kind,
            artifact=artifact,
            status=status,
            run_id=run_id,
            note=note,
        )
        rec = self.requirements[req_id]
        for i, e in enumerate(rec.evidence):
            if (
                e.kind == entry.kind
                and e.artifact == entry.artifact
                and e.run_id == entry.run_id
            ):
                rec.evidence[i] = entry
                break
        else:
            rec.evidence.append(entry)
        return entry

    # ------------------------------------------------------------------
    # completion
    # ------------------------------------------------------------------

    def kinds_passed(self, req_id: str) -> frozenset:
        rec = self.requirements[req_id]
        return frozenset(e.kind for e in rec.evidence if e.status == "pass")

    def kinds_failed(self, req_id: str) -> frozenset:
        rec = self.requirements[req_id]
        return frozenset(e.kind for e in rec.evidence if e.status == "fail")

    def missing_kinds(self, req_id: str) -> tuple:
        rec = self.requirements[req_id]
        passed = self.kinds_passed(req_id)
        return tuple(k for k in rec.required_evidence if k not in passed)

    def compute_completion(self, req_id: str) -> str:
        """The §56.01 verdict for one requirement.

        * ``fail`` — a required kind has failing evidence.
        * ``pass`` — every required kind has at least one passing entry.
        * ``partial`` — some evidence exists but required kinds are
          missing, pending, declared-only, or inconclusive.
        * ``unverified`` — no evidence registered at all.
        """
        rec = self.requirements.get(req_id)
        if rec is None:
            raise KeyError(f"unknown requirement {req_id!r}")
        required = set(rec.required_evidence)
        if required & self.kinds_failed(req_id):
            return VERDICT_FAIL
        if not rec.evidence:
            return VERDICT_UNVERIFIED
        if required and not (required - self.kinds_passed(req_id)):
            return VERDICT_PASS
        return VERDICT_PARTIAL

    def is_complete(self, req_id: str) -> bool:
        """``[x]`` eligibility — every required evidence kind passes."""
        return self.compute_completion(req_id) == VERDICT_PASS

    # ------------------------------------------------------------------
    # coverage
    # ------------------------------------------------------------------

    def coverage_report(self, *, gates: Optional[dict] = None) -> dict:
        """Counts per status, section, and evidence kind (§56.10).

        ``gates`` optionally carries the G0–G9 status map from the
        release manifest so the report can publish gate readiness beside
        requirement coverage without inventing a verdict.
        """
        by_status: dict[str, int] = {}
        by_section: dict[int, dict[str, int]] = {}
        kind_required: dict[str, int] = {k: 0 for k in EVIDENCE_KINDS}
        kind_passing: dict[str, int] = {k: 0 for k in EVIDENCE_KINDS}
        kind_registered: dict[str, int] = {k: 0 for k in EVIDENCE_KINDS}
        incomplete: list[str] = []

        for rid, rec in sorted(self.requirements.items()):
            verdict = self.compute_completion(rid)
            by_status[verdict] = by_status.get(verdict, 0) + 1
            sec = by_section.setdefault(
                rec.section,
                {"total": 0, "pass": 0, "fail": 0, "partial": 0,
                 "unverified": 0},
            )
            sec["total"] += 1
            sec[verdict] = sec.get(verdict, 0) + 1
            for k in rec.required_evidence:
                kind_required[k] += 1
            for e in rec.evidence:
                kind_registered[e.kind] += 1
            for k in self.kinds_passed(rid):
                kind_passing[k] += 1
            if verdict != VERDICT_PASS:
                incomplete.append(rid)

        return {
            "total_requirements": len(self.requirements),
            "verdicts": by_status,
            "by_section": {
                str(s): by_section[s] for s in sorted(by_section)
            },
            "evidence_kinds": {
                k: {
                    "required_refs": kind_required[k],
                    "registered": kind_registered[k],
                    "passing_requirements": kind_passing[k],
                }
                for k in EVIDENCE_KINDS
            },
            "complete": by_status.get(VERDICT_PASS, 0),
            "incomplete_requirements": incomplete,
            "gates": dict(gates or {}),
        }

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "spec": self.spec,
            "meta": self.meta,
            "requirements": {
                rid: rec.to_dict()
                for rid, rec in sorted(self.requirements.items())
            },
        }

    def write(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        return path


def load_ledger(path: str) -> Ledger:
    """Load a ledger.json into a :class:`Ledger`."""
    with open(path, "r", encoding="utf-8") as f:
        return Ledger(json.load(f))
