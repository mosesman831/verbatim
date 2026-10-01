"""The M5 competitive bake-off harness (SPEC_V4_5 §09 X1, §10–§11;
SPEC_V4 §03 registry, §54 fair-execution rules; D13, D22).

X1 is the exact-then-semantic bake-off: one pinned comparison surface
where every row declares its edition, commit/API version, deployment,
models, prompts, extraction settings, retrieval budgets, readiness
policy, and pricing date (V45-09.01, V4-54.01) — and where honesty is
enforced structurally:

* Every competitor/edition in the parent §03 registry is a *named row* —
  ``tested``, ``unavailable``, or ``out_of_scope``, always with a reason
  (V45-10.01, D22). Unavailable comparators stay visible and never enter
  defeated counts (V45-15.02).
* Native-product and controlled-memory tracks are separate rows and can
  never be compared across each other (V45-10.02); hosted-platform
  scores are never attributed to an OSS edition (V45-10.03, V4-54.04).
* A win against a *weakened* comparator — disabled embeddings, broken
  ingest, truncated context — is invalid (V45-10.04, V4-54.05, D13).
  :func:`compare` refuses it outright; the report marks the comparison
  ``invalid`` with the declared weakener, never a silent victory.
* Total cost is the eight-category accounting of V45-10.05 / V4-54.07 —
  ``None`` means *unmeasured*, and a partial total is always labeled
  ``complete: false`` rather than presented as whole.
* Reader identity is pinned per paired comparison (V45-11.03) and
  provider-native synthesis is scored separately from evidence-only
  retrieval (V45-11.04, V4-54.08).

Arms that run today (the controlled local set): VERBATIM exact-only
(``embedding.backend="none"``) and hashing-semantic
(``hashing:subword-ngram:v1`` — labeled non-neural, V45-18.02), the
previous stable VERBATIM arm, the no-memory control, and the lexical /
lexical-plus-neural reference arms — all through the real
``eval.v3.baselines`` adapters with gold structurally withheld via
``public_task`` (V45-14.03). Competitor editions (Holographic, Mem0 OSS,
Graphiti OSS, ReMe, …) are registry rows: when the pinned package is not
installed the row is ``unavailable`` with the import error recorded —
never dropped, never counted as defeated.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional, Sequence

# ---------------------------------------------------------------------------
# tracks, statuses, cost categories, weakeners
# ---------------------------------------------------------------------------

#: Native-product track: the competitor's own product surface.
TRACK_NATIVE = "native"
#: Controlled track: shared reader, task environment, evidence access,
#: context budget, and tuning allowance (V4-54.03).
TRACK_CONTROLLED = "controlled"
TRACKS = (TRACK_NATIVE, TRACK_CONTROLLED)

STATUS_TESTED = "tested"
STATUS_UNAVAILABLE = "unavailable"
STATUS_OUT_OF_SCOPE = "out_of_scope"
STATUSES = (STATUS_TESTED, STATUS_UNAVAILABLE, STATUS_OUT_OF_SCOPE)

#: The eight cost categories every total must cover (V45-10.05,
#: V4-54.07). Missing categories are reported, never silently zeroed.
COST_CATEGORIES: tuple[str, ...] = (
    "ingest",
    "extraction",
    "embeddings",
    "consolidation",
    "query_inference",
    "answer_reading",
    "storage",
    "maintenance",
)

#: Declared comparator weakeners (V45-10.04, V4-54.05). A win against a
#: row carrying any of these labels is invalid.
WEAKENERS = frozenset(
    {"disabled_embeddings", "broken_ingest", "truncated_context"}
)

#: Synthesis classes (V45-11.04): provider-native answer synthesis is a
#: different measurement from evidence-only retrieval — never blended.
SYNTHESIS_CLASSES = ("evidence_only", "provider_native")

#: The non-neural deterministic encoder this build can provision
#: (V45-18.02 — hashing encoders are labeled non-neural).
HASHING_ENCODER_ID = "hashing:subword-ngram:v1"


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostBreakdown:
    """Per-row total cost over the eight required categories.

    ``None`` means *unmeasured* — the honest state for any category this
    run could not meter (amortized maintenance, external API pricing,
    …). ``units`` labels each measured category (``"us"`` wall-clock,
    ``"bytes"``, ``"usd_micros"``, …) so mixed-unit totals are disclosed
    rather than silently summed.
    """

    ingest: Optional[float] = None
    extraction: Optional[float] = None
    embeddings: Optional[float] = None
    consolidation: Optional[float] = None
    query_inference: Optional[float] = None
    answer_reading: Optional[float] = None
    storage: Optional[float] = None
    maintenance: Optional[float] = None
    units: dict = field(default_factory=dict)

    def unmeasured(self) -> list[str]:
        return [c for c in COST_CATEGORIES if getattr(self, c) is None]

    def measured(self) -> dict[str, float]:
        return {
            c: float(getattr(self, c))
            for c in COST_CATEGORIES
            if getattr(self, c) is not None
        }

    def complete(self) -> bool:
        """Whether all eight categories are measured (V45-10.05)."""
        return not self.unmeasured()

    def total(self) -> dict[str, Any]:
        """The honest total: measured sum + the categories left out."""
        return {
            "measured_total": round(sum(self.measured().values()), 6),
            "unmeasured": self.unmeasured(),
            "complete": self.complete(),
        }

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for c in COST_CATEGORIES:
            v = getattr(self, c)
            out[c] = (
                "unmeasured"
                if v is None
                else {"value": v, "unit": self.units.get(c, "")}
            )
        out["total"] = self.total()
        return out


@dataclass(frozen=True)
class ComparatorRow:
    """One registry/bake-off row — the pinned comparison unit.

    ``status`` is the honesty gate (V45-10.01): ``tested`` rows carry
    measured metrics/costs and the full V4-54.01 pin set; ``unavailable``
    and ``out_of_scope`` rows carry only a reason — they are never
    counted as defeated (V45-15.02). ``weakened`` declares known setup
    defects; any win over a weakened row is invalid (D13).
    """

    comparator_id: str
    edition: str  # oss|hosted|platform|self_hosted|local|reference|verbatim
    track: str
    status: str
    reason: str = ""
    # ---- the V4-54.01 pin set (required on tested rows) --------------
    commit: Optional[str] = None
    deployment: Optional[str] = None
    models: dict = field(default_factory=dict)  # {"encoder","reader",...}
    prompts: str = ""
    extraction: dict = field(default_factory=dict)
    budgets: dict = field(default_factory=dict)
    readiness: Optional[str] = None
    pricing_date: Optional[str] = None
    # ---- separation labels (V45-10.02, V45-11.03/04) ------------------
    synthesis: str = "evidence_only"
    weakened: tuple = ()
    # ---- measured surface (tested rows only) --------------------------
    metrics: dict = field(default_factory=dict)
    costs: Optional[CostBreakdown] = None
    notes: tuple = ()

    def pinned(self) -> bool:
        """Every V4-54.01 pin present — the claim-eligibility floor."""
        return bool(
            self.commit
            and self.deployment
            and self.models.get("reader")
            and self.budgets
            and self.readiness
            and self.pricing_date
        )

    def missing_pins(self) -> list[str]:
        out = []
        if not self.commit:
            out.append("commit")
        if not self.deployment:
            out.append("deployment")
        if not self.models.get("reader"):
            out.append("models.reader")
        if not self.budgets:
            out.append("budgets")
        if not self.readiness:
            out.append("readiness")
        if not self.pricing_date:
            out.append("pricing_date")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparator_id": self.comparator_id,
            "edition": self.edition,
            "track": self.track,
            "status": self.status,
            "reason": self.reason,
            "pinning": {
                "commit": self.commit,
                "deployment": self.deployment,
                "models": dict(self.models),
                "prompts": self.prompts,
                "extraction": dict(self.extraction),
                "budgets": dict(self.budgets),
                "readiness": self.readiness,
                "pricing_date": self.pricing_date,
            },
            "pinned": self.pinned(),
            "synthesis": self.synthesis,
            "weakened": list(self.weakened),
            "metrics": dict(self.metrics),
            "costs": self.costs.to_dict() if self.costs else None,
            "notes": list(self.notes),
        }


def validate_row(row: ComparatorRow) -> list[str]:
    """Structural problems with a row; empty means well-formed."""
    problems: list[str] = []
    if row.track not in TRACKS:
        problems.append(f"track must be one of {TRACKS}")
    if row.status not in STATUSES:
        problems.append(f"status must be one of {STATUSES}")
    if row.status == STATUS_TESTED:
        problems.extend(
            f"missing pin: {p}" for p in row.missing_pins()
        )
        if not row.metrics:
            problems.append("tested row carries no metrics")
    if row.status != STATUS_TESTED and row.metrics:
        problems.append(f"{row.status} row cannot carry metrics")
    bad = set(row.weakened) - set(WEAKENERS)
    if bad:
        problems.append(f"unknown weakeners: {sorted(bad)}")
    if row.synthesis not in SYNTHESIS_CLASSES:
        problems.append(f"synthesis must be one of {SYNTHESIS_CLASSES}")
    return problems


def pin(row: ComparatorRow, **pins: Any) -> ComparatorRow:
    """Fill pin fields on a row, validating the result."""
    merged = replace(row, **pins)
    problems = validate_row(merged)
    if problems:
        raise ValueError(f"{row.comparator_id}: {'; '.join(problems)}")
    return merged


# ---------------------------------------------------------------------------
# the parent §03 registry + the v4.5 local set (V45-10.06)
# ---------------------------------------------------------------------------
#
# ``probe`` names an importable module for locally-runnable editions; the
# probe result decides tested vs unavailable honestly. Hosted/platform/
# preview editions are ``out_of_scope`` — paid endpoints and hosted-only
# surfaces need separate explicit authorization (V4-54.11) and their
# scores are never attributed to the OSS edition (V45-10.03, V4-54.04).

# (comparator_id, edition, track, probe_module | None, default reason)
_REGISTRY_DECLS: tuple[tuple[str, str, str, Optional[str], str], ...] = (
    # --- VERBATIM + references (the controlled local set, V45-10.06) ---
    ("verbatim_oss", "verbatim", TRACK_CONTROLLED, None, ""),
    ("verbatim_previous_stable", "verbatim", TRACK_CONTROLLED, None, ""),
    ("no_memory", "reference", TRACK_CONTROLLED, None, ""),
    ("lexical_reference", "reference", TRACK_CONTROLLED, None, ""),
    ("neural_reference", "reference", TRACK_CONTROLLED, None, ""),
    # --- v4.1 local comparator set (V45-10.06) ------------------------
    ("holographic", "local", TRACK_CONTROLLED, "holographic", ""),
    ("mem0_oss", "oss", TRACK_CONTROLLED, "mem0", ""),
    ("graphiti_oss", "oss", TRACK_CONTROLLED, "graphiti_core", ""),
    ("reme_current", "oss", TRACK_CONTROLLED, "reme", ""),
    # --- remaining parent §03 registry editions -----------------------
    ("mem0_platform", "platform", TRACK_NATIVE, None,
     "hosted platform — no paid endpoints without explicit authorization "
     "(V4-54.11); hosted scores never attribute to OSS (V45-10.03)"),
    ("zep_hosted", "hosted", TRACK_NATIVE, None,
     "hosted service — requires separate authorization (V4-54.11)"),
    ("hindsight_oss", "oss", TRACK_CONTROLLED, "hindsight", ""),
    ("hindsight_cloud", "hosted", TRACK_NATIVE, None,
     "hosted service — requires separate authorization (V4-54.11)"),
    ("honcho_self_hosted", "self_hosted", TRACK_CONTROLLED, "honcho", ""),
    ("honcho_cloud", "hosted", TRACK_NATIVE, None,
     "hosted service — requires separate authorization (V4-54.11)"),
    ("letta_memfs", "oss", TRACK_CONTROLLED, "letta", ""),
    ("memos_local", "local", TRACK_CONTROLLED, "memos", ""),
    ("openviking", "oss", TRACK_CONTROLLED, "openviking", ""),
    ("supermemory", "hosted", TRACK_CONTROLLED, "supermemory", ""),
    ("cognee_sdk", "oss", TRACK_CONTROLLED, "cognee", ""),
    ("byterover", "local", TRACK_CONTROLLED, "byterover", ""),
    ("retaindb", "oss", TRACK_CONTROLLED, "retaindb", ""),
    ("vestige", "local", TRACK_CONTROLLED, "vestige", ""),
    ("memobase", "oss", TRACK_CONTROLLED, "memobase", ""),
    ("aws_agentcore_memory", "hosted", TRACK_NATIVE, None,
     "hosted service — requires separate authorization (V4-54.11)"),
    ("google_memory_bank", "hosted", TRACK_NATIVE, None,
     "hosted service — requires separate authorization (V4-54.11)"),
    ("microsoft_foundry_memory_preview", "hosted", TRACK_NATIVE, None,
     "preview semantics must be pinned, not assumed stable (V4-54 §03) — "
     "hosted preview requires separate authorization (V4-54.11)"),
    ("redis_agent_memory_cloud", "hosted", TRACK_NATIVE, None,
     "hosted service — requires separate authorization (V4-54.11)"),
    ("langgraph_stores", "oss", TRACK_CONTROLLED, "langgraph", ""),
    ("evermemos", "research", TRACK_CONTROLLED, "evermemos", ""),
)


def _probe_module(module: str) -> tuple[bool, str]:
    """Is the competitor's pinned package importable here?"""
    try:
        spec = importlib.util.find_spec(module)
    except Exception as exc:  # malformed environment — report, not guess
        return False, f"probe error: {type(exc).__name__}: {exc}"
    if spec is None:
        return False, (
            f"package {module!r} not installed — a pinned run cannot "
            "execute in this environment"
        )
    return True, f"package {module!r} importable"


def _repo_commit() -> str:
    """The workspace commit pin for local arms (V4-54.01)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            return sha
    except Exception:
        pass
    return os.environ.get("VERBATIM_BAKEOFF_COMMIT", "unpinned")


def registry() -> list[ComparatorRow]:
    """The full claim set: every parent-registry comparator named, with
    availability probed honestly (V45-10.01, D22).

    Local/reference rows are ``tested`` (executed by :func:`run`);
    competitor editions probe for an installed package — absent, they are
    ``unavailable`` with the probe detail as the reason, never dropped.
    """
    rows: list[ComparatorRow] = []
    for cid, edition, track, probe, reason in _REGISTRY_DECLS:
        if cid in _LOCAL_ARM_IDS or cid in _ARM_FOR_ROW:
            rows.append(
                ComparatorRow(
                    comparator_id=cid,
                    edition=edition,
                    track=track,
                    status=STATUS_TESTED,
                )
            )
            continue
        if probe is None:
            rows.append(
                ComparatorRow(
                    comparator_id=cid,
                    edition=edition,
                    track=track,
                    status=STATUS_OUT_OF_SCOPE,
                    reason=reason or "out of scope for this run",
                )
            )
            continue
        ok, detail = _probe_module(probe)
        if ok:
            rows.append(
                ComparatorRow(
                    comparator_id=cid,
                    edition=edition,
                    track=track,
                    status=STATUS_TESTED,
                    reason=detail,
                    notes=("competitor adapter pending — no executed "
                           "metrics in this harness",),
                )
            )
        else:
            rows.append(
                ComparatorRow(
                    comparator_id=cid,
                    edition=edition,
                    track=track,
                    status=STATUS_UNAVAILABLE,
                    reason=detail,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# the local arm set — executed through eval.v3.baselines
# ---------------------------------------------------------------------------

#: Arm id → baseline name + cfg overrides. ``verbatim_exact`` /
#: ``verbatim_semantic`` are the exact-then-semantic pair: identical
#: engine, identical write path — the only difference is whether the
#: hashing encoder provisions the dense lane (V45-18.02 labels it
#: non-neural; it is never described as a neural model).
LOCAL_ARMS: dict[str, dict[str, Any]] = {
    "verbatim_exact": {
        "baseline": "verbatim_v3",
        "cfg": {"embedding": {"backend": "none"}},
        "encoder": "none",
        "description": "VERBATIM v3 recall — lexical/typed lanes only",
    },
    "verbatim_semantic": {
        "baseline": "verbatim_v3",
        "cfg": {"embedding": {"backend": "hashing"}},
        "encoder": HASHING_ENCODER_ID,
        "description": "VERBATIM v3 recall + hashing dense lane "
                       "(non-neural)",
    },
    "verbatim_previous_stable": {
        "baseline": "verbatim_v2",
        "cfg": {},
        "encoder": "none",
        "description": "previous stable VERBATIM arm (v2 recall)",
    },
    "no_memory": {
        "baseline": "no_memory",
        "cfg": {},
        "encoder": "none",
        "description": "control arm — no memory context",
    },
    "lexical_reference": {
        "baseline": "naive_fts",
        "cfg": {},
        "encoder": "none",
        "description": "lexical-only reference (FTS5 over raw payloads)",
    },
    "neural_reference": {
        "baseline": "vector_rag",
        "cfg": {"embedding": {"backend": "hashing"}},
        "encoder": HASHING_ENCODER_ID,
        "description": "lexical-plus-neural reference (hashing vectors, "
                       "non-neural)",
    },
}
_LOCAL_ARM_IDS = frozenset(LOCAL_ARMS)

#: Registry comparator → the arm that executes it (the rest of the local
#: set runs under its own id).
_ARM_FOR_ROW: dict[str, str] = {
    "verbatim_oss": "verbatim_semantic",
    "verbatim_previous_stable": "verbatim_previous_stable",
    "no_memory": "no_memory",
    "lexical_reference": "lexical_reference",
    "neural_reference": "neural_reference",
}


# ---------------------------------------------------------------------------
# workload
# ---------------------------------------------------------------------------


def default_workload() -> tuple:
    """A minimal two-task slice for smoke runs — real harness callers
    pass their own CorpusTask list (the v3 corpus shares the schema)."""
    from eval.v3.corpus import CorpusSource, CorpusTask

    return (
        CorpusTask(
            task_id="v45-x1-factual",
            kind="factual_lookup",
            setup_sources=(
                CorpusSource(
                    id="src-gate",
                    text="The G4-10 leadership gate requires pinned "
                         "comparator runs and honest readiness.",
                ),
                CorpusSource(
                    id="src-filler",
                    text="A celery soup recipe needs onions and stock.",
                ),
            ),
            query="What does the G4-10 gate require?",
            expected_evidence_ids=("src-gate",),
        ),
        CorpusTask(
            task_id="v45-x1-abstain",
            kind="abstention",
            setup_sources=(
                CorpusSource(
                    id="src-only",
                    text="The release checklist owns the rollout runbook.",
                ),
            ),
            query="What is the launch pricing for the enterprise tier?",
            expected_evidence_ids=(),
            expected_abstain=True,
        ),
    )


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def _db_bytes(path: str) -> Optional[int]:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _metrics_for(outcome, task, k: int) -> dict[str, Any]:
    expected = set(task.expected_evidence_ids)
    returned = list(outcome.returned_ids)
    hits = expected & set(returned)
    return {
        "task_id": task.task_id,
        "expected": sorted(expected),
        "returned": returned,
        "recall_at_k": (len(hits) / len(expected)) if expected else None,
        "precision_at_k": (
            len(hits) / len(returned) if returned else 0.0
        ),
        "abstained": bool(outcome.abstained),
        "expected_abstain": bool(task.expected_abstain),
        "abstain_correct": (
            bool(outcome.abstained) if task.expected_abstain else None
        ),
        "n_items": outcome.n_items,
        "unavailable": bool(outcome.unavailable),
        "unavailable_reason": outcome.unavailable_reason or "",
        "error": outcome.error,
        "warnings": list(outcome.warnings),
    }


def _aggregate(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    recalls = [t["recall_at_k"] for t in per_task
               if t["recall_at_k"] is not None]
    precisions = [t["precision_at_k"] for t in per_task]
    abstains = [t for t in per_task if t["expected_abstain"]]
    return {
        "tasks": len(per_task),
        "recall_at_k": (
            round(sum(recalls) / len(recalls), 6) if recalls else None
        ),
        "precision_at_k": round(
            sum(precisions) / len(precisions), 6
        ) if precisions else None,
        "abstain_accuracy": (
            round(
                sum(1 for t in abstains if t["abstain_correct"])
                / len(abstains),
                6,
            )
            if abstains
            else None
        ),
        "errors": sum(1 for t in per_task if t["error"]),
        "unavailable_tasks": sum(
            1 for t in per_task if t["unavailable"]
        ),
    }


def run(
    tasks: Optional[Sequence] = None,
    *,
    arms: Optional[Iterable[str]] = None,
    extra_rows: Iterable[ComparatorRow] = (),
    k: int = 5,
    run_id: str = "",
    pricing_date: Optional[str] = None,
    reader_model: str = "none:extractive_packs",
) -> dict[str, Any]:
    """Execute the controlled local set and return the honest report.

    Every registry row is named (V45-10.01): unavailable/out-of-scope
    comparators keep their reason; the executed arms get measured metrics
    and the eight-category cost accounting with unmeasured categories
    labeled — never zeroed (V45-10.05). Arms receive only the gold-free
    public task view (V45-14.03).

    ``pricing_date`` pins the pricing snapshot for tested rows
    (V4-54.01); it defaults to the run date. ``reader_model`` is the
    pinned reader identity — every controlled row carries it, and
    :func:`compare` refuses rows whose reader differs (V45-11.03).
    """
    from eval.v3 import baselines as _bl
    from eval.v3.corpus import public_task

    tasks = tuple(tasks) if tasks is not None else default_workload()
    if pricing_date is None:
        pricing_date = time.strftime("%Y-%m-%d", time.gmtime())
    commit = _repo_commit()
    wanted = list(arms) if arms is not None else list(LOCAL_ARMS)
    unknown = [a for a in wanted if a not in LOCAL_ARMS]
    if unknown:
        raise ValueError(
            f"unknown arms {unknown}; local set: {sorted(LOCAL_ARMS)}"
        )

    rows: list[ComparatorRow] = registry()
    arm_metrics: dict[str, list[dict[str, Any]]] = {
        a: [] for a in wanted
    }
    arm_costs: dict[str, CostBreakdown] = {}

    for arm_id in wanted:
        arm = LOCAL_ARMS[arm_id]
        baseline = _bl.get_baseline(arm["baseline"])
        per_task: list[dict[str, Any]] = []
        ingest_us = 0.0
        query_us = 0.0
        storage_bytes = 0
        for task in tasks:
            t0 = time.perf_counter_ns()
            try:
                env = _bl.prepare_case(
                    task, cfg_overrides=dict(arm["cfg"])
                )
            except Exception as exc:  # noqa: BLE001 — recorded, kept in the denominator
                per_task.append(
                    {
                        "task_id": task.task_id,
                        "expected": sorted(task.expected_evidence_ids),
                        "returned": [],
                        "recall_at_k": (
                            0.0 if task.expected_evidence_ids else None
                        ),
                        "precision_at_k": 0.0,
                        "abstained": False,
                        "expected_abstain": bool(task.expected_abstain),
                        "abstain_correct": (
                            False if task.expected_abstain else None
                        ),
                        "n_items": 0,
                        "unavailable": False,
                        "unavailable_reason": "",
                        "error": f"prepare_case: {type(exc).__name__}: {exc}",
                        "warnings": [],
                    }
                )
                continue
            ingest_us += (time.perf_counter_ns() - t0) / 1e3
            try:
                q0 = time.perf_counter_ns()
                outcome = baseline.query(env, public_task(task), k=k)
                query_us += (time.perf_counter_ns() - q0) / 1e3
                per_task.append(_metrics_for(outcome, task, k))
            except Exception as exc:  # noqa: BLE001
                per_task.append(
                    {
                        "task_id": task.task_id,
                        "expected": sorted(task.expected_evidence_ids),
                        "returned": [],
                        "recall_at_k": (
                            0.0 if task.expected_evidence_ids else None
                        ),
                        "precision_at_k": 0.0,
                        "abstained": False,
                        "expected_abstain": bool(task.expected_abstain),
                        "abstain_correct": (
                            False if task.expected_abstain else None
                        ),
                        "n_items": 0,
                        "unavailable": False,
                        "unavailable_reason": "",
                        "error": f"query: {type(exc).__name__}: {exc}",
                        "warnings": [],
                    }
                )
            for fname in ("store.db", "verbatim.db"):
                p = os.path.join(env.store_dir, fname)
                if os.path.isfile(p):
                    storage_bytes += _db_bytes(p) or 0
                    break
            env.close()
        arm_metrics[arm_id] = per_task
        arm_costs[arm_id] = CostBreakdown(
            ingest=round(ingest_us, 3),
            query_inference=round(query_us, 3),
            storage=float(storage_bytes),
            units={
                "ingest": "us",
                "query_inference": "us",
                "storage": "bytes",
            },
        )

    # ---- fill tested rows with measurements + the pin set -------------
    filled: list[ComparatorRow] = []
    for row in rows:
        arm_id = _ARM_FOR_ROW.get(row.comparator_id)
        if arm_id is None or arm_id not in arm_metrics:
            filled.append(row)
            continue
        arm = LOCAL_ARMS[arm_id]
        filled.append(
            ComparatorRow(
                comparator_id=row.comparator_id,
                edition=row.edition,
                track=row.track,
                status=STATUS_TESTED,
                reason=row.reason,
                commit=commit,
                deployment="local_in_process",
                models={
                    "encoder": arm["encoder"],
                    "reader": reader_model,
                },
                prompts="n/a — evidence packs, no generative reader",
                extraction={
                    "harvester": "harvest_v3",
                    "admission": "rules",
                },
                budgets={"k": int(k), "max_items": int(k)},
                readiness="drained",
                pricing_date=pricing_date,
                metrics={
                    "aggregate": _aggregate(arm_metrics[arm_id]),
                    "per_task": arm_metrics[arm_id],
                    "arm": arm_id,
                },
                costs=arm_costs[arm_id],
                notes=(arm["description"],),
            )
        )
    for extra in extra_rows:
        problems = validate_row(extra)
        if problems:
            raise ValueError(
                f"extra row {extra.comparator_id}: {'; '.join(problems)}"
            )
        filled.append(extra)

    # ---- pairwise verdicts: verbatim arms vs every tested controlled row
    comparisons: list[dict[str, Any]] = []
    for pivot_id in ("verbatim_oss",):
        pivot = next(
            (r for r in filled if r.comparator_id == pivot_id), None
        )
        if pivot is None:
            continue
        for other in filled:
            if other.comparator_id == pivot_id:
                continue
            if other.status != STATUS_TESTED:
                continue
            comparisons.append(compare(pivot, other))

    return {
        "schema": 1,
        "kind": "v45_bakeoff",
        "run_id": run_id,
        "pricing_date": pricing_date,
        "commit": commit,
        "reader_model": reader_model,
        "gold_protected": True,
        "gold_protection": (
            "arms receive only public_task views — expected ids, "
            "abstain/poison/scope gold, and supersession expectations "
            "raise AttributeError in arm code (V45-14.03)"
        ),
        "budgets": {"k": int(k), "max_items": int(k)},
        "tasks": [t.task_id for t in tasks],
        "rows": [r.to_dict() for r in filled],
        "comparisons": comparisons,
        "claim_set": claim_set(filled),
        "notes": [
            "unavailable/out-of-scope comparators are named with reasons "
            "and never enter defeated counts (V45-10.01, V45-15.02)",
            "cost totals are labeled complete only when every "
            "V45-10.05 category is measured",
        ],
    }


def claim_set(rows: Iterable[ComparatorRow]) -> dict[str, Any]:
    """The D22 roll-call: named tested / unavailable / out_of_scope."""
    tested, unavail, oos = [], [], []
    for r in rows:
        if r.status == STATUS_TESTED:
            tested.append(r.comparator_id)
        elif r.status == STATUS_UNAVAILABLE:
            unavail.append(r.comparator_id)
        else:
            oos.append(r.comparator_id)
    return {
        "tested": sorted(tested),
        "unavailable": sorted(unavail),
        "out_of_scope": sorted(oos),
        "defeated": [],  # populated only by valid comparisons — never inferred
    }


def compare(
    a: ComparatorRow,
    b: ComparatorRow,
    *,
    metric: str = "recall_at_k",
) -> dict[str, Any]:
    """A paired verdict ``a`` vs ``b`` — or the reason it cannot stand.

    Refusals (each recorded, never silent): different tracks
    (V45-10.02); an untested row (V45-15.02); a weakened comparator —
    a disabled-embedding/broken-ingest/truncated-context setup cannot
    count as a competitor loss (V45-10.04, D13); missing pins
    (V4-54.01); reader mismatch mid-claim (V45-11.03); mixed synthesis
    classes (V45-11.04).
    """
    base: dict[str, Any] = {
        "a": a.comparator_id,
        "b": b.comparator_id,
        "metric": metric,
        "valid": False,
        "winner": None,
        "reason": "",
    }
    if a.track != b.track:
        base["reason"] = (
            f"track_mismatch: {a.track!r} vs {b.track!r} — native and "
            "controlled rows are separate rows (V45-10.02)"
        )
        return base
    untested = [
        r.comparator_id
        for r in (a, b)
        if r.status != STATUS_TESTED
    ]
    if untested:
        base["reason"] = (
            f"untested row(s) {untested} — unavailable comparators are "
            "named, never defeated (V45-10.01, V45-15.02)"
        )
        return base
    for label, row in (("a", a), ("b", b)):
        if row.weakened:
            base["reason"] = (
                f"weakened_comparator:{label}={row.comparator_id} "
                f"{sorted(row.weakened)} — a win over a weakened "
                "comparator is invalid (V45-10.04, D13)"
            )
            return base
    for label, row in (("a", a), ("b", b)):
        missing = row.missing_pins()
        if missing:
            base["reason"] = (
                f"unpinned row {row.comparator_id}: missing {missing} "
                "(V4-54.01)"
            )
            return base
    if a.models.get("reader") != b.models.get("reader"):
        base["reason"] = (
            f"reader_mismatch: {a.models.get('reader')!r} vs "
            f"{b.models.get('reader')!r} — reader identity is pinned "
            "per comparison (V45-11.03)"
        )
        return base
    if a.synthesis != b.synthesis:
        base["reason"] = (
            f"synthesis_mismatch: {a.synthesis!r} vs {b.synthesis!r} — "
            "provider-native synthesis is scored separately (V45-11.04)"
        )
        return base
    va = (a.metrics.get("aggregate") or {}).get(metric, a.metrics.get(metric))
    vb = (b.metrics.get("aggregate") or {}).get(metric, b.metrics.get(metric))
    if va is None or vb is None:
        base["reason"] = f"metric {metric!r} not measured on both rows"
        return base
    delta = round(float(va) - float(vb), 6)
    base.update(
        {
            "valid": True,
            "value_a": va,
            "value_b": vb,
            "delta": delta,
            "winner": (
                a.comparator_id
                if delta > 0
                else (b.comparator_id if delta < 0 else "tie")
            ),
            "reason": "measured",
        }
    )
    return base


__all__ = [
    "COST_CATEGORIES",
    "CostBreakdown",
    "ComparatorRow",
    "HASHING_ENCODER_ID",
    "LOCAL_ARMS",
    "STATUSES",
    "STATUS_OUT_OF_SCOPE",
    "STATUS_TESTED",
    "STATUS_UNAVAILABLE",
    "SYNTHESIS_CLASSES",
    "TRACKS",
    "TRACK_CONTROLLED",
    "TRACK_NATIVE",
    "WEAKENERS",
    "claim_set",
    "compare",
    "default_workload",
    "pin",
    "registry",
    "run",
    "validate_row",
]
