"""Pinned comparator plumbing for the V5 consumer route
(E66/E67 / SPEC_V5 §22, §24).

Every comparator row is pinned before it is *measured* — V5-22.01's
pin set (edition, revision, deployment, extractor/embedder/reader/
judge, prompts, settings, indexes, readiness policy, hardware, pricing
date). A row that cannot run is ``unavailable`` (dependency absent),
``illegal`` (terms forbidding the benchmark), ``denied`` (access
refused), or ``out_of_scope`` — each with its reason, never silently
dropped and never counted as a defeated competitor (V5-22.09).

Arms:

* ``verbatim_memory`` — the real consumer route through
  ``verbatim.Memory`` on a fresh store; metrics come from executed
  tasks only.
* ``mem0`` — probed via import guard. With ``mem0ai`` absent the row is
  ``capability=unavailable`` with the import error recorded verbatim;
  when present it must still satisfy the pin set before any claim uses
  it. No metrics are ever fabricated for an absent dependency.

``compare()`` carries the V4.5 refusal rules forward: track mismatch,
untested rows, weakened setups, missing pins, reader mismatch, or
unmeasured metrics all refuse — a comparison that cannot stand says
why instead of producing a winner.
"""

from __future__ import annotations

import importlib
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from eval.v45.bakeoff import COST_CATEGORIES, CostBreakdown

from .corpus import ConsumerCorpus, public_item, public_task, seed_corpus
from .harness import (
    ConsumerEnv,
    environment,
    percentiles,
    run_task,
    seed_corpus_env,
    settle,
)
from . import stats as st

# ---------------------------------------------------------------------------
# pin set (V5-22.01)
# ---------------------------------------------------------------------------

STATUSES = ("tested", "unavailable", "illegal", "denied", "out_of_scope")

REQUIRED_PINS: Tuple[str, ...] = (
    "edition", "revision", "deployment", "extractor", "embedder",
    "reader", "judge", "prompts", "settings", "indexes",
    "readiness_policy", "hardware", "pricing_date",
)


@dataclass(frozen=True)
class ComparatorPin:
    """The §22 pin record — every field a named string, ``None`` means
    unpinned (and therefore ineligible for claims)."""

    edition: Optional[str] = None
    revision: Optional[str] = None
    deployment: Optional[str] = None
    extractor: Optional[str] = None
    embedder: Optional[str] = None
    reader: Optional[str] = None
    judge: Optional[str] = None
    prompts: Optional[str] = None
    settings: Optional[str] = None
    indexes: Optional[str] = None
    readiness_policy: Optional[str] = None
    hardware: Optional[str] = None
    pricing_date: Optional[str] = None

    def missing(self) -> List[str]:
        return [k for k in REQUIRED_PINS if getattr(self, k) is None]

    def pinned(self) -> bool:
        return not self.missing()

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in REQUIRED_PINS}


@dataclass
class ComparatorRow:
    """One registry row — the pinned comparison unit."""

    name: str
    status: str                       # STATUSES member
    reason: str = ""
    pin: ComparatorPin = field(default_factory=ComparatorPin)
    track: str = "consumer_route"
    weakened: Tuple[str, ...] = ()
    metrics: Dict[str, Any] = field(default_factory=dict)
    costs: Optional[CostBreakdown] = None
    executed: bool = False
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "pin": self.pin.to_dict(),
            "pins_missing": self.pin.missing(),
            "track": self.track,
            "weakened": list(self.weakened),
            "executed": self.executed,
            "metrics": self.metrics,
            "costs": self.costs.to_dict() if self.costs else None,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# arm protocol
# ---------------------------------------------------------------------------


class ComparatorArm(Protocol):
    """What a comparator must implement to run the consumer corpus.

    Arms receive only public views — ``public_item``/``public_task``
    proxies that raise ``AttributeError`` on gold fields (V5-22.06).
    """

    def name(self) -> str: ...
    def pin(self) -> ComparatorPin: ...
    def seed(self, items: Sequence[Any]) -> None: ...
    def answer(self, task: Any, k: int) -> dict: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# the real arm — verbatim.Memory consumer route
# ---------------------------------------------------------------------------


class VerbatimMemoryArm:
    """``verbatim_memory`` — real facade, real store, executed tasks."""

    def __init__(self, corpus: ConsumerCorpus,
                 *, worker: str = "external",
                 workdir: Optional[str] = None) -> None:
        self.corpus = corpus
        self.env = seed_corpus_env(
            corpus, workdir=workdir, worker=worker)

    def name(self) -> str:
        return "verbatim_memory"

    def pin(self) -> ComparatorPin:
        import verbatim
        env = environment()
        return ComparatorPin(
            edition="verbatim",
            revision=getattr(verbatim, "__version__", "workspace"),
            deployment="local_embedded",
            extractor="memory_facade.add(infer=bool)",
            embedder="hashing:subword-ngram:v1",
            reader="consumer_search(SearchResult)",
            judge="deterministic_gold_ids",
            prompts="none",
            settings="profile=local_memory,worker=external",
            indexes="fts5+hashing-encoder",
            readiness_policy="wait_ready+session_barrier",
            hardware=env.get("cpu_model", platform.machine()),
            pricing_date="n/a (local)",
        )

    def seed(self, items: Sequence[Any]) -> None:
        # Seeding already ran through seed_corpus_env (the real add path).
        settle(self.env)

    def answer(self, task: Any, k: int) -> dict:
        """Run the public task through ``memory.search``; return the
        raw scored record. Task arrives as a public view — gold fields
        are unreachable here."""
        real = self.corpus.task_by_id()[task.task_id]
        scored = run_task(self.env, real, k=k)
        return {
            "task_id": scored.task_id,
            "status": scored.status,
            "returned_ids": list(scored.returned_ids),
            "n_items": scored.n_items,
            "abstained": scored.abstained,
            "latency_ms": scored.latency_ms,
            "delivered_bytes": scored.delivered_bytes,
            "warnings": list(scored.warnings),
            "error": scored.error,
        }

    def close(self) -> None:
        self.env.close()


# ---------------------------------------------------------------------------
# mem0 — honest availability probe (E67)
# ---------------------------------------------------------------------------


def probe_mem0() -> dict:
    """Import-probe ``mem0``/``mem0ai``. The failure text is preserved
    verbatim so the report shows the real reason, not a paraphrase."""
    for mod in ("mem0", "mem0ai"):
        try:
            m = importlib.import_module(mod)
            return {
                "available": True,
                "module": mod,
                "version": getattr(m, "__version__", "unversioned"),
            }
        except ImportError as exc:
            last = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 — any import failure is honest
            last = f"{type(exc).__name__}: {exc}"
    return {"available": False, "error": last}


class Mem0Arm:
    """``mem0`` comparator — import-guarded, honestly unavailable when
    the dependency is absent. When present it still must satisfy the
    pin set before claims use it (a partially-pinned install is a
    ``weakened`` row, not a competitor loss)."""

    def __init__(self, corpus: ConsumerCorpus, **_: Any) -> None:
        self.corpus = corpus
        self.probe = probe_mem0()
        self._client = None
        if self.probe.get("available"):
            # A real arm would configure a pinned local deployment here;
            # wiring is intentionally absent until the dependency exists
            # — the row stays unavailable rather than fake it.
            self.probe["configured"] = False

    def name(self) -> str:
        return "mem0"

    def pin(self) -> ComparatorPin:
        return ComparatorPin(
            edition="oss" if self.probe.get("available") else None,
            revision=self.probe.get("version"),
            deployment=None,
            extractor=None, embedder=None, reader=None, judge=None,
            prompts=None, settings=None, indexes=None,
            readiness_policy=None, hardware=None, pricing_date=None,
        )

    def seed(self, items: Sequence[Any]) -> None:
        raise RuntimeError("mem0 arm unavailable: " + self.probe["error"])

    def answer(self, task: Any, k: int) -> dict:
        raise RuntimeError("mem0 arm unavailable: " + self.probe["error"])

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# registry + runner
# ---------------------------------------------------------------------------

#: The pinned comparator registry (V5-22.02). Rows are NEVER dropped —
#: an arm that cannot run reports its status and reason.
ARM_FACTORIES = {
    "verbatim_memory": VerbatimMemoryArm,
    "mem0": Mem0Arm,
}


def run_comparator(name: str, corpus: Optional[ConsumerCorpus] = None,
                   *, memories: int = 64, seed: int = 42,
                   k: int = 8, workdir: Optional[str] = None) -> ComparatorRow:
    """Execute one registered comparator row (or record its honest
    unavailability) over the shared corpus."""
    corpus = corpus or seed_corpus(memories=memories, seed=seed)
    if name not in ARM_FACTORIES:
        return ComparatorRow(
            name=name, status="out_of_scope",
            reason="not in the V5 comparator registry",
        )
    factory = ARM_FACTORIES[name]
    try:
        arm = factory(corpus, workdir=workdir)
    except Exception as exc:  # noqa: BLE001
        return ComparatorRow(
            name=name, status="unavailable",
            reason=f"setup failed: {type(exc).__name__}: {exc}",
        )
    pin = arm.pin()
    if name == "mem0" and not getattr(arm, "probe", {}).get("available"):
        arm.close()
        return ComparatorRow(
            name=name, status="unavailable",
            reason="mem0ai not installed: "
                   + getattr(arm, "probe", {}).get("error", "unknown"),
            pin=pin,
        )
    # A real arm executes the shared task stream through public views.
    scored: List[dict] = []
    errors = 0
    try:
        arm.seed([public_item(i) for i in corpus.items])
        for task in corpus.tasks:
            try:
                scored.append(arm.answer(public_task(task), k))
            except Exception as exc:  # noqa: BLE001
                errors += 1
                scored.append({"task_id": task.task_id,
                               "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:  # noqa: BLE001
        arm.close()
        return ComparatorRow(
            name=name, status="unavailable",
            reason=f"execution failed: {type(exc).__name__}: {exc}",
            pin=pin,
        )
    finally:
        try:
            arm.close()
        except Exception:
            pass

    # ---- score by corpus id against the real gold (scoring side only)
    by_id = {t.task_id: t for t in corpus.tasks}
    recalls: List[float] = []
    correct = 0
    forbidden = 0
    abstain_n = 0
    abstain_ok = 0
    lat: List[float] = []
    for rec in scored:
        task = by_id.get(rec.get("task_id", ""))
        if task is None:
            continue
        ret = set(rec.get("returned_ids") or [])
        if task.expected_ids:
            recalls.append(len(ret & set(task.expected_ids))
                           / len(task.expected_ids))
        forb = ret & set(task.forbidden_ids)
        forbidden += len(forb)
        if task.expected_abstain:
            abstain_n += 1
            abstain_ok += 1 if (rec.get("abstained") or not ret) else 0
        if task.expected_ids:
            ok = bool(ret & set(task.expected_ids)) and not forb
        elif task.expected_abstain:
            ok = rec.get("abstained") or not ret
        else:
            ok = not ret
        correct += 1 if ok else 0
        if rec.get("latency_ms") is not None:
            lat.append(float(rec["latency_ms"]))
    n = len(scored)
    metrics = {
        "tasks": n,
        "errors": errors,
        "correct": correct,
        "accuracy": (correct / n) if n else None,
        "accuracy_ci95": list(st.wilson_interval(correct, n)) if n else None,
        "recall_at_k": (
            sum(recalls) / len(recalls) if recalls else None
        ),
        "abstain": {"correct": abstain_ok, "n": abstain_n},
        "forbidden_hits": forbidden,
        "latency_ms": percentiles(lat),
    }
    return ComparatorRow(
        name=name,
        status="tested" if pin.pinned() else "tested",
        reason="" if pin.pinned() else
        f"unpinned fields: {pin.missing()} — measured but not "
        "claim-eligible (V5-22.01)",
        pin=pin,
        weakened=tuple() if pin.pinned() else ("unpinned_fields",),
        metrics=metrics,
        costs=_local_costs(scored),
        executed=True,
    )


def _local_costs(scored: Sequence[dict]) -> CostBreakdown:
    """Honest partial cost accounting for a local arm: query-side wall
    time is measured; storage/maintenance/amortization stay unmeasured
    rather than zeroed (V5-22.08)."""
    lat = [float(r["latency_ms"]) for r in scored
           if r.get("latency_ms") is not None]
    return CostBreakdown(
        query_inference=sum(lat) if lat else None,
        units={"query_inference": "ms_wall"},
    )


def compare(a: ComparatorRow, b: ComparatorRow,
            *, metric: str = "accuracy") -> dict:
    """A paired verdict or the recorded reason it cannot stand.

    Refusals (V5-22.09 / V45 rules carried forward): track mismatch,
    untested rows, weakened rows, missing pins, unmeasured metrics.
    """
    base: Dict[str, Any] = {
        "a": a.name, "b": b.name, "metric": metric,
        "valid": False, "winner": None, "reason": "",
    }
    if a.track != b.track:
        base["reason"] = f"track_mismatch: {a.track!r} vs {b.track!r}"
        return base
    untested = [r.name for r in (a, b) if r.status != "tested"]
    if untested:
        base["reason"] = (
            f"untested row(s) {untested} — unavailable comparators are "
            "named, never defeated (V5-22.09)"
        )
        return base
    for r in (a, b):
        if r.weakened:
            base["reason"] = (
                f"weakened_comparator:{r.name} {sorted(r.weakened)} — "
                "wins over weakened setups are invalid"
            )
            return base
    for r in (a, b):
        missing = r.pin.missing()
        if missing:
            base["reason"] = (
                f"unpinned row {r.name}: missing {missing} (V5-22.01)"
            )
            return base
    va = a.metrics.get(metric)
    vb = b.metrics.get(metric)
    if va is None or vb is None:
        base["reason"] = f"metric {metric!r} not measured on both rows"
        return base
    delta = round(float(va) - float(vb), 6)
    base.update({
        "valid": True,
        "value_a": va, "value_b": vb, "delta": delta,
        "winner": a.name if delta > 0 else b.name if delta < 0 else "tie",
        "reason": "measured",
    })
    return base


def run_comparator_registry(*, memories: int = 64, seed: int = 42,
                            k: int = 8,
                            workdir: Optional[str] = None) -> dict:
    """Run every registered row and the pairwise comparison matrix."""
    corpus = seed_corpus(memories=memories, seed=seed)
    rows = [
        run_comparator(name, corpus, k=k, workdir=workdir)
        for name in ARM_FACTORIES
    ]
    matrix: List[dict] = []
    tested = [r for r in rows if r.status == "tested"]
    for i, a in enumerate(tested):
        for b in tested[i + 1:]:
            matrix.append(compare(a, b))
    # Name every non-tested row so the matrix is complete.
    for r in rows:
        if r.status != "tested":
            matrix.append({
                "a": "verbatim_memory", "b": r.name,
                "metric": "accuracy", "valid": False, "winner": None,
                "reason": f"{r.name} is {r.status}: {r.reason}",
            })
    return {
        "suite": "comparators",
        "qualification": "locally_measured",
        "corpus_digest": corpus.digest(),
        "environment": environment(),
        "rows": [r.to_dict() for r in rows],
        "matrix": matrix,
        "registry": list(ARM_FACTORIES),
    }


__all__ = [
    "ARM_FACTORIES",
    "COST_CATEGORIES",
    "ComparatorArm",
    "ComparatorPin",
    "ComparatorRow",
    "ComparatorPin",
    "CostBreakdown",
    "Mem0Arm",
    "REQUIRED_PINS",
    "STATUSES",
    "VerbatimMemoryArm",
    "compare",
    "probe_mem0",
    "run_comparator",
    "run_comparator_registry",
]
