"""The four v3 evaluation suites (SPEC_V3 §53): definitions, case schema,
runner protocol, result records, and stdlib-only interval math.

Suite intent, per the §53 workload table:

* ``conformance`` — requirement-tagged unit/integration cases. Each case
  names the ``V3-NN.MM`` requirement ids it exercises (markers declare
  links, never completion — §56.01) and asserts observable behavior
  through a real store, not import or source-string checks (§56.02).
* ``retrieval_quality`` — recall/precision/MRR/latency over a *declared*
  corpus: the corpus identity, seed, and digest are part of the manifest
  so denominators are auditable (§53.01). Failed or errored queries stay
  in every denominator (§53.05).
* ``paired_usefulness`` — paired task executions across arms
  (``no_memory`` floor, ``flat_retrieval``/BM25 earned-complexity bar,
  ``memory`` under test; the §53 control table's ``oracle`` arm is a
  diagnostic ceiling and is never a claimed result). This is the honest
  comparison §53/§54.13 demand: actual executed arms with common starting
  snapshots and matched budgets (§53.14) — **not** shadow logs.
* ``security_privacy`` — poisoning lifecycle, quarantine, deletion
  closure, and authorization cases (§33–§36), each tagged positive /
  denied / boundary as the §56.01 evidence matrix requires.

Result semantics (the honesty contract, §54.11):

* ``pass`` requires the expectation to hold *and* the declared minimum
  sample support; missing support is ``inconclusive`` — never ``pass``.
* ``error`` keeps the case in the denominator; nothing is dropped for
  looking bad (§53.05).
* ``skipped`` means the case was declared but never executed; it reports
  separately and still blocks suite-level ``pass`` when required, because
  missing required evidence is not completion (§56.01).

Intervals are Wilson score intervals for binomial proportions and a
discordant-pair interval for paired binary outcomes — pure ``math``,
no scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple, runtime_checkable

# ---------------------------------------------------------------------------
# suite identifiers (SPEC_V3 §53 table)
# ---------------------------------------------------------------------------

CONFORMANCE = "conformance"
RETRIEVAL_QUALITY = "retrieval_quality"
PAIRED_USEFULNESS = "paired_usefulness"
SECURITY_PRIVACY = "security_privacy"

SUITE_IDS: Tuple[str, ...] = (
    CONFORMANCE,
    RETRIEVAL_QUALITY,
    PAIRED_USEFULNESS,
    SECURITY_PRIVACY,
)

SUITE_DESCRIPTIONS = {
    CONFORMANCE: (
        "Requirement-tagged unit/integration cases asserting observable "
        "behavior through public entry points (§56.01/§56.02, §55.01)."
    ),
    RETRIEVAL_QUALITY: (
        "Recall, precision, MRR, and latency over a declared corpus with "
        "frozen manifest, gold denominators, and no dropped failures "
        "(§53.01, §53.03, §53.05)."
    ),
    PAIRED_USEFULNESS: (
        "Paired task executions: memory vs no-memory vs simple-retrieval "
        "control arms on common snapshots with matched budgets — executed "
        "comparisons, not shadow logs (§53.04, §53.14, §54.13)."
    ),
    SECURITY_PRIVACY: (
        "Poisoning lifecycle, quarantine, deletion-closure, and "
        "authorization cases with stage denominators and benign twins "
        "(§33–§36, §54.12, G1/G9)."
    ),
}

#: The §53 control-arm table. ``oracle`` is a diagnostic ceiling — it may
#: appear in reports but must never be presented as a system result.
CONTROL_ARMS: Tuple[str, ...] = (
    "no_memory",
    "full_history",
    "bm25_only",
    "dense_only",
    "hybrid_rrf_no_controller",
    "memory",
    "oracle",
)

#: Arms whose presence makes a paired comparison *executed* rather than
#: observational. Absent these, competitor/claim readiness is
#: ``not_estimable`` (§54.13: no shadow-run, source-count, or spec-count
#: evidence satisfies an empirical gate).
PAIRED_REQUIRED_ARMS: Tuple[str, ...] = ("no_memory", "memory")
SIMPLE_RETRIEVAL_ARMS: Tuple[str, ...] = ("bm25_only", "flat_retrieval")


# ---------------------------------------------------------------------------
# case schema
# ---------------------------------------------------------------------------


class Outcome(str, Enum):
    """First-class case outcomes. ``inconclusive`` is a result, not an
    omission — a case without statistical support can never be `pass`
    (§54.11)."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"
    SKIPPED = "skipped"


#: Outcomes that keep a case in the executed denominator (§53.05).
EXECUTED_OUTCOMES = frozenset(
    {Outcome.PASS, Outcome.FAIL, Outcome.INCONCLUSIVE, Outcome.ERROR}
)


@dataclass(frozen=True)
class Case:
    """One declared evaluation case.

    ``input`` and ``expected`` are JSON-serializable payloads interpreted
    by the suite's runner — the schema fixes *shape*, not semantics, so a
    frozen manifest can pin them (§53.01, §55.02).

    ``requirement_ids`` declares the V3-NN.MM links; per §56.01 the link
    is a marker, and completion is decided by the evidence ledger, never
    by this field alone.

    ``min_support`` is the smallest executed-sample count at which the
    case may report ``pass`` (§54.11). For a single deterministic
    assertion the default 1 is honest; empirical cases declare more.

    ``tags`` carry the evidence-matrix axes (§56.01): ``positive``,
    ``denied``, ``boundary``, ``restart``, ``migration``,
    ``profile:<name>``, ``capability:<name>``, ``arm:<name>``,
    ``benign_twin``. ``required=False`` marks advisory cases that inform
    but never block the suite verdict (diagnostic/oracle arms).
    """

    case_id: str
    suite: str
    requirement_ids: Tuple[str, ...] = ()
    input: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)
    tags: Tuple[str, ...] = ()
    required: bool = True
    min_support: int = 1

    def __post_init__(self) -> None:
        if self.suite not in SUITE_IDS:
            raise ValueError(f"unknown suite {self.suite!r}")
        if not self.case_id:
            raise ValueError("case_id required")
        if isinstance(self.min_support, bool) or self.min_support < 1:
            raise ValueError("min_support must be >= 1")

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "suite": self.suite,
            "requirement_ids": list(self.requirement_ids),
            "input": self.input,
            "expected": self.expected,
            "tags": list(self.tags),
            "required": self.required,
            "min_support": self.min_support,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        return cls(
            case_id=d["case_id"],
            suite=d["suite"],
            requirement_ids=tuple(d.get("requirement_ids", ())),
            input=dict(d.get("input", {})),
            expected=dict(d.get("expected", {})),
            tags=tuple(d.get("tags", ())),
            required=bool(d.get("required", True)),
            min_support=int(d.get("min_support", 1)),
        )


@dataclass(frozen=True)
class CaseResult:
    """The recorded outcome of one executed case.

    ``support`` is the executed-sample count behind the outcome (e.g.
    queries answered, paired trials completed); the harness compares it
    to the case's ``min_support`` when deciding ``inconclusive``.
    ``metrics`` carries per-case measurements (rank, latency_ns, arm
    results, attack-stage flags) for suite-level aggregation.
    """

    case_id: str
    outcome: Outcome
    support: int = 1
    detail: str = ""
    metrics: dict = field(default_factory=dict)
    latency_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, Outcome):
            object.__setattr__(self, "outcome", Outcome(self.outcome))

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "outcome": self.outcome.value,
            "support": self.support,
            "detail": self.detail,
            "metrics": self.metrics,
            "latency_ns": self.latency_ns,
        }


@dataclass(frozen=True)
class ArmResult:
    """One arm of a paired case execution (§53.14). ``success`` stays
    three-valued: ``None`` means the arm did not run or produced no
    countable outcome — it is *not* a failure and not a success."""

    arm: str
    success: Optional[bool]
    detail: str = ""
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# runner protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CaseRunner(Protocol):
    """Execute one case against the prepared context.

    The context is whatever ``Suite.prepare`` returned — a store, an
    engine, or a per-arm factory — depending on the suite. A runner
    returns a :class:`CaseResult` (or ``bool`` for simple assertions);
    it must never raise for expected failures: an exception is recorded
    as ``error`` by the harness and stays in the denominator (§53.05).
    """

    def __call__(self, case: Case, ctx: Any) -> CaseResult: ...


# ---------------------------------------------------------------------------
# suite definition
# ---------------------------------------------------------------------------


@dataclass
class Suite:
    """A declared suite: cases plus the runner protocol that interprets
    them and the optional ``prepare``/``aggregate`` hooks.

    ``prepare(store_factory) -> ctx`` runs once per suite run. A
    conformance suite calls the factory for a fresh store; a retrieval
    suite uses it to build and populate an engine; a paired suite keeps
    the factory itself so each arm gets an identical fresh store (§53.14
    common starting snapshots).

    ``aggregate(results) -> dict`` adds suite-specific metrics (MRR,
    latency percentiles, paired deltas) to the report; the harness always
    computes the base denominators itself.
    """

    suite_id: str
    name: str
    requirement_ids: Tuple[str, ...]
    cases: Tuple[Case, ...]
    runner: CaseRunner
    prepare: Optional[Callable[[Callable[[], Any]], Any]] = None
    aggregate: Optional[Callable[[Sequence[CaseResult]], dict]] = None
    #: Minimum executed cases for the suite to report anything but
    #: ``inconclusive`` (§54.11 missing-support rule).
    min_support: int = 1
    #: Frozen-manifest digest recorded in the report (§53.01); '' when
    #: the suite is declared but not yet manifest-frozen.
    manifest_digest: str = ""

    def __post_init__(self) -> None:
        if self.suite_id not in SUITE_IDS:
            raise ValueError(f"unknown suite {self.suite_id!r}")
        if self.min_support < 1:
            raise ValueError("min_support must be >= 1")
        for c in self.cases:
            if c.suite != self.suite_id:
                raise ValueError(
                    f"case {c.case_id} belongs to suite {c.suite}, "
                    f"not {self.suite_id}"
                )

    @property
    def required_cases(self) -> Tuple[Case, ...]:
        return tuple(c for c in self.cases if c.required)


# ---------------------------------------------------------------------------
# interval math — stdlib only (§54.11: report uncertainty, never a bare
# point estimate)
# ---------------------------------------------------------------------------

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_interval(
    successes: int, n: int, z: float = Z_95
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(0.0, 0.0)`` for an empty denominator — an interval with no
    support is undefined, and callers must treat that as ``inconclusive``
    rather than as ``1.0`` or ``0.0`` evidence (§54.11).
    """
    if n <= 0 or successes < 0 or successes > n:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def wilson_lower_bound(successes: int, n: int, z: float = Z_95) -> float:
    """One-sided Wilson lower bound — the form §54's automation gates use
    ("lower 95% bound ≥ 0.98")."""
    return wilson_interval(successes, n, z)[0]


def wilson_upper_bound(successes: int, n: int, z: float = Z_95) -> float:
    """One-sided Wilson upper bound (§54 "upper 95% bound ≤ 0.05")."""
    return wilson_interval(successes, n, z)[1]


def paired_delta_interval(
    trials: int, wins_memory: int, wins_control: int, z: float = Z_95
) -> Tuple[float, float, float]:
    """Interval on the paired success difference (memory − control).

    ``wins_memory`` counts matched trials where the memory arm succeeded
    and the control failed (paired benefit, §54.10); ``wins_control`` is
    the reverse (paired harm). The estimate is
    ``(wins_memory − wins_control) / trials``; the interval uses the
    discordant-pair variance ``Var = (b + c − (b−c)²/n) / n²``, which is
    the standard paired-proportion form and needs no scipy.

    Returns ``(estimate, low, high)``; ``(0.0, 0.0, 0.0)`` for no trials —
    again, undefined intervals are inconclusive, never zero evidence.
    """
    if trials <= 0 or wins_memory < 0 or wins_control < 0:
        return (0.0, 0.0, 0.0)
    b, c = float(wins_memory), float(wins_control)
    n = float(trials)
    est = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / (n * n)
    se = math.sqrt(max(var, 0.0))
    return (est, max(-1.0, est - z * se), min(1.0, est + z * se))


def percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of pre-sorted values; 0.0 when empty."""
    vals = list(sorted_vals)
    if not vals:
        return 0.0
    i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return float(vals[i])


# ---------------------------------------------------------------------------
# aggregation helpers shared by all suites
# ---------------------------------------------------------------------------


def outcome_counts(results: Sequence[CaseResult]) -> dict:
    """Denominator bookkeeping (§53.05): every executed outcome stays in
    its count; skipped is reported but not part of the executed base."""
    counts = {o.value: 0 for o in Outcome}
    for r in results:
        counts[r.outcome.value] += 1
    counts["declared"] = len(results)
    counts["executed"] = sum(counts[o.value] for o in EXECUTED_OUTCOMES)
    return counts


def base_metrics(results: Sequence[CaseResult]) -> dict:
    """Pass rate and its Wilson interval over the executed denominator.

    ``skipped`` cases never enter the rate; ``error`` and ``inconclusive``
    always do — they are evidence of what the run could not show.
    """
    counts = outcome_counts(results)
    executed = counts["executed"]
    lo, hi = wilson_interval(counts["pass"], executed)
    return {
        "counts": counts,
        "pass_rate": (counts["pass"] / executed) if executed else None,
        "ci95": [round(lo, 4), round(hi, 4)] if executed else None,
    }


def suite_verdict(
    suite: Suite, results: Sequence[CaseResult]
) -> str:
    """Honest suite verdict (§54.02/§54.11, §56.01).

    ``fail`` — any required case failed or errored.
    ``inconclusive`` — required cases remain inconclusive or skipped, or
    the executed count lacks the suite's declared minimum support.
    ``pass`` — every declared required case executed and passed with
    sufficient per-case support.
    """
    by_id = {r.case_id: r for r in results}
    required = suite.required_cases
    counts = outcome_counts(results)
    if counts["executed"] < suite.min_support:
        return "inconclusive"
    saw_fail = False
    saw_gap = False
    for case in required:
        r = by_id.get(case.case_id)
        if r is None or r.outcome in (Outcome.SKIPPED, Outcome.INCONCLUSIVE):
            saw_gap = True
            continue
        if r.outcome in (Outcome.FAIL, Outcome.ERROR):
            saw_fail = True
            continue
        if r.support < case.min_support:
            saw_gap = True
    if saw_fail:
        return "fail"
    if saw_gap:
        return "inconclusive"
    return "pass"


# ---------------------------------------------------------------------------
# suite constructors — the declared case schema per suite (§53)
# ---------------------------------------------------------------------------


def make_suite(
    suite_id: str,
    *,
    name: str,
    cases: Sequence[Case],
    runner: CaseRunner,
    requirement_ids: Sequence[str] = (),
    prepare: Optional[Callable[[Callable[[], Any]], Any]] = None,
    aggregate: Optional[Callable[[Sequence[CaseResult]], dict]] = None,
    min_support: int = 1,
    manifest_digest: str = "",
) -> Suite:
    """Declare a suite with validated cases (idempotent constructor —
    the same declaration always yields the same suite)."""
    return Suite(
        suite_id=suite_id,
        name=name,
        requirement_ids=tuple(requirement_ids),
        cases=tuple(cases),
        runner=runner,
        prepare=prepare,
        aggregate=aggregate,
        min_support=min_support,
        manifest_digest=manifest_digest,
    )


def conformance_case(
    case_id: str,
    requirement_ids: Sequence[str],
    *,
    check: str,
    expected: Any = True,
    tags: Sequence[str] = ("positive",),
    required: bool = True,
) -> Case:
    """Conformance case: ``input.check`` names the runner's check
    function; ``expected`` is the value it must produce."""
    return Case(
        case_id=case_id,
        suite=CONFORMANCE,
        requirement_ids=tuple(requirement_ids),
        input={"check": check},
        expected={"value": expected},
        tags=tuple(tags),
        required=required,
    )


def retrieval_case(
    case_id: str,
    query: str,
    *,
    expect: Sequence[str] = (),
    kind: str = "point",
    mode: str = "current",
    valid_at_us: Optional[int] = None,
    requirement_ids: Sequence[str] = (),
    tags: Sequence[str] = (),
    required: bool = True,
    min_support: int = 1,
) -> Case:
    """Retrieval-quality case: a gold-linked query. ``expect`` holds gold
    ids (never fuzzy text matches); empty ``expect`` with kind
    ``no_answer`` declares an abstention probe (§53.05 keeps failures in
    the denominator either way)."""
    return Case(
        case_id=case_id,
        suite=RETRIEVAL_QUALITY,
        requirement_ids=tuple(requirement_ids),
        input={
            "query": query,
            "mode": mode,
            "kind": kind,
            "valid_at_us": valid_at_us,
        },
        expected={"expect": list(expect)},
        tags=tuple(tags) or (kind,),
        required=required,
        min_support=min_support,
    )


def paired_case(
    case_id: str,
    task: dict,
    *,
    arms: Sequence[str] = PAIRED_REQUIRED_ARMS,
    requirement_ids: Sequence[str] = (),
    tags: Sequence[str] = (),
    required: bool = True,
    min_support: int = 1,
) -> Case:
    """Paired-usefulness case: one task executed once per declared arm.

    ``task`` is runner-defined (e.g. a coding task descriptor with its
    checker); ``arms`` names the control arms to execute — the §53.14
    contract requires common starting snapshots and matched budgets,
    which the runner enforces, not this schema."""
    return Case(
        case_id=case_id,
        suite=PAIRED_USEFULNESS,
        requirement_ids=tuple(requirement_ids),
        input={"task": dict(task), "arms": list(arms)},
        expected={"success_metric": "task_success"},
        tags=tuple(tags) or ("paired",),
        required=required,
        min_support=min_support,
    )


def security_case(
    case_id: str,
    requirement_ids: Sequence[str],
    *,
    probe: str,
    payload: Optional[dict] = None,
    expected: Optional[dict] = None,
    stage: Optional[str] = None,
    benign_twin: bool = False,
    tags: Sequence[str] = (),
    required: bool = True,
) -> Case:
    """Security/privacy case. ``probe`` names the runner's probe (e.g.
    ``poison_write``, ``quarantine_release``, ``deletion_probe``,
    ``unauthorized_read``); ``stage`` marks the §34 lifecycle stage so
    G9 can report per-stage denominators (§54.12); ``benign_twin`` marks
    the retained-coverage twin of an attack fixture (§53.08)."""
    return Case(
        case_id=case_id,
        suite=SECURITY_PRIVACY,
        requirement_ids=tuple(requirement_ids),
        input={"probe": probe, **(payload or {})},
        expected=dict(expected or {}),
        tags=tuple(tags)
        + ((f"stage:{stage}",) if stage else ())
        + (("benign_twin",) if benign_twin else ()),
        required=required,
    )


def declare_suites() -> dict:
    """The four-suite declaration (§53): ids, intent, control arms.

    Concrete case populations come from frozen manifests/corpora loaded
    by callers — this function fixes the *contract* each suite must
    satisfy, which is itself the P0 deliverable (§58 P0)."""
    return {
        sid: {
            "suite_id": sid,
            "description": SUITE_DESCRIPTIONS[sid],
            "control_arms": list(CONTROL_ARMS)
            if sid == PAIRED_USEFULNESS
            else [],
        }
        for sid in SUITE_IDS
    }
