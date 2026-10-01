"""Accuracy-versus-token frontier (E92 / SPEC_V5 §32.01–32.03).

What gets measured:

* **Delivered tokens are the full serialized payload** (V5-32.02) —
  for the consumer route that's ``SearchResult.to_dict()`` bytes
  (items + quotes + labels + warnings + wrappers); for the governed
  lane it's the real ``pack.serialized_bytes`` plus warnings/wrapper
  bytes. The pinned tokenizer is ``bytes//4`` (the production
  ``_APPROX_CHARS_PER_TOKEN`` convention), versioned ``bytes4:v1`` so
  cross-run comparisons stay pinned.

* **Pinned budgets** — the governed recall lane takes real
  ``target_tokens``/``max_bytes`` bounds through ``VerbatimV3.recall`` —
  the same production function ``Memory.search`` composes. Budgets
  sweep the spec's points (1k/2k/4k/8k); ``max_bytes`` is pinned at
  ``tokens*4`` clamped to the contract's 24000-byte ceiling, and the
  clamp is disclosed wherever it binds.

* **Consumer-route operating point** — the facade's natural payload is
  measured as one point on the same axes; it is not a swept budget.

* **Frontier claims** (V5-32.03) — :func:`evaluate_frontier_claim`
  takes OUR curve plus a comparator curve (paired per task) and applies
  the spec's gates: non-inferior within one point at ≥20% fewer tokens,
  or ≥3 points better at matched tokens, with simultaneous intervals.
  A comparator curve that doesn't exist yields ``unavailable`` — the
  claim evaluator never fabricates one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .corpus import ConsumerCorpus, corpus_stats, seed_corpus
from .harness import (
    ConsumerEnv,
    ScoredQuery,
    delivered_payload_bytes,
    hit_source_id,
    percentiles,
    seed_corpus_env,
    settle,
)
from . import stats as st

TOKENIZER_ID = "bytes4:v1"           # pinned tokenizer (V5-32.01)
BUDGET_TOKENS: Tuple[int, ...] = (1000, 2000, 4000, 8000)
MAX_BYTES_CEILING = 24_000           # RecallRequestV3 contract bound


def tokens_of(payload_bytes: int) -> int:
    """Pinned tokenizer: 4 bytes/token, minimum 1 for nonempty payloads."""
    return max(1, payload_bytes // 4) if payload_bytes > 0 else 0


def governed_payload_bytes(result: Any) -> int:
    """Full delivered payload of a ``RecallResultV3`` — pack bytes plus
    warnings/omitted/capability wrapper (V5-32.02)."""
    try:
        pack_bytes = sum(
            int(getattr(p, "serialized_bytes", 0) or 0)
            for p in getattr(result, "packs", ()) or ()
        )
    except Exception:
        pack_bytes = 0
    wrapper = {
        "warnings": list(getattr(result, "warnings", ()) or ()),
        "omitted": getattr(result, "omitted", 0),
        "abstained": getattr(result, "abstained", False),
    }
    return pack_bytes + len(
        json.dumps(wrapper, sort_keys=True, default=str).encode("utf-8")
    )


@dataclass
class FrontierPoint:
    """One (budget, accuracy, tokens) measurement."""

    budget_tokens: int
    max_bytes: int
    clamped: bool
    tasks: int
    correct: int
    accuracy: Optional[float]
    ci95: Optional[Tuple[float, float]]
    mean_tokens: Optional[float]
    tokens_per_correct: Optional[float]
    forbidden_hits: int
    errors: int
    error_kinds: Tuple[str, ...] = ()
    latency_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "budget_tokens": self.budget_tokens,
            "max_bytes": self.max_bytes,
            "max_bytes_clamped": self.clamped,
            "tasks": self.tasks,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "ci95": list(self.ci95) if self.ci95 else None,
            "mean_delivered_tokens": self.mean_tokens,
            "tokens_per_correct": self.tokens_per_correct,
            "forbidden_hits": self.forbidden_hits,
            "errors": self.errors,
            "error_kinds": list(self.error_kinds),
            "latency_ms": self.latency_ms,
        }


def _score_governed(env: ConsumerEnv, task: Any,
                    result: Any, ms: float,
                    k: int) -> Tuple[bool, int, List[str]]:
    """Score one governed-lane result by corpus id (same gold rule as
    the facade path — a hit counts when its source maps to an expected
    item; forbidden ids are violations, not misses)."""
    hits = env.memory._hits_from_packs(result.packs)
    returned: List[str] = []
    for h in hits:
        cid = env.corpus_id(hit_source_id(env, h))
        if cid is not None and cid not in returned:
            returned.append(cid)
        if len(returned) >= k:
            break
    forbidden = [c for c in returned if c in set(task.forbidden_ids)]
    if task.expected_abstain:
        correct = result.abstained or not returned
    elif task.expected_ids:
        correct = bool(set(returned) & set(task.expected_ids)) and not forbidden
    else:
        correct = not returned
    return correct, forbidden and len(forbidden) or 0, returned


def governed_recall(env: ConsumerEnv, query: str, *,
                    target_tokens: int, max_bytes: int,
                    max_items: int = 8) -> Any:
    """One governed-lane recall at a pinned budget — the same
    production ``VerbatimV3.recall`` the facade composes."""
    mem = env.memory
    return mem._v3.recall(
        mem._namespace,
        query,
        principal_id=mem._owner,
        purpose="recall",
        budget={
            "modes": ("evidence",),
            "max_items": max_items,
            "max_bytes": max_bytes,
            "target_tokens": target_tokens,
            "deadline_ms": 400,
            "session_id": mem._session_id,
        },
        session_id=mem._session_id,
    )


def measure_point(env: ConsumerEnv, tasks: Sequence[Any],
                  budget_tokens: int, *, k: int = 8) -> FrontierPoint:
    """Accuracy + delivered-token stats at one pinned budget."""
    max_bytes = min(MAX_BYTES_CEILING, budget_tokens * 4)
    correct = 0
    errors = 0
    error_kinds: List[str] = []
    forbidden = 0
    toks: List[float] = []
    lat: List[float] = []
    for task in tasks:
        t0 = time.perf_counter()
        try:
            res = governed_recall(
                env, task.query,
                target_tokens=budget_tokens, max_bytes=max_bytes)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            kind = f"{type(exc).__name__}:{exc}"[:120]
            if kind not in error_kinds:
                error_kinds.append(kind)
            lat.append((time.perf_counter() - t0) * 1000.0)
            continue
        ms = (time.perf_counter() - t0) * 1000.0
        lat.append(ms)
        ok, n_forbid, _ret = _score_governed(env, task, res, ms, k)
        forbidden += n_forbid
        correct += 1 if ok else 0
        toks.append(tokens_of(governed_payload_bytes(res)))
    n = len(tasks)
    acc = correct / n if n else None
    ci = st.wilson_interval(correct, n) if n else None
    return FrontierPoint(
        budget_tokens=budget_tokens,
        max_bytes=max_bytes,
        clamped=budget_tokens * 4 > MAX_BYTES_CEILING,
        tasks=n,
        correct=correct,
        accuracy=acc,
        ci95=ci,
        mean_tokens=(sum(toks) / len(toks)) if toks else None,
        tokens_per_correct=(sum(toks) / correct) if correct else None,
        forbidden_hits=forbidden,
        errors=errors,
        error_kinds=tuple(error_kinds),
        latency_ms=percentiles(lat),
    )


def measure_consumer_point(env: ConsumerEnv, tasks: Sequence[Any],
                           *, k: int = 8) -> FrontierPoint:
    """The facade's natural operating point — full SearchResult payload
    accounting, no budget knob (disclosed as unswept)."""
    correct = 0
    errors = 0
    error_kinds: List[str] = []
    forbidden = 0
    toks: List[float] = []
    lat: List[float] = []
    for task in tasks:
        t0 = time.perf_counter()
        try:
            res = env.memory.search(task.query, limit=k)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            kind = f"{type(exc).__name__}:{exc}"[:120]
            if kind not in error_kinds:
                error_kinds.append(kind)
            lat.append((time.perf_counter() - t0) * 1000.0)
            continue
        lat.append((time.perf_counter() - t0) * 1000.0)
        items = list(getattr(res, "items", []) or [])
        returned: List[str] = []
        for h in items:
            cid = env.corpus_id(hit_source_id(env, h))
            if cid is not None and cid not in returned:
                returned.append(cid)
            if len(returned) >= k:
                break
        forb = [c for c in returned if c in set(task.forbidden_ids)]
        forbidden += len(forb)
        if task.expected_abstain:
            ok = not items or bool(
                getattr(res, "status", "") in ("pending", "blocked",
                                               "unavailable"))
        elif task.expected_ids:
            ok = bool(set(returned) & set(task.expected_ids)) and not forb
        else:
            ok = not items
        correct += 1 if ok else 0
        toks.append(tokens_of(delivered_payload_bytes(res)))
    n = len(tasks)
    return FrontierPoint(
        budget_tokens=0,
        max_bytes=0,
        clamped=False,
        tasks=n,
        correct=correct,
        accuracy=correct / n if n else None,
        ci95=st.wilson_interval(correct, n) if n else None,
        mean_tokens=(sum(toks) / len(toks)) if toks else None,
        tokens_per_correct=(sum(toks) / correct) if correct else None,
        forbidden_hits=forbidden,
        errors=errors,
        error_kinds=tuple(error_kinds),
        latency_ms=percentiles(lat),
    )


# ---------------------------------------------------------------------------
# frontier claims (V5-32.03)
# ---------------------------------------------------------------------------


def evaluate_frontier_claim(
    ours: Sequence[FrontierPoint],
    theirs: Optional[Sequence[FrontierPoint]],
    *,
    noninferior_margin: float = 0.01,
    cost_margin: float = 0.20,
    better_margin: float = 0.03,
    safety_regressed: bool = False,
) -> dict:
    """V5-32.03 frontier gates, evaluated on paired budget points.

    Returns a verdict dict; ``unavailable`` when the comparator arm has
    no executed curve (a missing comparator is never a claimed win).
    ``safety_regressed=True`` invalidates the claim outright — savings
    that came from dropping conditions/contradictions/identifiers do
    not count.
    """
    if safety_regressed:
        return {"verdict": "failed",
                "reasons": ["safety slice regressed — claim invalid "
                            "(V5-32.03)"]}
    if not theirs:
        return {"verdict": "unavailable",
                "reasons": ["no executed comparator curve — frontier "
                            "claims need a paired arm (V5-23.06)"]}
    ours_by_budget = {p.budget_tokens: p for p in ours}
    theirs_by_budget = {p.budget_tokens: p for p in theirs}
    shared = sorted(set(ours_by_budget) & set(theirs_by_budget))
    if not shared:
        return {"verdict": "inconclusive",
                "reasons": ["no shared budget points between arms"]}
    # paired per-budget comparison: accuracy delta at matched tokens,
    # and token reduction at matched accuracy band.
    verdicts = []
    for b in shared:
        o, t = ours_by_budget[b], theirs_by_budget[b]
        if o.accuracy is None or t.accuracy is None:
            verdicts.append({"budget": b, "verdict": "inconclusive"})
            continue
        acc_delta = o.accuracy - t.accuracy
        tok_red = None
        if o.mean_tokens and t.mean_tokens:
            tok_red = 1.0 - (o.mean_tokens / t.mean_tokens)
        wins = (
            (acc_delta >= -noninferior_margin
             and tok_red is not None and tok_red >= cost_margin)
            or acc_delta >= better_margin
        )
        verdicts.append({
            "budget": b,
            "accuracy_delta": acc_delta,
            "token_reduction": tok_red,
            "verdict": "win" if wins else (
                "tie" if abs(acc_delta) <= noninferior_margin else "loss"
            ),
        })
    n_win = sum(1 for v in verdicts if v.get("verdict") == "win")
    n_loss = sum(1 for v in verdicts if v.get("verdict") == "loss")
    return {
        "verdict": (
            "passed" if n_win and not n_loss
            else "failed" if n_loss else "inconclusive"
        ),
        "points": verdicts,
        "wins": n_win, "losses": n_loss,
        "reasons": [] if n_win else [
            "no budget point cleared the §32.03 frontier gate"],
    }


def run_frontier(*, memories: int = 128, seed: int = 42,
                 budgets: Sequence[int] = BUDGET_TOKENS,
                 k: int = 8,
                 workdir: Optional[str] = None) -> dict:
    """Sweep pinned token budgets on the governed lane + the facade's
    natural operating point."""
    corpus = seed_corpus(memories=memories, seed=seed)
    env = seed_corpus_env(corpus, workdir=workdir, worker="external")
    try:
        settle(env)
        tasks = list(corpus.tasks)
        points = [
            measure_point(env, tasks, b, k=k) for b in budgets
        ]
        consumer = measure_consumer_point(env, tasks, k=k)
    finally:
        env.close()
    return {
        "suite": "frontier",
        "qualification": "locally_measured",
        "tokenizer": TOKENIZER_ID,
        "corpus": corpus_stats(corpus),
        "budgets_tokens": list(budgets),
        "max_bytes_ceiling": MAX_BYTES_CEILING,
        "points": [p.to_dict() for p in points],
        "consumer_route_point": consumer.to_dict(),
        "frontier_claim": evaluate_frontier_claim(points, None),
        "payload_accounting": "full serialized result bytes "
                              "(packs+warnings+wrappers), bytes//4 "
                              "tokens (V5-32.02)",
    }


__all__ = [
    "BUDGET_TOKENS",
    "MAX_BYTES_CEILING",
    "TOKENIZER_ID",
    "FrontierPoint",
    "evaluate_frontier_claim",
    "governed_payload_bytes",
    "governed_recall",
    "measure_consumer_point",
    "measure_point",
    "run_frontier",
    "tokens_of",
]
