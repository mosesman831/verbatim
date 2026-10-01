"""Track R metric functions — SPEC_V7 §33 (V7-33.01..07, V7-33.12 per
SPEC_V7_5 V75-04.05), `track_r/v7-a`.

Pure functions over gold refs and delivered ref lists: no I/O, no
verbatim imports, no clock reads.  Unit-testable against hand-computed
fixtures; identical inputs give identical outputs (V7 wave-A
determinism rule).

Conventions
-----------
* ``gold`` — the question's gold evidence set at the scored granularity
  (LoCoMo ``evidence`` dialog ids; LongMemEval ``has_answer`` turn ids
  or ``answer_session_ids`` session ids).  Accepted shapes:

    - an iterable of refs → binary relevance (grade 1.0 each);
    - a mapping ``{ref: grade}`` → graded relevance for NDCG
      (V7-33.03's binary case is the grade-1.0 special case).

  An empty gold set means *no eligible evidence*: the per-question
  metric returns ``None`` so the question is excluded from recall
  denominators honestly (V7-33.01 "averaged over questions with
  non-empty G") rather than scored a silent 0.

* ``delivered`` — the arm's rank-ordered delivered units at the asked
  cut.  Each element is either a ref string or a set/frozenset of refs
  — a delivered *unit* may cover several gold refs (e.g. a turn whose
  item id and session id are both addressable granularities; scoring
  picks the granularity's ref-set per unit).

* "first k distinct delivered units" (V7-33.01): duplicate *scalar*
  refs collapse (the same unit delivered twice is one unit); set
  elements are already-distinct units sharing coverage and are never
  collapsed against each other.

* NDCG uses linear gain and the log2(rank+1) discount with the ideal
  DCG over ``G`` — hand-computable per §33.03.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Result statuses that mean "the verdict withheld" — a typed
#: non-answer, distinct from a lane simply returning nothing.
ABSTAIN_STATUSES = frozenset({"insufficient", "abstained", "no_answer"})

#: Result statuses that mean the run degraded rather than answered —
#: never counted as a retrieval-quality miss or a refusal.
DEGRADED_STATUSES = frozenset(
    {"pending", "blocked", "unavailable", "error"}
)


# ---------------------------------------------------------------------------
# normalization helpers
# ---------------------------------------------------------------------------


def gold_map(gold: Any) -> Dict[str, float]:
    """Normalize gold evidence to ``{ref: grade}`` (binary default)."""
    if gold is None:
        return {}
    if isinstance(gold, Mapping):
        return {str(r): float(g) for r, g in gold.items()}
    if isinstance(gold, (str, bytes)):
        return {str(gold): 1.0}
    return {str(r): 1.0 for r in gold}


def _unit_set(el: Any) -> frozenset:
    """One delivered unit's gold-granularity coverage set."""
    if el is None:
        return frozenset()
    if isinstance(el, (set, frozenset)):
        return frozenset(str(r) for r in el)
    if isinstance(el, (list, tuple)):
        return frozenset(str(r) for r in el)
    return frozenset((str(el),))


def _first_k_units(delivered: Sequence[Any], k: int) -> List[frozenset]:
    """First ``k`` distinct delivered units → coverage sets.

    Scalar refs dedupe by value (a re-delivered ref is the same unit);
    set/tuple elements are distinct units and pass through.
    """
    out: List[frozenset] = []
    seen_scalars: set = set()
    for el in delivered or ():
        if isinstance(el, (str, bytes)):
            key = str(el)
            if key in seen_scalars:
                continue
            seen_scalars.add(key)
        out.append(_unit_set(el))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# per-question evidence metrics (§33.01–33.04, V7-33.12)
# ---------------------------------------------------------------------------


def evidence_any_at_k(
    gold: Any, delivered: Sequence[Any], k: int
) -> Optional[float]:
    """1.0 iff ≥ 1 gold ref is covered by the first k distinct units.

    ``None`` when ``G`` is empty — excluded from the mean denominator.
    """
    g = gold_map(gold)
    if not g:
        return None
    gset = set(g)
    for unit in _first_k_units(delivered, int(k)):
        if unit & gset:
            return 1.0
    return 0.0


def evidence_all_at_k(
    gold: Any, delivered: Sequence[Any], k: int
) -> Optional[float]:
    """1.0 iff *every* gold ref is covered by the first k units."""
    g = gold_map(gold)
    if not g:
        return None
    gset = set(g)
    covered: set = set()
    for unit in _first_k_units(delivered, int(k)):
        covered |= unit & gset
    return 1.0 if gset <= covered else 0.0


def evidence_proportional_at_k(
    gold: Any, delivered: Sequence[Any], k: int
) -> Optional[float]:
    """V7-33.12 (V75-04.05) — LoCoMo's official proportional recall@k.

    ``|G ∩ R_k| / |G|`` where ``R_k`` is the coverage union of the
    first ``k`` *distinct* delivered units — the identical candidate
    set ``evidence_any_at_k``/``evidence_all_at_k`` score, so the
    three metrics always see the same prefix.  ``None`` when ``G``
    is empty: excluded from the mean denominator honestly (the count
    is still reported via the slice's ``n`` vs ``n_gold``).
    """
    g = gold_map(gold)
    if not g:
        return None
    gset = set(g)
    covered: set = set()
    for unit in _first_k_units(delivered, int(k)):
        covered |= unit & gset
    return len(covered) / len(gset)


def ndcg_at_k(
    gold: Any, delivered: Sequence[Any], k: int
) -> Optional[float]:
    """NDCG@k, graded by gold relevance (binary for set-shaped gold).

    ``DCG = Σ_i rel_i / log2(rank_i + 1)`` over the first k units, where
    ``rel_i`` is the best grade among the unit's *fresh* gold refs: each
    gold ref is credited at most once across the ranking (V8-15.01,
    §21.8), so a unit covering only already-credited refs earns 0 and
    nDCG ≤ 1.0 by construction.  ``IDCG`` ranks the top ``min(k, |G|)``
    gold grades ideally.
    """
    g = gold_map(gold)
    if not g:
        return None
    dcg = 0.0
    credited: set = set()
    for rank, unit in enumerate(
        _first_k_units(delivered, int(k)), start=1
    ):
        fresh = {r for r in unit if r in g and r not in credited}
        if fresh:
            dcg += max(g[r] for r in fresh) / math.log2(rank + 1)
            credited |= fresh
    ideal = sorted(g.values(), reverse=True)[: int(k)]
    idcg = sum(
        rel / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1)
    )
    return dcg / idcg if idcg > 0 else None


def mrr_at_k(
    gold: Any, delivered: Sequence[Any], k: int
) -> Optional[float]:
    """Reciprocal rank of the first gold unit within k (0 if absent)."""
    g = gold_map(gold)
    if not g:
        return None
    gset = set(g)
    for rank, unit in enumerate(
        _first_k_units(delivered, int(k)), start=1
    ):
        if unit & gset:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# per-task record + aggregate rates (§33.05–33.07)
# ---------------------------------------------------------------------------


@dataclass
class TaskScore:
    """One task × arm scored outcome (the aggregation record).

    ``gold`` is normalized ``{ref: grade}`` at this record's
    granularity; ``delivered`` is the rank-ordered unit list (each
    element a ref or ref-set).  ``withheld`` marks a verdict-level
    abstention (``status == "insufficient"`` counts implicitly).
    """

    task_id: str
    arm: str
    category: str = "unknown"
    answerable: bool = True
    gold: Dict[str, float] = field(default_factory=dict)
    delivered: Tuple[Any, ...] = ()
    n_items: int = 0
    withheld: bool = False
    status: str = "ok"
    latency_ms: float = 0.0
    tokens: int = 0
    delivered_bytes: int = 0
    attribution: str = ""
    granularity: str = "item"
    warnings: Tuple[str, ...] = ()
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "arm": self.arm,
            "category": self.category,
            "answerable": self.answerable,
            "n_gold": len(self.gold),
            "delivered": [
                sorted(u) if isinstance(u, (set, frozenset)) else u
                for u in self.delivered
            ],
            "n_items": self.n_items,
            "withheld": self.withheld,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 3),
            "tokens": self.tokens,
            "delivered_bytes": self.delivered_bytes,
            "attribution": self.attribution,
            "granularity": self.granularity,
            "warnings": list(self.warnings),
            "error": self.error,
        }


def _is_withheld(rec: TaskScore) -> bool:
    return bool(rec.withheld) or str(rec.status) in ABSTAIN_STATUSES


def _answerable(records: Iterable[TaskScore]) -> List[TaskScore]:
    """§33.05 denominator: answerable questions with non-empty gold."""
    return [
        r for r in records if r.answerable and r.gold
    ]


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def zero_result_rate(records: Iterable[TaskScore]) -> Optional[float]:
    """§33.05 — answerable (non-empty G) questions with zero items."""
    elig = _answerable(records)
    if not elig:
        return None
    return sum(1 for r in elig if r.n_items == 0) / len(elig)


def false_abstention_rate(records: Iterable[TaskScore]) -> Optional[float]:
    """§33.05 — answerable questions whose verdict withheld."""
    elig = _answerable(records)
    if not elig:
        return None
    return sum(1 for r in elig if _is_withheld(r)) / len(elig)


def correct_refusal_rate(records: Iterable[TaskScore]) -> Optional[float]:
    """§33.05 — unanswerable questions refused (verdict withheld or,
    for arms without verdict machinery, zero delivered items)."""
    elig = [r for r in records if not r.answerable]
    if not elig:
        return None
    return sum(
        1 for r in elig if _is_withheld(r) or r.n_items == 0
    ) / len(elig)


def abstention_rate(records: Iterable[TaskScore]) -> Optional[float]:
    """V7-22.22 — fraction of all executed questions withheld."""
    recs = list(records)
    if not recs:
        return None
    return sum(1 for r in recs if _is_withheld(r)) / len(recs)


# ---------------------------------------------------------------------------
# aggregation + category slicing
# ---------------------------------------------------------------------------


def _percentile(samples: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (eval.v5 convention)."""
    vals = sorted(float(s) for s in samples)
    if not vals:
        return None
    i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[i]


def latency_summary(records: Iterable[TaskScore]) -> Dict[str, Any]:
    """§33.07 — wall-time percentiles over executed questions."""
    vals = [r.latency_ms for r in records]
    return {
        "n": len(vals),
        "p50": _percentile(vals, 0.50),
        "p95": _percentile(vals, 0.95),
        "mean": (sum(vals) / len(vals)) if vals else None,
    }


def aggregate(
    records: Iterable[TaskScore], k_list: Sequence[int] = (10, 20)
) -> Dict[str, Any]:
    """Aggregate one slice of task records into the §33 metric row.

    NDCG and MRR are computed at every k in ``k_list`` plus k=10 always
    (V7-22.22 names NDCG@10 regardless of the run's k list).
    """
    recs = list(records)
    ks = sorted({int(k) for k in k_list} | {10})
    scored = [r for r in recs if r.gold]  # non-empty G denominators
    out: Dict[str, Any] = {
        "n": len(recs),
        "n_gold": len(scored),
        "n_answerable": sum(1 for r in recs if r.answerable),
        "n_unanswerable": sum(1 for r in recs if not r.answerable),
        "n_errors": sum(1 for r in recs if r.error),
        # V7-33.12 (V75-04.05): mean |G| over every question in the
        # slice — empty-gold questions count 0, matching the published
        # per-category LoCoMo gold-size means.
        "gold_mean": _mean(float(len(r.gold)) for r in recs),
    }
    for k in ks:
        out[f"any@{k}"] = _mean(
            evidence_any_at_k(r.gold, r.delivered, k) for r in scored
        )
        out[f"all@{k}"] = _mean(
            evidence_all_at_k(r.gold, r.delivered, k) for r in scored
        )
        out[f"prop@{k}"] = _mean(
            evidence_proportional_at_k(r.gold, r.delivered, k)
            for r in scored
        )
        out[f"ndcg@{k}"] = _mean(
            ndcg_at_k(r.gold, r.delivered, k) for r in scored
        )
        out[f"mrr@{k}"] = _mean(
            mrr_at_k(r.gold, r.delivered, k) for r in scored
        )
    out["zero_rate"] = zero_result_rate(recs)
    out["abstain_rate"] = abstention_rate(recs)
    out["false_abstention"] = false_abstention_rate(recs)
    out["correct_refusal"] = correct_refusal_rate(recs)
    out["items_mean"] = _mean(float(r.n_items) for r in recs)
    toks = [float(r.tokens) for r in recs]
    out["tokens_mean"] = (sum(toks) / len(toks)) if toks else None
    out["tokens_p95"] = _percentile(toks, 0.95)
    out["latency_ms"] = latency_summary(recs)
    return out


def per_category(
    records: Iterable[TaskScore], k_list: Sequence[int] = (10, 20)
) -> Dict[str, Dict[str, Any]]:
    """§33.08 micro-average within each category (sorted, stable)."""
    recs = list(records)
    cats = sorted({r.category for r in recs})
    return {
        c: aggregate([r for r in recs if r.category == c], k_list)
        for c in cats
    }


# ---------------------------------------------------------------------------
# tok/v1 estimate (§32.15 formula — eval-side mirror; the engine-side
# implementation lives with the pack worker and is never imported here)
# ---------------------------------------------------------------------------

_TOK_PIECES = re.compile(r"\w+|[^\w\s]", re.U)


def estimate_tokens(text: str) -> int:
    """``tok/v1``: ``ceil(utf8/4 + 0.25 · boundary-pieces)`` clamped
    ≥ ``word_count × 0.75`` (§32.15)."""
    if not text:
        return 0
    utf8 = len(str(text).encode("utf-8"))
    pieces = len(_TOK_PIECES.findall(str(text)))
    words = len(str(text).split())
    est = math.ceil(utf8 / 4 + 0.25 * pieces)
    return max(est, math.ceil(words * 0.75))


__all__ = [
    "ABSTAIN_STATUSES",
    "DEGRADED_STATUSES",
    "TaskScore",
    "abstention_rate",
    "aggregate",
    "correct_refusal_rate",
    "estimate_tokens",
    "evidence_all_at_k",
    "evidence_any_at_k",
    "evidence_proportional_at_k",
    "false_abstention_rate",
    "gold_map",
    "latency_summary",
    "mrr_at_k",
    "ndcg_at_k",
    "per_category",
    "zero_result_rate",
]
