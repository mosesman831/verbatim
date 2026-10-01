"""``ranking/v1`` — the V5 deterministic fusion contract (SPEC_V5 §31.2).

Signals rank admitted candidates; eligibility admits (V5-31.05). This
module is a pure function over its inputs: no store access, no clock —
identical inputs over an identical snapshot produce an identical order
(V5-31.03).

Contract (docs/v5_contracts.md §6, V5-31.03):

- **Normalization** — per-signal min-max over the *candidate set*: each
  signal's raw values are mapped into ``[0, 1]`` against the observed
  ``(min, max)`` of the admitted candidates that carry it. Degenerate
  ranges (``max == min``) normalize to ``1.0`` for positive values and
  ``0.0`` otherwise — a uniformly-present signal neither discriminates
  nor is zeroed out. A candidate missing a signal contributes ``0.0``
  for it (no evidence), never a negative share. Non-finite values are
  dropped to absent and counted.
- **Weights** — :data:`RANKING_V1_WEIGHTS` is the frozen default table;
  :data:`RANKING_V1_CLASS_WEIGHTS` holds the declared per-class tables
  (V5-31.03). The ``identifier`` table makes an exact identifier hit
  strictly dominant — no other signal sum can outrank it — honoring
  V5-30.17 ("a fuzzy entity hit can never outrank an exact identifier
  hit for an identifier-bearing query").
- **Combination** — ``score = Σ weight_s · normalized_s`` accumulated
  with ``math.fsum`` (order-independent exact rounding).
- **Tie-breaks** — :data:`RANKING_V1_TIE_BREAK`: fused score descending,
  then ``source_id`` ascending, then ``revision`` descending.
- **Transparency** — every emitted hit carries ``score_detail`` naming
  each contributing signal with its raw value, normalized value, weight,
  and weighted contribution (V5-31.06); diagnostics never include
  objects outside the admitted list.

``fuse`` NEVER admits, generates, or drops eligibility — it reorders the
admitted list it is given. A candidate that failed eligibility upstream
must not be present in ``candidates`` at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

from ...core.types import ErrorCode, VerbatimError
from ...memory.types import RANKING_VERSION
from .. import candidates as _cand

#: The frozen ranking-contract version this module implements.
RANKING_V1 = RANKING_VERSION  # "ranking/v1"

#: Signals the v5 ranking contract recognizes (§31.2 table). Unknown
#: signals in the input are ignored for scoring and counted — a lane can
#: never smuggle an undeclared signal into the order.
SIGNAL_NAMES: tuple = (
    "lexical",
    "similarity",
    "identifier_hit",
    "entity_overlap",
    "temporal_match",
    "type_affinity",
    "corroboration",
    "lifecycle_current",
)

#: Frozen default weight table for ``ranking/v1``. Identifier hits carry
#: the largest single weight (exact-match dominance, V5-30.17); the
#: lifecycle signal keeps current records ahead under current intent.
_DEFAULT_WEIGHTS: dict = {
    "lexical": 1.0,
    "similarity": 0.8,
    "identifier_hit": 2.0,
    "entity_overlap": 0.5,
    "temporal_match": 0.4,
    "type_affinity": 0.3,
    "corroboration": 0.4,
    "lifecycle_current": 0.6,
}

#: Frozen default table — read-only; a caller wanting a per-class table
#: passes it explicitly via ``weights``/``query_class`` (V5-31.03).
RANKING_V1_WEIGHTS: Mapping[str, float] = MappingProxyType(
    dict(_DEFAULT_WEIGHTS)
)

# Per-class tables (V5-31.03 "per-class weight tables"). Each entry is a
# full table: defaults overlaid with the class overrides.
_CLASS_OVERRIDES: dict = {
    # Identifier-bearing queries: an exact hit strictly dominates every
    # other signal combination (2× the sum of all remaining maxima).
    "identifier": {"identifier_hit": 10.0},
    # Entity-centric queries: entity overlap leads; exact identifiers
    # still dominate raw similarity.
    "entity": {"entity_overlap": 1.2, "identifier_hit": 2.0},
    # Temporal intent: the intent/time match is the strongest nudge.
    "temporal": {"temporal_match": 1.0},
}

RANKING_V1_CLASS_WEIGHTS: Mapping[str, Mapping[str, float]] = (
    MappingProxyType({
        cls: MappingProxyType({**_DEFAULT_WEIGHTS, **over})
        for cls, over in _CLASS_OVERRIDES.items()
    })
)

#: The declared tie-break order (V5-31.03) — score descending, then
#: source_id ascending, then revision descending.
RANKING_V1_TIE_BREAK: tuple = (
    "score_desc",
    "source_id_asc",
    "revision_desc",
)


@dataclass
class FusedHit:
    """One fused candidate: identity, final score, and provenance.

    ``signals`` keeps the raw values the caller supplied (finite ones);
    ``score_detail`` names each contributing signal with
    ``{"value", "normalized", "weight", "contribution"}`` — the compact
    per-signal audit V5-31.06 requires on delivered hits.
    """

    source_id: str
    revision: int
    score: float
    rank: int = 0
    signals: dict = field(default_factory=dict)
    score_detail: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.source_id, self.revision)


class Fused(list):
    """``list[FusedHit]`` plus the contract's run metadata.

    ``version``/``weights``/``tie_break`` echo the applied contract for
    the delivery receipt (V5-10.10); ``stats`` carries honest coverage
    (candidates, per-signal min/max, unknown/nonfinite drops, whether a
    ``limit`` truncated the reordered list or a deadline cut the pass).
    """

    def __init__(self, rows=(), *, version=RANKING_V1, weights=None,
                 query_class=None, stats=None) -> None:
        super().__init__(rows)
        self.version = version
        self.weights = dict(weights or RANKING_V1_WEIGHTS)
        self.query_class = query_class
        self.tie_break = RANKING_V1_TIE_BREAK
        self.stats: dict = dict(stats or {})
        self.truncated: bool = bool(self.stats.get("truncated"))


def _cand_key(c: Any) -> tuple:
    """Normalize a candidate identity to ``(source_id, revision)``.

    Accepts ``(source_id, revision)`` pairs, mappings with those keys,
    objects exposing the attributes, or a bare ``source_id`` string
    (revision 0). Anything else is a contract violation — validation
    error, never a guessed key.
    """
    if isinstance(c, (tuple, list)) and len(c) >= 2:
        try:
            return str(c[0]), int(c[1])
        except (TypeError, ValueError):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"candidate pair has non-integer revision: {c!r}",
            )
    if isinstance(c, str):
        return c, 0
    if isinstance(c, Mapping):
        sid = c.get("source_id")
        if sid is None:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"candidate mapping lacks source_id: {c!r}",
            )
        try:
            return str(sid), int(c.get("revision") or 0)
        except (TypeError, ValueError):
            raise VerbatimError(
                ErrorCode.VALIDATION,
                f"candidate mapping has non-integer revision: {c!r}",
            )
    sid = getattr(c, "source_id", None)
    if sid is None:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"candidate has no source identity: {c!r}",
        )
    try:
        return str(sid), int(getattr(c, "revision", 0) or 0)
    except (TypeError, ValueError):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"candidate has non-integer revision: {c!r}",
        )


def _signals_of(c: Any) -> dict:
    """Candidate-carried signals (``.signals`` attr or ``["signals"]``)."""
    if isinstance(c, Mapping):
        raw = c.get("signals")
    else:
        raw = getattr(c, "signals", None)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _normalize_signals_arg(signals: Any) -> tuple:
    """Normalize ``signals`` to ``(by_key, by_id)`` lookup maps.

    Accepted forms:

    - ``{(source_id, revision): {signal: value}}`` — candidate-major;
    - ``{source_id: {signal: value}}`` — revision-agnostic;
    - ``{signal: {key: value}}`` — signal-major, transposed when every
      top-level key is a known signal name;
    - ``None`` — candidates carry their own ``signals``.
    """
    if signals is None:
        return {}, {}
    if not isinstance(signals, Mapping):
        raise VerbatimError(
            ErrorCode.VALIDATION, "signals must be a mapping"
        )
    items = list(signals.items())
    if items and all(
        isinstance(k, str) and k in SIGNAL_NAMES for k, _v in items
    ) and all(isinstance(v, Mapping) for _k, v in items):
        # signal-major: transpose to candidate-major.
        transposed: dict = {}
        for sig, per_key in items:
            for k, v in per_key.items():
                nk = _cand_key(k)
                transposed.setdefault(nk, {})[sig] = v
        by_key = transposed
    else:
        by_key = {}
        for k, v in items:
            if not isinstance(v, Mapping):
                raise VerbatimError(
                    ErrorCode.VALIDATION,
                    f"signals[{k!r}] must be a signal mapping",
                )
            by_key[_cand_key(k)] = dict(v)
    by_id: dict = {}
    for (sid, rev), sigs in by_key.items():
        if rev == 0:
            by_id.setdefault(sid, sigs)
    return by_key, by_id


def _resolve_deadline(deadline: Any) -> Optional[_cand.Deadline]:
    if deadline is None or isinstance(deadline, _cand.Deadline):
        return deadline
    if isinstance(deadline, (int, float)) and not isinstance(deadline, bool):
        return _cand.Deadline(float(deadline))
    return None


def fuse(
    candidates: Iterable,
    signals: Optional[Mapping] = None,
    *,
    weights: Optional[Mapping[str, float]] = None,
    query_class: Optional[str] = None,
    limit: Optional[int] = None,
    deadline: Any = None,
) -> Fused:
    """Reorder admitted candidates under the ``ranking/v1`` contract.

    ``candidates`` is the *admitted* list — this function reorders it and
    never adds, removes for eligibility, or mints candidates (V5-31.05).
    ``signals`` supplies per-candidate raw signal values (see
    ``_normalize_signals_arg`` for accepted shapes); absent signals on a
    candidate contribute zero.

    ``weights`` overrides the default table outright; ``query_class``
    selects a declared per-class table when ``weights`` is not given.
    ``limit`` truncates the reordered list (reported, never silent).
    ``deadline`` accepts a ``candidates.Deadline`` or millisecond budget —
    a cut pass is flagged ``partial`` in ``stats`` rather than silently
    returning an incomplete ordering.
    """
    w = weights
    if w is None and query_class is not None:
        w = RANKING_V1_CLASS_WEIGHTS.get(str(query_class))
    if w is None:
        w = RANKING_V1_WEIGHTS
    w = dict(w)
    dl = _resolve_deadline(deadline)

    stats: dict = {
        "candidates": 0,
        "duplicates_dropped": 0,
        "nonfinite_dropped": 0,
        "unknown_signals": 0,
        "signals": {},
        "truncated": False,
        "partial": False,
        "deadline_exceeded": False,
    }

    # ---- normalize the admitted list -----------------------------------
    keys: list = []
    raw_signals: list = []
    seen: set = set()
    by_key, by_id = _normalize_signals_arg(signals)
    for c in candidates:
        key = _cand_key(c)
        if key in seen:
            stats["duplicates_dropped"] += 1
            continue
        seen.add(key)
        keys.append(key)
        if signals is None:
            sig = _signals_of(c)
        else:
            sig = dict(by_key.get(key) or by_id.get(key[0]) or {})
        clean: dict = {}
        for name, value in sig.items():
            if name not in SIGNAL_NAMES and name not in w:
                stats["unknown_signals"] += 1
                continue
            try:
                fv = float(value)
            except (TypeError, ValueError):
                stats["nonfinite_dropped"] += 1
                continue
            if not math.isfinite(fv):
                stats["nonfinite_dropped"] += 1
                continue
            clean[name] = fv
        raw_signals.append(clean)
    stats["candidates"] = len(keys)
    if not keys:
        return Fused([], weights=w, query_class=query_class, stats=stats)

    # ---- per-signal min-max over the admitted candidate set ------------
    # The normalized space is every signal observed on a candidate plus
    # every declared signal the weight table credits — declared-but-
    # absent signals simply contribute nothing to every candidate.
    present = sorted(
        {s for sig in raw_signals for s in sig}
        | {s for s in SIGNAL_NAMES if w.get(s)}
        | {s for s, weight in w.items() if weight}
    )
    lo: dict = {}
    hi: dict = {}
    for name in present:
        vals = [sig[name] for sig in raw_signals if name in sig]
        if vals:
            lo[name] = min(vals)
            hi[name] = max(vals)
            stats["signals"][name] = {
                "min": lo[name], "max": hi[name],
                "present": len(vals), "weight": w.get(name, 0.0),
            }

    # ---- weighted sum + declared tie-breaks ----------------------------
    scored: list = []
    for i, key in enumerate(keys):
        if dl is not None and dl.expired():
            stats["partial"] = True
            stats["deadline_exceeded"] = True
            break
        sig = raw_signals[i]
        detail: dict = {}
        parts: list = []
        for name in present:
            weight = float(w.get(name, 0.0))
            if weight == 0.0 or name not in sig:
                continue
            value = sig[name]
            lo_v, hi_v = lo[name], hi[name]
            if hi_v > lo_v:
                norm = (value - lo_v) / (hi_v - lo_v)
            else:
                # Degenerate range: uniform presence gets full credit
                # when positive, none when zero/negative.
                norm = 1.0 if value > 0.0 else 0.0
            contribution = weight * norm
            parts.append(contribution)
            if contribution != 0.0:
                # score_detail names contributing signals (V5-31.06) —
                # a present-but-zero-normalized signal is recorded in
                # ``signals`` yet is not a contributor.
                detail[name] = {
                    "value": value,
                    "normalized": norm,
                    "weight": weight,
                    "contribution": contribution,
                }
        score = math.fsum(parts)
        scored.append(FusedHit(
            source_id=key[0], revision=key[1], score=score,
            signals=dict(sig), score_detail=detail,
        ))

    scored.sort(key=lambda h: (-h.score, h.source_id, -h.revision))
    truncated = False
    if limit is not None and len(scored) > max(int(limit), 0):
        scored = scored[: max(int(limit), 0)]
        truncated = True
    stats["truncated"] = truncated
    stats["returned"] = len(scored)
    for rank, hit in enumerate(scored, start=1):
        hit.rank = rank
    return Fused(
        scored, weights=w, query_class=query_class, stats=stats,
    )


__all__ = [
    "Fused",
    "FusedHit",
    "RANKING_V1",
    "RANKING_V1_CLASS_WEIGHTS",
    "RANKING_V1_TIE_BREAK",
    "RANKING_V1_WEIGHTS",
    "SIGNAL_NAMES",
    "fuse",
]
