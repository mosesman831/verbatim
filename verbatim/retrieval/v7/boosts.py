"""S6 — bounded multiplicative boosts (``boosts/v1``, ``provisional/v7-r0``).

Implements SPEC_V7 §32.5 / V7-10.10–11 (Hindsight's bounded design,
adopted):

```text
recency   = clamp(1 − days(query_time − t_unit) / 365, 0.1, 1.0)
            0.5 if undated; neutral (0.5) when an explicit past window exists
proximity = 1 − min(|t_unit − window_centre| / (window_width / 2), 1)
            0.5 if no window or the unit is undated
proof     = clamp(0.5 + ln(proof_count) / 10, 0, 1) for derived items
            0.5 for raw units
final     = base × (1 + 0.2·(recency − 0.5))
                 × (1 + 0.2·(proximity − 0.5))
                 × (1 + 0.1·(proof − 0.5))
base      = CE-blended score when CE ran (detail["ce_score"/"ce_blend"]),
            else the S4 score replaced by a rank-scaled base in [0.1, 1.0]
            — never a constant (V7-10.11, fixture H20: a pass-through base
            seeded from rank means boosts can never turn the ranking into a
            recency sort, even when every S4 score ties)
```

``t_unit`` is the midpoint of the occurred interval when any occurred bound
is known, else the recorded time (V7-10.10: "occurred axis when known, else
recorded"). All three signals default to neutral ``0.5`` when absent.

**Swing guard (V7-10.10, fixture H19):** the combined multiplicative factor
is hard-clamped to ``[0.75, 1.30]`` (the declared [−25%, +30%] band) before
application; whether the clamp fired is disclosed per candidate. With the
declared α table the natural range is already inside the band (worst cases
≈ 0.787 and ≈ 1.271), so the clamp is a safety net, not a shaper.

Determinism: ordering is ``(-final, -base, source_id, revision desc,
unit_id)`` — total even when two finals coincide. No clock beyond the
caller-supplied ``now_us``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Iterable, Optional

from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    QueryViewV7,
    ScoredCandidate,
)

#: Boost stage identifier recorded in details/stats.
BOOSTS_MODEL_ID = "boosts/v1"

#: Declared α table (§32.5, α_r ≤ 0.2, α_t ≤ 0.2, α_p ≤ 0.1).
ALPHA_RECENCY = 0.2
ALPHA_PROXIMITY = 0.2
ALPHA_PROOF = 0.1

#: Hard combined-swing band (V7-10.10): [−25%, +30%].
SWING_MIN = 0.75
SWING_MAX = 1.30

#: Recency curve: full credit at age 0, floored at 0.1 (§32.5).
RECENCY_FLOOR = 0.1
RECENCY_SPAN_DAYS = 365.0

#: Rank-scaled pass-through base range (§32.5 "scaled to [0.1, 1.0]").
BASE_TOP = 1.0
BASE_BOTTOM = 0.1

_US_PER_DAY = 86_400_000_000.0
_NEUTRAL = 0.5


class BoostedList(list):
    """``list[ScoredCandidate]`` plus run metadata for coverage/explain."""

    def __init__(self, rows: Iterable = (), stats: Optional[dict] = None) -> None:
        super().__init__(rows)
        self.stats: dict = dict(stats or {})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _detail_signals(cand: Any) -> Mapping:
    """Merged unit signals: ``detail["signals"]`` on a ScoredCandidate, or a
    bare ``signals`` attribute (a FusedCandidate passed straight through)."""
    detail = getattr(cand, "detail", None) or {}
    sig = detail.get("signals") if isinstance(detail, Mapping) else None
    if isinstance(sig, Mapping):
        return sig
    sig = getattr(cand, "signals", None)
    return sig if isinstance(sig, Mapping) else {}


def _detail_get(cand: Any, signals: Mapping, *keys: str) -> Any:
    detail = getattr(cand, "detail", None) or {}
    for key in keys:
        if isinstance(detail, Mapping) and key in detail:
            return detail[key]
        if key in signals:
            return signals[key]
    return None


def _unit_time_us(cand: Any, signals: Mapping) -> Optional[float]:
    """Midpoint of the occurred interval when known, else recorded time."""
    start = _num(_detail_get(cand, signals, "occurred_start_us"))
    end = _num(_detail_get(cand, signals, "occurred_end_us"))
    if start is not None and end is not None:
        return (start + end) / 2.0
    if start is not None:
        return start
    if end is not None:
        return end
    return _num(_detail_get(cand, signals, "recorded_at_us"))


def _window(query: QueryViewV7) -> Any:
    return getattr(getattr(query, "intent", None), "window", None)


def _explicit_past_window(window: Any, now_us: Optional[float]) -> bool:
    """V7-09.07/§32.5: recency is neutral for explicit past windows.

    A window that resolved and ends at/before ``now`` is past; a window that
    is present but unresolved/open-ended still marks explicit temporal
    intent, so it also neutralizes recency (never boost "newest" on a
    temporal question).
    """
    if window is None:
        return False
    end = _num(getattr(window, "end_us", None))
    if end is None or now_us is None:
        return True
    return end <= float(now_us)


def _recency(t_unit: Optional[float], query: QueryViewV7,
             now_us: Optional[float]) -> float:
    window = _window(query)
    if _explicit_past_window(window, now_us):
        return _NEUTRAL
    if t_unit is None or now_us is None:
        return _NEUTRAL
    days = (float(now_us) - t_unit) / _US_PER_DAY
    value = 1.0 - days / RECENCY_SPAN_DAYS
    return min(1.0, max(RECENCY_FLOOR, value))


def _proximity(t_unit: Optional[float], query: QueryViewV7) -> float:
    window = _window(query)
    if window is None:
        return _NEUTRAL
    start = _num(getattr(window, "start_us", None))
    end = _num(getattr(window, "end_us", None))
    if start is None or end is None or t_unit is None:
        return _NEUTRAL
    centre = (start + end) / 2.0
    half = (end - start) / 2.0
    if half <= 0:
        return 1.0 if t_unit == centre else 0.0
    return 1.0 - min(abs(t_unit - centre) / half, 1.0)


def _proof(cand: Any, signals: Mapping) -> float:
    derived = _detail_get(cand, signals, "derived")
    if not derived:
        return _NEUTRAL
    pc = _num(_detail_get(cand, signals, "proof_count"))
    if pc is None or pc < 1:
        return _NEUTRAL
    return min(1.0, max(0.0, _NEUTRAL + math.log(pc) / 10.0))


def _base_score(cand: Any, rank0: int, n: int) -> float:
    """CE-blended score when S5 ran, else rank-scaled into [0.1, 1.0].

    Rank scaling is deliberately *not* value scaling (V7-10.11): a pool of
    identical S4 scores still yields strictly decreasing bases, so boosts
    can never collapse the order into a pure recency sort.
    """
    detail = getattr(cand, "detail", None) or {}
    ce = None
    if isinstance(detail, Mapping):
        ce = _num(detail.get("ce_blend"))
        if ce is None:
            ce = _num(detail.get("ce_score"))
    if ce is not None:
        return max(ce, 1e-9)
    if n <= 1:
        return BASE_TOP
    return BASE_TOP - (BASE_TOP - BASE_BOTTOM) * (rank0 / (n - 1))


def _pre_boost_key(cand: Any) -> tuple:
    """Deterministic rank order entering S6: current score desc, then the
    declared identity order."""
    return (
        -float(getattr(cand, "score", 0.0) or 0.0),
        str(getattr(cand, "source_id", "")),
        -int(getattr(cand, "revision", 0) or 0),
        str(getattr(cand, "unit_id", "")),
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def apply_boosts(
    scored: Iterable[ScoredCandidate],
    query: QueryViewV7,
    now_us: Optional[int],
) -> BoostedList:
    """Apply the bounded multiplicative boosts (§32.5).

    ``scored`` — S4/S5 output (already ranked; re-sorted deterministically
    here so input order never matters). ``query`` supplies the intent window;
    ``now_us`` is the query time (falls back to ``query.query_time_us`` when
    ``None``). Returns a :class:`BoostedList` re-sorted by final score with
    per-candidate ``detail["boosts"]`` disclosing every factor.
    """
    if now_us is None:
        now_us = getattr(query, "query_time_us", None)
    now = _num(now_us)

    items = sorted(list(scored or ()), key=_pre_boost_key)
    n = len(items)
    stats: dict = {
        "model": BOOSTS_MODEL_ID,
        "formula_status": FORMULA_STATUS_PROVISIONAL,
        "candidates": n,
        "alphas": {
            "recency": ALPHA_RECENCY,
            "proximity": ALPHA_PROXIMITY,
            "proof": ALPHA_PROOF,
        },
        "swing_band": [SWING_MIN, SWING_MAX],
        "clamped": 0,
        "ce_bases": 0,
        "recency_neutral": 0,
    }
    if n == 0:
        return BoostedList([], stats=stats)

    out: list = []
    for rank0, cand in enumerate(items):
        signals = _detail_signals(cand)
        t_unit = _unit_time_us(cand, signals)
        base = _base_score(cand, rank0, n)
        recency = _recency(t_unit, query, now)
        proximity = _proximity(t_unit, query)
        proof = _proof(cand, signals)

        factor = (
            (1.0 + ALPHA_RECENCY * (recency - _NEUTRAL))
            * (1.0 + ALPHA_PROXIMITY * (proximity - _NEUTRAL))
            * (1.0 + ALPHA_PROOF * (proof - _NEUTRAL))
        )
        clamped = factor < SWING_MIN or factor > SWING_MAX
        factor = min(SWING_MAX, max(SWING_MIN, factor))
        final = base * factor

        if clamped:
            stats["clamped"] += 1
        if isinstance((getattr(cand, "detail", None) or {}), Mapping) and (
            _num((cand.detail or {}).get("ce_blend")) is not None
            or _num((cand.detail or {}).get("ce_score")) is not None
        ):
            stats["ce_bases"] += 1
        if recency == _NEUTRAL:
            stats["recency_neutral"] += 1

        detail = dict(getattr(cand, "detail", None) or {})
        detail["boosts"] = {
            "model": BOOSTS_MODEL_ID,
            "base": base,
            "base_kind": "ce" if (
                _num(detail.get("ce_blend")) is not None
                or _num(detail.get("ce_score")) is not None
            ) else "rank_scaled",
            "pre_boost_score": float(getattr(cand, "score", 0.0) or 0.0),
            "pre_boost_rank": rank0 + 1,
            "t_unit_us": t_unit,
            "recency": recency,
            "proximity": proximity,
            "proof": proof,
            "factor": factor,
            "clamped": clamped,
        }
        out.append((
            ScoredCandidate(
                unit_id=str(getattr(cand, "unit_id", "")),
                source_id=str(getattr(cand, "source_id", "")),
                revision=int(getattr(cand, "revision", 0) or 0),
                score=final,
                score_family=str(getattr(cand, "score_family", "") or "ranking/v7"),
                detail=detail,
            ),
            base,
        ))

    out.sort(key=lambda pair: (
        -pair[0].score,
        -pair[1],
        pair[0].source_id,
        -pair[0].revision,
        pair[0].unit_id,
    ))
    stats["factor_min"] = min(p[0].detail["boosts"]["factor"] for p in out)
    stats["factor_max"] = max(p[0].detail["boosts"]["factor"] for p in out)
    return BoostedList([p[0] for p in out], stats=stats)


__all__ = [
    "ALPHA_PROOF",
    "ALPHA_PROXIMITY",
    "ALPHA_RECENCY",
    "BASE_BOTTOM",
    "BASE_TOP",
    "BOOSTS_MODEL_ID",
    "BoostedList",
    "RECENCY_FLOOR",
    "RECENCY_SPAN_DAYS",
    "SWING_MAX",
    "SWING_MIN",
    "apply_boosts",
]
