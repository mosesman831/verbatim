"""Per-query failure attribution — SPEC_V7 V7-22.21, `track_r/v7-a`;
attribution v2 per SPEC_V8 V8-15.03.

The v3 F27 attribution method carried forward: each scored question is
classified by the *first observable stage* where its gold evidence
died, never by a guess.  Classes (frozen contract, docs/v7_contracts;
``delivered_withheld`` added by V8-15.03):

* ``delivered``   — ≥ 1 gold ref in the first k delivered units and the
  verdict did not withhold (not a miss; counted so denominators stay
  auditable);
* ``delivered_withheld`` — ≥ 1 gold ref in the first k delivered units
  under a withheld verdict (``insufficient`` / explicit abstain).  A
  trust/status defect (SPEC_V8 D8-26), never a recall loss — v1 folded
  these into ``delivered``;
* ``lane_miss``   — gold exists in the arm's index but no lane surfaced
  it into the observable pool;
* ``rank_shift``  — gold surfaced at rank > k (below the delivery cut);
* ``packed_out``  — gold surfaced at rank ≤ k but was not delivered
  (a pack/verdict stage dropped a top-k candidate);
* ``abstain``     — the verdict withheld (``insufficient`` / explicit
  abstain) — the observable withhold is the class even when the
  pre-verdict pool is invisible;
* ``unsupported`` — no eligible evidence: the task is unanswerable or
  its gold refs were never ingested (V7-22.21 ``not_indexed`` folds
  here — ``detail`` on the diag may still separate them);
* ``unattributed``— observable data cannot distinguish the surviving
  classes (pool below the probe cut, suppressed/omitted tails,
  degraded status, arm error).  Recorded, never guessed.

Attribution v2 (V8-15.03)
-------------------------
``attribute_detail(task, diag)`` returns the full record:

* ``attribution`` — the class label above, identical to ``attribute()``;
* ``withheld``    — the verdict withheld (flag or abstain status),
  recorded independently of the label so a withheld question keeps
  both facts;
* ``miss_stage``  — the underlying stage (``lane_miss``/``rank_shift``/
  ``packed_out``/``unattributed``/``unsupported``) computed beneath a
  withhold — the precedence ambiguity of the v1 ordering (delivered
  checked before the verdict) is removed: a withheld question's record
  always carries both the ``abstain`` label and the stage where its
  gold actually died;
* ``gold_delivered`` — ≥ 1 gold ref in the delivered prefix;
* ``provenance``  — sorted V8-15.03 tags (``ctx_injected``,
  ``dense_slot``, ``facet``, ``joint``, ``mention``, ``rescue``)
  harvested from per-item explain data on gold-covering delivered
  units; empty when the arm exposes no per-item explain.

``attribute(task, diag)`` is pure: ``task`` is any duck-typed record
with gold + answerable fields, ``diag`` a :class:`QueryOutcome`-shaped
object or a plain dict.  The function never imports engine code.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .metrics import ABSTAIN_STATUSES, DEGRADED_STATUSES, gold_map

CLASSES = (
    "delivered",
    "delivered_withheld",
    "lane_miss",
    "rank_shift",
    "packed_out",
    "abstain",
    "unsupported",
    "unattributed",
)

#: The actual failure classes — ``delivered`` is the non-miss outcome.
MISS_CLASSES = tuple(c for c in CLASSES if c != "delivered")

#: Stages the miss ladder can report (V8-15.03 ``miss_stage`` values).
STAGES = ("lane_miss", "rank_shift", "packed_out", "unsupported",
          "unattributed")

#: V8-15.03 provenance tags ← the per-item explain fields (V8-20.04)
#: that evidence each tag.  A tag lands when any listed field is truthy
#: on a gold-covering delivered unit's explain record.
PROVENANCE_TAGS = (
    "ctx_injected",
    "dense_slot",
    "facet",
    "joint",
    "mention",
    "rescue",
)

_PROVENANCE_FIELDS = {
    "ctx_injected": ("ctx_injected", "ctx_from", "injected"),
    "dense_slot": ("dense_slot",),
    "facet": ("facet",),
    "joint": ("joint",),
    "mention": ("mention", "mention_interval"),
    "rescue": ("rescue",),
}

#: Keys a list-shaped per-item explain record may carry its ref under.
_ITEM_REF_KEYS = (
    "ref", "id", "unit", "unit_id", "item", "item_id",
    "memory_id", "source_id", "object_ref",
)


# ---------------------------------------------------------------------------
# tolerant field access (QueryOutcome, dict, or namespace)
# ---------------------------------------------------------------------------


def _dget(diag: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if isinstance(diag, Mapping) and n in diag:
            return diag[n]
        v = getattr(diag, n, None)
        if v is not None:
            return v
    inner = getattr(diag, "diag", None)
    if isinstance(inner, Mapping):
        for n in names:
            if n in inner:
                return inner[n]
    return default


def _task_gold(task: Any, granularity: str) -> Dict[str, float]:
    """Gold refs at the asked granularity, normalized {ref: grade}."""
    if granularity == "session":
        raw = _dget(
            task,
            "gold_session",
            "gold_session_evidence",
            "evidence_session_ids",
            "gold_session_ids",
            "session_gold",
        )
    else:
        raw = _dget(
            task,
            "gold_item",
            "gold_evidence",
            "evidence_ids",
            "gold",
            "expected_ids",
            "expected",
            "gold_unit_ids",
        )
    return gold_map(raw)


def _task_answerable(task: Any) -> bool:
    a = _dget(task, "answerable")
    if a is not None:
        return bool(a)
    abst = _dget(task, "expected_abstain")
    if abst is not None:
        return not abst
    return True


def _unit_cov(el: Any) -> frozenset:
    if isinstance(el, (set, frozenset, list, tuple)):
        return frozenset(str(r) for r in el)
    return frozenset((str(el),))


def _covered(unit_refs: Any, gold: Dict[str, float]) -> bool:
    return bool(_unit_cov(unit_refs) & set(gold))


def _diag_withheld(diag: Any, status: str) -> bool:
    return bool(_dget(diag, "abstained", default=False)) or (
        status in ABSTAIN_STATUSES
    )


# ---------------------------------------------------------------------------
# provenance tags (V8-15.03)
# ---------------------------------------------------------------------------


def _item_explain_map(diag: Any) -> Dict[str, Mapping]:
    """Per-delivered-item explain records → ``{ref: mapping}``.

    Accepts a mapping ``ref → explain-dict`` or a list of dicts each
    carrying a ref key — the producer-side shape is duck-typed so arms
    can surface the V8-20.04 fields without a schema migration.
    """
    raw = _dget(
        diag,
        "item_explain",
        "explain",
        "items_explain",
        "delivered_items",
        "pack_items",
    )
    out: Dict[str, Mapping] = {}
    if isinstance(raw, Mapping):
        for ref, body in raw.items():
            if isinstance(body, Mapping):
                out[str(ref)] = body
    elif isinstance(raw, (list, tuple)):
        for body in raw:
            if not isinstance(body, Mapping):
                continue
            ref = next(
                (body[k] for k in _ITEM_REF_KEYS if body.get(k) is not None),
                None,
            )
            if ref is not None:
                out[str(ref)] = body
    return out


def _provenance_tags(
    diag: Any, delivered: Iterable[Any], gold: Dict[str, float]
) -> list:
    """Sorted V8-15.03 tags on delivered units covering ≥ 1 gold ref."""
    ex = _item_explain_map(diag)
    if not ex:
        return []
    tags = set()
    for unit in delivered:
        if not _covered(unit, gold):
            continue
        for ref in _unit_cov(unit):
            body = ex.get(ref)
            if body is None:
                continue
            for tag, fields in _PROVENANCE_FIELDS.items():
                if any(body.get(f) for f in fields):
                    tags.add(tag)
    return sorted(tags)


# ---------------------------------------------------------------------------
# the classifier
# ---------------------------------------------------------------------------


def _miss_stage(
    diag: Any,
    gold: Dict[str, float],
    indexed: Optional[Iterable[str]],
    cut: int,
    granularity: str,
    status: str,
) -> Tuple[str, bool]:
    """``(stage, hard)`` — the first failing stage beneath the verdict.

    ``hard`` marks stages that keep their own label even under a
    withhold (arm error → ``unattributed``, never-ingested gold →
    ``unsupported``): those outcomes outrank the abstain label exactly
    as they did in v1.
    """
    if _dget(diag, "error") or status == "error":
        return "unattributed", True

    if indexed is not None:
        idx = {str(r) for r in indexed}
        if granularity == "session":
            # session gold is indexed iff some ingested item carries it
            sess = _dget(diag, "indexed_sessions")
            idx = {str(s) for s in sess} if sess is not None else idx
        if not (set(gold) & idx):
            return "unsupported", True  # not_indexed — never ingested

    surfaced = _dget(diag, "surfaced")
    if surfaced is None:
        # The arm exposes no candidate pool — nothing else is knowable.
        return "unattributed", False

    surf_ranks = [
        i + 1
        for i, u in enumerate(surfaced)
        if _covered(u, gold)
    ]
    if not surf_ranks:
        suppressed = _dget(diag, "suppressed", default=0) or 0
        omitted = _dget(diag, "omitted", default=0) or 0
        if suppressed or omitted or status in DEGRADED_STATUSES:
            # Candidates the verdict suppressed or the limit cut are
            # invisible tails — gold may hide there; never guess.
            return "unattributed", False
        return "lane_miss", False

    best = min(surf_ranks)
    return ("packed_out" if best <= cut else "rank_shift"), False


def attribute_detail(
    task: Any,
    diag: Any,
    *,
    indexed: Optional[Iterable[str]] = None,
    k: Optional[int] = None,
    granularity: str = "item",
) -> Dict[str, Any]:
    """The V8-15.03 attribution record for one task × arm outcome.

    Same observable inputs as v1's :func:`attribute`; the record keeps
    the label AND the withheld flag AND the underlying miss stage so a
    withheld question is never flattened to one fact.
    """
    gold = _task_gold(task, granularity)
    answerable = _task_answerable(task)
    status = str(_dget(diag, "status", default=""))
    withheld = _diag_withheld(diag, status)

    detail: Dict[str, Any] = {
        "attribution": "unattributed",
        "withheld": withheld,
        "miss_stage": None,
        "gold_delivered": False,
        "provenance": [],
    }

    # No eligible evidence — unanswerable probes and gold-empty tasks
    # are not misses; they belong to the refusal metrics.
    if not answerable or not gold:
        detail["attribution"] = "unsupported"
        return detail

    refs = list(_dget(diag, "refs", "delivered", default=()) or ())
    cut = int(k if k is not None else (_dget(diag, "k", default=0) or 0))
    cut = cut or len(refs)
    delivered = refs[:cut]

    if any(_covered(u, gold) for u in delivered):
        detail["gold_delivered"] = True
        detail["provenance"] = _provenance_tags(diag, delivered, gold)
        # V8-15.03 / D8-26: gold shipped under a withheld verdict is a
        # trust defect (``delivered_withheld``), not a recall success.
        detail["attribution"] = (
            "delivered_withheld" if withheld else "delivered"
        )
        return detail

    stage, hard = _miss_stage(diag, gold, indexed, cut, granularity, status)
    detail["miss_stage"] = stage
    # Withheld questions keep the ``abstain`` label with the stage
    # recorded beside it (V8-15.03); hard stages (error, never-indexed)
    # outrank it exactly as v1's ordering did.
    detail["attribution"] = (
        stage if hard else ("abstain" if withheld else stage)
    )
    return detail


def attribute(
    task: Any,
    diag: Any,
    *,
    indexed: Optional[Iterable[str]] = None,
    k: Optional[int] = None,
    granularity: str = "item",
) -> str:
    """Classify one task × arm outcome → an attribution class.

    ``indexed`` is the set of refs the arm actually ingested (its
    ``indexed_refs()``); when ``None`` the not-indexed check is skipped
    rather than assumed.  ``k`` is the delivery cut — defaults to
    ``diag.k`` (the cut the arm answered at).

    Returns ``attribute_detail(...)["attribution"]`` — the v1 string
    contract is unchanged.
    """
    return attribute_detail(
        task, diag, indexed=indexed, k=k, granularity=granularity
    )["attribution"]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def aggregate_table(
    records: Iterable[Any],
    *,
    misses_only: bool = False,
) -> Dict[str, Any]:
    """Count attribution classes over task records (``attribution`` +
    ``category`` fields).  ``misses_only`` drops ``delivered`` rows —
    the failure-analysis view of V7-22.21.

    V8-15.03 adds three views alongside the v1 ``by_class`` /
    ``by_category`` / ``misses`` shape: ``by_stage`` counts the
    ``miss_stage`` field wherever a record carries one (every
    non-delivered outcome), ``withheld_by_stage`` restricts it to
    withheld questions (the abstain label's underlying split), and
    ``provenance`` tallies delivered-gold provenance tags.  ``withheld``
    counts records flagged withheld regardless of label.
    """
    by_class = {c: 0 for c in CLASSES}
    by_cat: Dict[str, Dict[str, int]] = {}
    by_stage: Dict[str, int] = {}
    withheld_by_stage: Dict[str, int] = {}
    provenance: Dict[str, int] = {}
    n_withheld = 0
    total = 0
    for rec in records:
        cls = str(_dget(rec, "attribution", default="unattributed"))
        if cls not in by_class:
            cls = "unattributed"
        if misses_only and cls == "delivered":
            continue
        cat = str(_dget(rec, "category", default="unknown"))
        by_class[cls] += 1
        by_cat.setdefault(cat, {c: 0 for c in CLASSES})
        by_cat[cat][cls] += 1
        stage = _dget(rec, "miss_stage", "attribution_miss_stage")
        withheld = bool(
            _dget(rec, "withheld", "attribution_withheld", default=False)
        )
        if withheld:
            n_withheld += 1
        if stage:
            by_stage[str(stage)] = by_stage.get(str(stage), 0) + 1
            if withheld:
                key = str(stage)
                withheld_by_stage[key] = withheld_by_stage.get(key, 0) + 1
        for tag in (
            _dget(rec, "provenance", "attribution_provenance", default=())
            or ()
        ):
            t = str(tag)
            provenance[t] = provenance.get(t, 0) + 1
        total += 1
    return {
        "total": total,
        "by_class": {c: n for c, n in by_class.items()},
        "by_category": {c: by_cat[c] for c in sorted(by_cat)},
        "by_stage": {s: by_stage.get(s, 0) for s in STAGES},
        "withheld_by_stage": {
            s: withheld_by_stage.get(s, 0) for s in STAGES
        },
        "provenance": dict(sorted(provenance.items())),
        "withheld": n_withheld,
        "misses": sum(by_class[c] for c in MISS_CLASSES),
    }


__all__ = [
    "CLASSES",
    "MISS_CLASSES",
    "PROVENANCE_TAGS",
    "STAGES",
    "aggregate_table",
    "attribute",
    "attribute_detail",
]
