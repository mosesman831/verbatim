"""Track R — the retrieval-only evaluation runner (SPEC_V7 §22.3,
V7-22.12/22.22, metrics per §33).

Judge-free, reader-free: every number is "did the gold evidence unit
reach the delivered prefix", never a model verdict.  The runner drives
each arm over the corpus's real task stream, scores evidence
``any@k``/``all@k``/``prop@k`` (V7-33.12 proportional recall per
V75-04.05)/``ndcg@k``/``mrr@k``/``zero_rate``/refusal metrics plus
per-category mean |G|,
collects per-question attribution (V7-22.21), and emits a manifest-
stubbed report — a run without a manifest digest is labeled
``unpinned``, never silently canonical (V7-22.10/22.24).

Usage:

    python -m eval.v7.track_r --dataset <registry-id|corpus.json>
        --arms verbatim,flat_bm25,fts5 [--k 10,20] [--split dev]
        [--lanes-disabled dense,graph] [--out report.json]

Corpus is any object satisfying the eval.v7 corpus protocol —
``eval/v7/corpora.py``'s ``Corpus`` is resolved lazily through the
dataset registry at CLI time; :class:`DictCorpus` covers plain-dict
corpora and tests.  Arms get gold-free :class:`ArmTask` views only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import metrics as M
from .arms import (
    DictCorpus,
    QueryOutcome,
    arm_name,
    arm_task,
    corpus_digest,
    corpus_items,
    corpus_name,
    corpus_tasks,
    item_document,
    item_ref,
    item_session,
    make_arm,
)
from .attribution import (
    CLASSES,
    STAGES,
    aggregate_table,
    attribute_detail,
)

SCHEMA = "track_r/v7-a"

#: All §32 constants this runner depends on are wave-A provisional.
CONSTANTS_TAG = "provisional/v7-r0"


# ---------------------------------------------------------------------------
# task normalization (scorer side — gold never reaches the arm)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskView:
    """Normalized scorer-side task: gold refs at both granularities."""

    task_id: str
    query: str
    category: str
    answerable: bool
    gold_item: Dict[str, float]          # item/turn granularity
    gold_session: Dict[str, float]       # session granularity
    group_id: Optional[str] = None
    raw: Any = None


def _gold_field(task: Any, *names: str) -> Dict[str, float]:
    for n in names:
        if isinstance(task, Mapping):
            if task.get(n) is not None:
                return M.gold_map(task[n])
        else:
            v = getattr(task, n, None)
            if v is not None:
                return M.gold_map(v)
    return {}


def task_view(task: Any) -> TaskView:
    if isinstance(task, Mapping):
        get = lambda *n, d=None: next(  # noqa: E731
            (task[k] for k in n if task.get(k) is not None), d
        )
    else:
        get = lambda *n, d=None: next(  # noqa: E731
            (getattr(task, k) for k in n if getattr(task, k, None) is not None),
            d,
        )
    answerable = get("answerable")
    if answerable is None:
        abst = get("expected_abstain")
        answerable = (not abst) if abst is not None else True
    return TaskView(
        task_id=str(get("task_id", "id", d="task")),
        query=str(get("query", "question", d="")),
        category=str(get("category", "kind", d="unknown")),
        answerable=bool(answerable),
        gold_item=_gold_field(
            task, "gold_evidence", "evidence_ids", "gold",
            "expected_ids", "expected", "gold_unit_ids",
        ),
        gold_session=_gold_field(
            task, "evidence_session_ids", "gold_session_ids",
            "gold_session_evidence", "session_gold",
        ),
        group_id=get("group_id", "group"),
        raw=task,
    )


def _category_id(task: Any) -> Any:
    """LoCoMo numeric category id when the task carries one
    (``metadata.category_id`` or a top-level ``category_id`` field);
    ``None`` otherwise.  Used to print J14's "category id and name"
    labels without making corpora.py a module dependency."""
    meta: Any = None
    if isinstance(task, Mapping):
        if task.get("category_id") is not None:
            return task["category_id"]
        meta = task.get("metadata")
    else:
        cid = getattr(task, "category_id", None)
        if cid is not None:
            return cid
        meta = getattr(task, "metadata", None)
    if isinstance(meta, Mapping):
        return meta.get("category_id")
    return getattr(meta, "category_id", None)


def _locomo_cat_ids() -> Dict[str, Any]:
    """LoCoMo category-name → numeric id fallback (V7-22.07 table).

    Lazy import — ``eval.v7.corpora`` is deliberately not a module
    dependency of this runner; failure degrades to no id labels.
    """
    try:
        from eval.v7.corpora import LOCOMO_CATEGORY_NAMES

        return {name: cid for cid, name in LOCOMO_CATEGORY_NAMES.items()}
    except Exception:  # noqa: BLE001 — labels degrade, never crash
        return {}


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


def _manifest_stub(manifest: Any) -> Dict[str, Any]:
    """Digest hook — a run without a manifest is labeled ``unpinned``."""
    if manifest is None:
        return {"digest": None, "status": "unpinned"}
    dig = getattr(manifest, "digest", None)
    if callable(dig):
        try:
            return {"digest": str(dig()), "status": "pinned"}
        except Exception:
            pass
    if isinstance(manifest, (str, bytes)):
        return {"digest": str(manifest), "status": "pinned"}
    if isinstance(manifest, Mapping):
        return {
            "digest": str(manifest.get("digest") or manifest.get("id") or ""),
            "status": "pinned" if manifest.get("digest") else "unpinned",
        }
    return {"digest": str(manifest), "status": "pinned"}


def _env_block() -> Dict[str, Any]:
    try:
        from eval.v5.harness import environment

        return environment()
    except Exception:
        return {}


def _delivered_units(
    refs: Sequence[Any],
    granularity: str,
    ref_session: Mapping[str, str],
    gold: Mapping[str, float],
) -> Tuple[frozenset, ...]:
    """Map the arm's delivered item refs to gold-granularity ref-sets."""
    out: List[frozenset] = []
    for r in refs:
        if granularity == "session":
            sess = ref_session.get(str(r))
            # identity fallback covers corpora whose refs *are* sessions
            out.append(frozenset((sess,)) if sess else frozenset((str(r),)))
        else:
            out.append(frozenset((str(r),)))
    return tuple(out)


# ---------------------------------------------------------------------------
# tokens-to-first-gold (V8-22.05) + the V8-15.02 answerability split
# ---------------------------------------------------------------------------

#: Answerability values that count as a premise refusal (V8-22.04's
#: answerability-based rate — null until S5 ships the field).
ANSWERABILITY_REFUSAL = frozenset(
    {"contradicted_premise", "unverified_premise"}
)


def _unit_token_estimates(
    refs: Sequence[Any],
    delivered_text: str,
    ref_toks: Mapping[str, int],
    outcome: Any,
) -> Tuple[List[float], str]:
    """Per-delivered-unit tok/v1 estimates parallel to ``refs``.

    Basis precedence (declared in the report): (1) the arm's own
    per-unit counts/texts (``unit_tokens`` / ``unit_texts`` /
    ``delivered_texts`` on the outcome or its diag) — the delivered
    bytes themselves; (2) the corpus item document per ref — exact for
    arms delivering whole items; (3) refs with no corpus item split the
    unaccounted remainder of ``delivered_text`` evenly.  The basis
    string always names what was measured.
    """
    diag = getattr(outcome, "diag", None) or {}
    pre = diag.get("unit_tokens") or getattr(outcome, "unit_tokens", None)
    if pre:
        vals = [float(t) for t in list(pre)[: len(refs)]]
        if len(vals) == len(refs):
            return vals, "arm_unit_tokens"
    texts = (
        diag.get("unit_texts")
        or diag.get("delivered_texts")
        or getattr(outcome, "unit_texts", None)
    )
    if texts:
        vals = [
            float(M.estimate_tokens(t))
            for t in list(texts)[: len(refs)]
        ]
        if len(vals) == len(refs):
            return vals, "delivered_unit_text"
    vals: List[float] = []
    unmapped: List[int] = []
    for i, r in enumerate(refs):
        t = ref_toks.get(str(r))
        if t is None:
            unmapped.append(i)
            vals.append(0.0)
        else:
            vals.append(float(t))
    basis = "corpus_item_document"
    if unmapped:
        rem = max(
            0.0, float(M.estimate_tokens(delivered_text)) - sum(vals)
        )
        share = rem / len(unmapped)
        for i in unmapped:
            vals[i] = share
        basis = "corpus_item_document+delivered_share"
    return vals, basis


def _tokens_to_first_gold(
    units: Sequence[frozenset],
    gold: Mapping[str, float],
    unit_toks: Sequence[float],
) -> Optional[float]:
    """V8-22.05 — delivered tokens preceding the first unit covering a
    gold ref (the gold unit itself excluded).  ``None`` when no gold is
    delivered or gold is empty — counted separately, never folded into
    the median."""
    gset = set(gold)
    if not gset:
        return None
    acc = 0.0
    for u, t in zip(units, unit_toks):
        if u & gset:
            return acc
        acc += t
    return None


def _tfg_summary(
    tfg_list: Sequence[Mapping[str, Any]], idxs: Sequence[int]
) -> Dict[str, Any]:
    """Median/p90 over gold-delivered questions; undelivered counted."""
    vals = [
        float(tfg_list[i]["value"])
        for i in idxs
        if tfg_list[i]["value"] is not None
    ]
    undel = sum(
        1
        for i in idxs
        if tfg_list[i]["has_gold"] and tfg_list[i]["value"] is None
    )
    return {
        "median": M._percentile(vals, 0.50),
        "p90": M._percentile(vals, 0.90),
        "n_gold_delivered": len(vals),
        "n_gold_undelivered": undel,
    }


def _rec_withheld(rec: Any) -> bool:
    return bool(getattr(rec, "withheld", False)) or (
        str(getattr(rec, "status", "")) in M.ABSTAIN_STATUSES
    )


# ---------------------------------------------------------------------------
# geic@B — gold-evidence-in-context at token budgets (V85-04.04)
# ---------------------------------------------------------------------------
#
# The arm ships the V85-02.04 expansion's delivered ref list per budget
# inside ``QueryOutcome.diag["geic"]`` (``session_messages`` mode only —
# per-item ingest has no session ordinals to expand over).  The scorer
# here intersects those refs with gold — arms never see gold.


def _geic_task_record(tv: Any, outcome: Any) -> Optional[Dict[str, Any]]:
    """One task's geic@B record, or ``None`` when the arm reported no
    geic block (non-session ingest modes stay silent, not zero)."""
    block = (getattr(outcome, "diag", None) or {}).get("geic")
    if not block or not block.get("budgets"):
        return None
    budgets: Dict[str, Any] = {}
    for bkey, b in (block.get("budgets") or {}).items():
        refs = list(b.get("refs") or ())
        drefs = set(refs)
        hit = len(set(tv.gold_item) & drefs)
        budgets[str(bkey)] = {
            "n_delivered": len(refs),
            "tokens": b.get("tokens"),
            "n_gold_hit": hit,
            "geic_any": bool(hit),
            "geic_all": bool(tv.gold_item) and hit == len(tv.gold_item),
            "gold_delivered": sorted(set(tv.gold_item) & drefs),
        }
    return {
        "task_id": tv.task_id,
        "category": tv.category,
        "answerable": tv.answerable,
        "gold": dict(tv.gold_item),
        "budgets": budgets,
        "meta": {
            "window": block.get("window"),
            "meter": block.get("meter"),
            "basis": block.get("basis"),
        },
    }


def _geic_agg(recs: List[Dict[str, Any]], bkey: str) -> Dict[str, Any]:
    """Aggregate geic@B over one slice — T3's denominator is
    answerable questions with non-empty gold that actually carry a
    record for this budget (a missing entry is dropped, never counted
    as a 0)."""
    scored = [
        r for r in recs
        if r["answerable"] and r["gold"] and bkey in r["budgets"]
    ]
    out: Dict[str, Any] = {
        "n": len(scored),
        "n_all_gold": sum(1 for r in recs if r["gold"]),
    }
    if not scored:
        out.update({
            "geic_any": None, "geic_all": None, "geic_prop": None,
            "delivered_mean": None, "tokens_mean": None,
            "tokens_p95": None,
        })
        return out
    anys, alls, props, nds, toks = [], [], [], [], []
    for r in scored:
        b = r["budgets"].get(bkey) or {}
        gold = set(r["gold"])
        hit = int(b.get("n_gold_hit") or 0)
        anys.append(1.0 if hit else 0.0)
        alls.append(1.0 if hit == len(gold) else 0.0)
        props.append(hit / len(gold))
        nds.append(float(b.get("n_delivered") or 0))
        t = b.get("tokens")
        toks.append(float(t) if t is not None else 0.0)
    out.update({
        "geic_any": sum(anys) / len(anys),
        "geic_all": sum(alls) / len(alls),
        "geic_prop": sum(props) / len(props),
        "delivered_mean": sum(nds) / len(nds),
        "tokens_mean": sum(toks) / len(toks),
        "tokens_p95": M._percentile(toks, 0.95),
    })
    return out


def _geic_report(
    recs: List[Dict[str, Any]], meta: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Arm-level ``geic`` block: per-budget aggregates overall and per
    category.  ``None`` when no task carried a geic record — the key
    exists on every arm report but stays null for non-session arms."""
    if not recs:
        return None
    bkeys = sorted(
        {bk for r in recs for bk in r["budgets"]},
        key=lambda k: (k == "unbounded",
                       int(k) if k.isdigit() else 0),
    )
    cats = sorted({r["category"] for r in recs})
    return {
        "window": (meta or {}).get("window"),
        "meter": (meta or {}).get("meter"),
        "basis": (meta or {}).get("basis"),
        "scope": "answerable+gold",
        "budgets": {bk: _geic_agg(recs, bk) for bk in bkeys},
        "by_category": {
            c: {bk: _geic_agg(
                [r for r in recs if r["category"] == c], bk)
                for bk in bkeys}
            for c in cats
        },
    }


def run_track_r(
    corpus: Any,
    arms: Iterable[Any],
    k_list: Sequence[int] = (10, 20),
    *,
    seed: int = 0,
    manifest: Any = None,
    lanes_disabled: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Execute Track R: every task × every arm, scored per §33.

    ``arms`` are protocol instances (``name``/``ingest``/``query``);
    ``lanes_disabled`` is forwarded to arms exposing ``configure()``
    before ingest — arms without it record the request unapplied.
    """
    ks = tuple(sorted({int(k) for k in k_list}))
    if not ks:
        raise ValueError("k_list must be non-empty")
    kmax = ks[-1]
    lanes_req = sorted(str(x) for x in (lanes_disabled or ())) or None

    items = corpus_items(corpus)
    views = [task_view(t) for t in corpus_tasks(corpus)]
    ref_session: Dict[str, str] = {}
    ref_toks: Dict[str, int] = {}
    for i, it in enumerate(items):
        ref = item_ref(it, i)
        sess = item_session(it)
        if sess is not None:
            ref_session[ref] = sess
        try:
            ref_toks[ref] = M.estimate_tokens(item_document(it))
        except Exception:  # noqa: BLE001 — a degenerate item has no text
            pass

    # V75-04.05 / J14: category name → numeric id, read off the tasks
    # that actually ran (observed ids win over the static V7-22.07
    # name table at render time).
    category_ids: Dict[str, Any] = {}
    for tv in views:
        cid = _category_id(tv.raw)
        if cid is not None and tv.category not in category_ids:
            category_ids[tv.category] = cid

    # V8-15.02 split: cat-5 = the category resolving to LoCoMo id 5
    # (observed ids win; the static name table covers dict corpora).
    merged_ids = _locomo_cat_ids()
    merged_ids.update(category_ids)
    cat5_names = {n for n, cid in merged_ids.items() if cid == 5}
    cat5_label = (
        "5 " + "+".join(sorted(cat5_names))
        if cat5_names
        else "5 (unobserved)"
    )

    dataset = {
        "id": str(getattr(corpus, "dataset_id", "") or corpus_name(corpus)),
        "name": corpus_name(corpus),
        "split": getattr(corpus, "split", None),
        "digest": corpus_digest(corpus),
        "n_items": len(items),
        "n_tasks": len(views),
        "n_answerable": sum(1 for v in views if v.answerable),
        "n_session_gold": sum(1 for v in views if v.gold_session),
        "category_ids": category_ids,
    }

    arm_reports: Dict[str, Any] = {}
    per_task: List[Dict[str, Any]] = []
    seen_names: set = set()

    for arm in arms:
        name = arm_name(arm)
        if name in seen_names:
            raise ValueError(f"duplicate arm name {name!r}")
        seen_names.add(name)

        lane_info = None
        if lanes_req:
            cfg_fn = getattr(arm, "configure", None)
            if callable(cfg_fn):
                lane_info = cfg_fn(lanes_disabled=lanes_req)
            else:
                lane_info = {
                    "requested": lanes_req,
                    "applied": [],
                    "unapplied": lanes_req,
                    "reason": "arm exposes no configure()",
                }

        try:
            ingest = arm.ingest(corpus)
        except Exception as exc:  # noqa: BLE001 — recorded, row still named
            arm_reports[name] = {
                "status": "unavailable",
                "reason": f"ingest failed: {type(exc).__name__}: {exc}",
                "lanes_disabled": lane_info,
            }
            continue
        try:
            indexed = set(arm.indexed_refs())
        except Exception:
            indexed = {item_ref(it, i) for i, it in enumerate(items)}
        indexed_sessions = {
            ref_session[r] for r in indexed if r in ref_session
        }

        item_recs: List[M.TaskScore] = []
        sess_recs: List[M.TaskScore] = []
        # V8-15.02/15.03 parallel records aligned with item_recs:
        # attribution detail, answerability, tokens-to-first-gold.
        attr_recs: List[Dict[str, Any]] = []
        answ_list: List[Any] = []
        tfg_list: List[Dict[str, Any]] = []
        tfg_bases: set = set()
        # V85-04.04 — geic@B records (present when the arm ships a
        # ``diag["geic"]`` block — the ``session_messages`` arm does).
        geic_list: List[Dict[str, Any]] = []
        geic_meta: Optional[Dict[str, Any]] = None
        for tv in views:
            try:
                outcome = arm.query(arm_task(tv), kmax)
            except Exception as exc:  # noqa: BLE001 — typed outcome, not a crash
                outcome = QueryOutcome(
                    refs=[], surfaced=None, status="error",
                    error=f"{type(exc).__name__}: {exc}", k=kmax,
                )
            # Attribution at the granularity the task actually carries
            # gold for (item-first when both exist).  V8-15.03: the
            # detail record carries label + withheld + miss stage.
            if tv.gold_item:
                det = attribute_detail(
                    tv, outcome, indexed=indexed, k=kmax
                )
            elif tv.gold_session:
                det = attribute_detail(
                    tv, outcome, indexed=indexed_sessions, k=kmax,
                    granularity="session",
                )
            else:
                det = attribute_detail(
                    tv, outcome, indexed=indexed, k=kmax
                )
            att = det["attribution"]
            # The session record gets its own granularity's class — an
            # item-level miss may still deliver the session unit.
            att_sess = (
                attribute_detail(
                    tv, outcome, indexed=indexed_sessions, k=kmax,
                    granularity="session",
                )["attribution"]
                if tv.gold_session else att
            )
            toks = M.estimate_tokens(outcome.delivered_text)
            delivered_refs = list(outcome.refs)

            # V8-22.05 tokens to first gold (item granularity).
            item_units = _delivered_units(
                delivered_refs, "item", ref_session, tv.gold_item
            )
            unit_toks, tfg_basis = _unit_token_estimates(
                delivered_refs, outcome.delivered_text, ref_toks, outcome
            )
            if delivered_refs:
                tfg_bases.add(tfg_basis)
            tfg = _tokens_to_first_gold(item_units, tv.gold_item, unit_toks)

            # V8-12.03 answerability — null until S5 ships the field.
            answ_val = outcome.diag.get("answerability") or getattr(
                outcome, "answerability", None
            )

            item_recs.append(M.TaskScore(
                task_id=tv.task_id, arm=name, category=tv.category,
                answerable=tv.answerable, gold=dict(tv.gold_item),
                delivered=_delivered_units(
                    delivered_refs, "item", ref_session, tv.gold_item
                ),
                n_items=outcome.n_items, withheld=outcome.abstained,
                status=outcome.status, latency_ms=outcome.latency_ms,
                tokens=toks,
                delivered_bytes=len(outcome.delivered_text.encode("utf-8")),
                attribution=att, granularity="item",
                warnings=tuple(outcome.warnings), error=outcome.error,
            ))
            if tv.gold_session:
                sess_recs.append(M.TaskScore(
                    task_id=tv.task_id, arm=name, category=tv.category,
                    answerable=tv.answerable, gold=dict(tv.gold_session),
                    delivered=_delivered_units(
                        delivered_refs, "session", ref_session,
                        tv.gold_session,
                    ),
                    n_items=outcome.n_items, withheld=outcome.abstained,
                    status=outcome.status, latency_ms=outcome.latency_ms,
                    tokens=toks,
                    delivered_bytes=len(
                        outcome.delivered_text.encode("utf-8")
                    ),
                    attribution=att_sess, granularity="session",
                    warnings=tuple(outcome.warnings), error=outcome.error,
                ))
            attr_recs.append({
                "task_id": tv.task_id,
                "category": tv.category,
                "attribution": att,
                "withheld": det["withheld"],
                "miss_stage": det["miss_stage"],
                "gold_delivered": det["gold_delivered"],
                "provenance": det["provenance"],
            })
            answ_list.append(answ_val)
            tfg_list.append({
                "has_gold": bool(tv.gold_item),
                "value": tfg,
            })
            # V85-04.04 — scorer-side gold only: the arm shipped the
            # expansion's delivered refs in ``diag["geic"]``; the scorer
            # intersects them with gold here.  Gold never reaches the arm.
            geic_rec = _geic_task_record(tv, outcome)
            if geic_rec is not None:
                geic_meta = geic_rec.pop("meta")
                geic_list.append(geic_rec)
            per_task.append({
                "task_id": tv.task_id,
                "arm": name,
                "category": tv.category,
                "answerable": tv.answerable,
                "n_gold": len(tv.gold_item),
                "n_gold_session": len(tv.gold_session),
                "delivered": delivered_refs,
                "n_surfaced": (
                    len(outcome.surfaced)
                    if outcome.surfaced is not None else None
                ),
                "status": outcome.status,
                "withheld": outcome.abstained,
                "verdict": outcome.diag.get("verdict"),
                "answerability": answ_val,
                "suppressed": outcome.diag.get("suppressed"),
                "omitted": outcome.diag.get("omitted"),
                "latency_ms": round(outcome.latency_ms, 3),
                "tokens": toks,
                "tokens_first_gold": (
                    round(tfg, 1) if tfg is not None else None
                ),
                "attribution": att,
                "miss_stage": det["miss_stage"],
                "provenance": det["provenance"],
                "error": outcome.error,
                "geic": (
                    {bk: dict(b) for bk, b in geic_rec["budgets"].items()}
                    if geic_rec is not None else None
                ),
            })

        # V8-15.02 split — answerable (cats 1–4), cat-5 premise, all.
        def _split_group(
            group: str, cat: str, idxs: List[int]
        ) -> Dict[str, Any]:
            recs = [item_recs[i] for i in idxs]
            g = M.aggregate(recs, ks)
            g["group"] = group
            g["cat_label"] = cat
            g["tokens_first_gold"] = _tfg_summary(tfg_list, idxs)
            # V8-22.03 literal: insufficient on answerable / answerable
            # (aggregate()'s false_abstention denominates on gold too).
            n_ans = sum(1 for r in recs if r.answerable)
            g["false_insufficient"] = (
                sum(
                    1 for r in recs if r.answerable and _rec_withheld(r)
                ) / n_ans
                if n_ans else None
            )
            # V8-22.04: correct refusal ships beside the answerability-
            # based rate once S5 exposes the field (null until then).
            ref_idx = [i for i in idxs if not item_recs[i].answerable]
            known = [answ_list[i] for i in ref_idx if answ_list[i]]
            g["answerability_refusal"] = (
                sum(
                    1 for a in known if a in ANSWERABILITY_REFUSAL
                ) / len(ref_idx)
                if ref_idx and known else None
            )
            return g

        all_idx = list(range(len(item_recs)))
        ans_idx = [i for i, r in enumerate(item_recs) if r.answerable]
        cat5_idx = [
            i for i, r in enumerate(item_recs)
            if r.category in cat5_names
        ]
        split = {
            "answerable": _split_group("answerable", "1–4", ans_idx),
            "cat5_premise": _split_group(
                "cat5_premise", cat5_label, cat5_idx
            ),
            "all": _split_group("all", "all", all_idx),
        }

        arm_reports[name] = {
            "status": "executed",
            "overall": M.aggregate(item_recs, ks),
            "categories": M.per_category(item_recs, ks),
            "split": split,
            "session": (
                {
                    "overall": M.aggregate(sess_recs, ks),
                    "categories": M.per_category(sess_recs, ks),
                }
                if sess_recs
                else None
            ),
            "attribution": aggregate_table(attr_recs),
            "tokens_first_gold_basis": sorted(tfg_bases),
            "geic": _geic_report(geic_list, geic_meta),
            "ingest": ingest,
            "lanes_disabled": lane_info,
            "notes": list(getattr(arm, "notes", []) or []),
        }

    return {
        "schema": SCHEMA,
        "constants_tag": CONSTANTS_TAG,
        "dataset": dataset,
        "seed": int(seed),
        "k_list": list(ks),
        "k_attribution": kmax,
        "manifest": _manifest_stub(manifest),
        "lanes_disabled": lanes_req,
        "arms": arm_reports,
        "per_task": per_task,
        "environment": _env_block(),
    }


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _row(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Comparison table renderer — per-arm overall + per-category +
    session granularity + attribution (V7-22.24 layout subset)."""
    ds = report.get("dataset") or {}
    ks = list(report.get("k_list") or (10, 20))
    ks_full = sorted(set(ks) | {10})
    lines = [
        f"## Track R — {ds.get('id', '?')} "
        f"(split={ds.get('split') or 'full'}, seed={report.get('seed')})",
        "",
        f"schema `{report.get('schema')}` · constants "
        f"`{report.get('constants_tag')}` · manifest "
        f"`{(report.get('manifest') or {}).get('status')}` "
        f"{(report.get('manifest') or {}).get('digest') or ''}".rstrip(),
        "",
        f"items={ds.get('n_items')} tasks={ds.get('n_tasks')} "
        f"answerable={ds.get('n_answerable')} · "
        f"k={ks} · corpus digest `{ds.get('digest')}`",
        "",
    ]
    arms = report.get("arms") or {}

    # J14: category rows label "id name" — observed ids from the run's
    # tasks win; the static LoCoMo name table covers dict corpora.
    cat_ids = _locomo_cat_ids()
    cat_ids.update(ds.get("category_ids") or {})

    def cat_label(cat: Any) -> str:
        cid = cat_ids.get(cat)
        return f"{cid} {cat}" if cid is not None else str(cat)

    def metric_cells(agg: Mapping[str, Any]) -> List[str]:
        cells = [_f(agg.get("gold_mean"))]
        cells += [_f(agg.get(f"any@{k}")) for k in ks_full]
        cells += [_f(agg.get(f"all@{k}")) for k in ks_full]
        cells += [_f(agg.get(f"prop@{k}")) for k in ks_full]
        cells += [
            _f(agg.get("ndcg@10")),
            _f(agg.get("mrr@10")),
            _f(agg.get("zero_rate")),
            _f(agg.get("abstain_rate")),
        ]
        lat = agg.get("latency_ms") or {}
        cells += [_f(lat.get("p50"), 1), _f(lat.get("p95"), 1)]
        return cells

    header = (
        ["arm", "cat", "n", "n_gold", "mean\\|G\\|"]
        + [f"any@{k}" for k in ks_full]
        + [f"all@{k}" for k in ks_full]
        + [f"prop@{k}" for k in ks_full]
        + ["ndcg@10", "mrr@10", "zero", "abstain", "p50ms", "p95ms"]
    )
    lines.append("### Evidence recall — item granularity")
    lines.append("")
    lines.append(_row(header))
    lines.append(_row(["---"] * len(header)))
    for name, rep in arms.items():
        if rep.get("status") != "executed":
            lines.append(_row([name, "—", "—", "—"]
                              + ["—"] * (len(header) - 4)
                              ))
            continue
        ov = rep.get("overall") or {}
        lines.append(_row(
            [f"**{name}**", "all", ov.get("n"), ov.get("n_gold")]
            + metric_cells(ov)
        ))
        for cat, agg in (rep.get("categories") or {}).items():
            lines.append(_row(
                [name, cat_label(cat), agg.get("n"), agg.get("n_gold")]
                + metric_cells(agg)
            ))
    lines.append("")

    # V8-15.02 — the answerability split.  Three row groups per arm:
    # answerable (LoCoMo cats 1–4), cat-5 premise, and all.  Cat-5
    # premise-turn delivery (any@k over premise gold) and correct
    # refusal are different numbers on different rows.
    split_header = (
        ["arm", "group", "cat", "n", "mean\\|G\\|"]
        + [f"any@{k}" for k in ks_full]
        + [f"all@{k}" for k in ks_full]
        + [f"prop@{k}" for k in ks_full]
        + [
            "ndcg@10", "mrr@10", "zero", "false_ins", "corr_ref",
            "answ_ref", "tok→G", "tok→G90", "p50ms", "p95ms",
        ]
    )
    n_recall_cells = 1 + 3 * len(ks_full) + 3  # mean|G| .. zero

    def split_cells(g: Mapping[str, Any]) -> List[Any]:
        cells: List[Any] = [_f(g.get("gold_mean"))]
        cells += [_f(g.get(f"any@{k}")) for k in ks_full]
        cells += [_f(g.get(f"all@{k}")) for k in ks_full]
        cells += [_f(g.get(f"prop@{k}")) for k in ks_full]
        tfg = g.get("tokens_first_gold") or {}
        cells += [
            _f(g.get("ndcg@10")),
            _f(g.get("mrr@10")),
            _f(g.get("zero_rate")),
            _f(g.get("false_insufficient")),
            _f(g.get("correct_refusal")),
            _f(g.get("answerability_refusal")),
            _f(tfg.get("median"), 1),
            _f(tfg.get("p90"), 1),
        ]
        lat = g.get("latency_ms") or {}
        cells += [_f(lat.get("p50"), 1), _f(lat.get("p95"), 1)]
        return cells

    lines.append(
        "### Answerability split — item granularity (V8-15.02)"
    )
    lines.append("")
    lines.append(_row(split_header))
    lines.append(_row(["---"] * len(split_header)))
    for name, rep in arms.items():
        sp = rep.get("split") or {}
        if rep.get("status") != "executed" or not sp:
            lines.append(
                _row([name, "—", "—", "—"]
                     + ["—"] * (len(split_header) - 4))
            )
            continue
        for gkey in ("answerable", "cat5_premise", "all"):
            g = sp.get(gkey) or {}
            lines.append(_row(
                [name, gkey, g.get("cat_label"), g.get("n")]
                + split_cells(g)
            ))
            if gkey == "cat5_premise":
                # Refusal on its own row: premise-turn any@k (above)
                # and refused-rate are reported separately (V8-15.02,
                # V8-22.02/22.04) — delivery cells stay empty.
                refusal_cells = (
                    ["—"] * n_recall_cells
                    + [
                        "—",  # false_ins — not an answerable row
                        _f(g.get("correct_refusal")),
                        _f(g.get("answerability_refusal")),
                    ]
                    + ["—"] * 4  # tok→G, tok→G90, p50ms, p95ms
                )
                lines.append(_row(
                    [name, "cat5_premise refusal", g.get("cat_label"),
                     g.get("n")]
                    + refusal_cells
                ))
    lines.append("")

    if any((rep.get("session") or {}).get("overall") for rep in arms.values()):
        lines.append("### Evidence recall — session granularity")
        lines.append("")
        lines.append(_row(header))
        lines.append(_row(["---"] * len(header)))
        for name, rep in arms.items():
            sess = rep.get("session") or {}
            if not sess:
                continue
            ov = sess.get("overall") or {}
            lines.append(_row(
                [f"**{name}**", "all", ov.get("n"), ov.get("n_gold")]
                + metric_cells(ov)
            ))
            for cat, agg in (sess.get("categories") or {}).items():
                lines.append(_row(
                    [name, cat_label(cat), agg.get("n"),
                     agg.get("n_gold")] + metric_cells(agg)
                ))
        lines.append("")

    lines.append("### Attribution (per-question, item granularity)")
    lines.append("")
    lines.append(_row(["arm"] + list(CLASSES)))
    lines.append(_row(["---"] * (len(CLASSES) + 1)))
    for name, rep in arms.items():
        att = (rep.get("attribution") or {}).get("by_class") or {}
        lines.append(_row([name] + [att.get(c, 0) for c in CLASSES]))
    lines.append("")

    # V8-15.03 — the abstain label's underlying split: for every
    # withheld question the stage where its gold actually died.
    stage_counts = [
        (rep.get("attribution") or {}).get("withheld_by_stage") or {}
        for rep in arms.values()
    ]
    if any(any(sc.values()) for sc in stage_counts):
        lines.append(
            "### Withheld questions — underlying miss stage (V8-15.03)"
        )
        lines.append("")
        lines.append(_row(["arm"] + list(STAGES)))
        lines.append(_row(["---"] * (len(STAGES) + 1)))
        for name, rep in arms.items():
            sc = (rep.get("attribution") or {}).get(
                "withheld_by_stage") or {}
            lines.append(_row([name] + [sc.get(s, 0) for s in STAGES]))
        lines.append("")

    # V8-15.03 — provenance tags on delivered gold (ctx_injected,
    # dense_slot, facet, joint, mention, rescue), when arms expose
    # per-item explain.
    prov_tags = sorted({
        t
        for rep in arms.values()
        for t in ((rep.get("attribution") or {}).get("provenance") or {})
    })
    if prov_tags:
        lines.append("### Delivered-gold provenance (V8-15.03)")
        lines.append("")
        lines.append(_row(["arm"] + prov_tags))
        lines.append(_row(["---"] * (len(prov_tags) + 1)))
        for name, rep in arms.items():
            prov = (rep.get("attribution") or {}).get("provenance") or {}
            lines.append(
                _row([name] + [prov.get(t, 0) for t in prov_tags])
            )
        lines.append("")

    # V85-04.04 — geic@B: gold-evidence-in-context across the V8-15.08
    # budget sweep, for arms that shipped a diag["geic"] block.
    geic_arms = {
        n: r.get("geic") for n, r in arms.items() if r.get("geic")
    }
    if geic_arms:
        lines.append(
            "### Gold evidence in context — geic@B (V85-04.04, "
            "answerable+gold)"
        )
        lines.append("")
        for name, g in geic_arms.items():
            meter = g.get("meter") or "?"
            lines.append(
                f"**{name}** — W_r={g.get('window')}, meter=`{meter}`, "
                f"basis: {g.get('basis')}"
            )
            lines.append("")
            head = ["budget", "n", "geic_any", "geic_all", "geic_prop",
                    "delivered_mean", "tokens_mean", "tokens_p95"]
            lines.append(_row(head))
            lines.append(_row(["---"] * len(head)))
            for bk, agg in (g.get("budgets") or {}).items():
                lines.append(_row([
                    bk, agg.get("n"),
                    _f(agg.get("geic_any")), _f(agg.get("geic_all")),
                    _f(agg.get("geic_prop")),
                    _f(agg.get("delivered_mean"), 1),
                    _f(agg.get("tokens_mean"), 1),
                    _f(agg.get("tokens_p95"), 1),
                ]))
            lines.append("")
            cats = g.get("by_category") or {}
            if cats:
                lines.append("per category:")
                lines.append("")
                chead = ["category", "budget", "n", "geic_any",
                         "geic_all", "geic_prop", "tokens_mean"]
                lines.append(_row(chead))
                lines.append(_row(["---"] * len(chead)))
                for cat, bs in cats.items():
                    for bk, agg in bs.items():
                        lines.append(_row([
                            cat_label(cat), bk, agg.get("n"),
                            _f(agg.get("geic_any")),
                            _f(agg.get("geic_all")),
                            _f(agg.get("geic_prop")),
                            _f(agg.get("tokens_mean"), 1),
                        ]))
                lines.append("")

    for name, rep in arms.items():
        if rep.get("status") != "executed":
            lines.append(f"- **{name}**: {rep.get('status')} — "
                         f"{rep.get('reason', '')}")
        for note in rep.get("notes") or []:
            lines.append(f"- {name}: {note}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_dataset(ref: str, split: Optional[str], seed: int) -> Any:
    """Registry first (lazy — concurrent worker module), then a plain
    ``{"items": [...], "tasks": [...]}`` JSON file as DictCorpus."""
    reg_err: Optional[str] = None
    try:
        from eval.v7 import corpora as _corpora  # lazy: not a module dep

        return _corpora.load_corpus(
            ref, split=split, seed=seed if seed else None
        )
    except Exception as exc:  # noqa: BLE001 — recorded honestly
        reg_err = f"{type(exc).__name__}: {exc}"
    if os.path.exists(ref):
        with open(ref, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, Mapping):
            return DictCorpus(
                data.get("items") or (),
                data.get("tasks") or (),
                name=os.path.basename(ref),
                dataset_id=ref,
                split=split,
            )
        raise SystemExit(f"{ref}: expected a JSON object with items/tasks")
    raise SystemExit(
        f"dataset {ref!r}: registry load failed ({reg_err}) "
        "and no readable JSON file at that path"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m eval.v7.track_r",
        description="Track R — retrieval-only evidence-recall runner",
    )
    ap.add_argument("--dataset", required=True,
                    help="registry dataset id or a corpus JSON path")
    ap.add_argument("--split", default=None, choices=[None, "dev", "test"])
    ap.add_argument("--arms", default="flat_bm25,fts5",
                    help="comma-separated arm names")
    ap.add_argument("--k", default="10,20", help="comma-separated cut list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lanes-disabled", default=None,
                    help="comma-separated lanes for leave-one-lane-out")
    ap.add_argument("--workdir", default=None,
                    help="verbatim arm store dir (default: temp)")
    ap.add_argument("--out", default=None, help="write report JSON here")
    ap.add_argument("--md", default=None,
                    help="write markdown here (default: stdout)")
    args = ap.parse_args(argv)

    corpus = _load_dataset(args.dataset, args.split, args.seed)
    ks = tuple(int(x) for x in args.k.split(",") if x.strip())
    lanes = (
        [x.strip() for x in args.lanes_disabled.split(",") if x.strip()]
        if args.lanes_disabled
        else None
    )
    arms = []
    for name in (a.strip() for a in args.arms.split(",") if a.strip()):
        kw = {"workdir": args.workdir} if (name == "verbatim" and args.workdir) else {}
        arms.append(make_arm(name, **kw))
    report = run_track_r(
        corpus, arms, ks, seed=args.seed, lanes_disabled=lanes
    )
    text = render_markdown(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONSTANTS_TAG",
    "SCHEMA",
    "TaskView",
    "main",
    "render_markdown",
    "run_track_r",
    "task_view",
]
