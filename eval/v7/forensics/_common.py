"""Shared machinery for the V8 forensic measurement tools
(SPEC_V8 V8-05.01, V8-07.01, V8-11.01, V8-11.02, V8-14.02).

Everything in this package is *measurement scaffolding*: it runs the
real ``Memory`` write + read path (``VerbatimArm`` / ``Memory.add`` →
durable-queue drain → ``Memory.search``) and records exactly which
configuration produced each number.  Nothing here fabricates a metric —
when an input the measurement needs is unobservable (no explain payload,
an unmapped hit, a lane the policy cannot name) the row says so.

Override channels (all recorded in every report's ``manifest``):

* ``policy_overrides`` — passed verbatim to
  :class:`eval.v7.arms.VerbatimArm`; they merge into the ``Memory``
  config mapping at construction (the arm's documented surface).
* ``policy_doc`` — a ``load_policy(profile, policy_json)``-shaped table
  (``lanes`` / ``lane_weights`` / ``lane_gates`` /
  ``nominate_terms_max`` / ``nominate_df_theta``).  The facade currently
  resolves ``load_policy(profile)`` with no document
  (``memory/facade.py::_search_v7``), so the doc is applied through the
  :class:`PolicyPatch` harness seam: ``facade._v7_load_policy`` is
  wrapped to call the *real* ``load_policy``/``ablation_lanes`` for the
  duration of the measurement.  The §02.3 trap is avoided because the
  lane leaves ``ctx.policy.lanes`` itself — the pipeline's declared
  tuple — not the inert ``config.v3.retrieval.*`` gate.
* ``feature_weights`` — rerank-feature weight overrides injected through
  ``rerank_features.score_candidates(weights=...)``, the function's own
  override parameter (e.g. ``{"speaker_match": 0.0}`` for V8-11.02).

Application is *verified*, not assumed: the arm captures
``coverage.explain.policy.lanes`` / ``coverage.lanes`` on every query,
and the manifest records the lane set the pipeline actually printed.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .. import metrics as M
from ..arms import (
    QueryOutcome,
    VerbatimArm,
    arm_task,
    corpus_digest,
    corpus_items,
    corpus_name,
    corpus_tasks,
    item_ref,
    item_session,
)
from ..attribution import attribute
from ..track_r import task_view

SCHEMA = "forensics/v8-a"
CONSTANTS_TAG = "provisional/v7-r0"

#: Fields exempt from the determinism comparison (V8-20.06 t_* rule plus
#: the eval-side wall clocks) — everything else in a report is expected
#: byte-stable across identical runs.
DETERMINISM_EXEMPT = (
    "latency_ms",
    "ingest_ms",
    "wall_ms",
    "elapsed_s",
    "t_ms",
    "slice_ms",
    "t_*",
)


# ---------------------------------------------------------------------------
# environment / manifest helpers
# ---------------------------------------------------------------------------


def env_block() -> Dict[str, Any]:
    try:
        from eval.v5.harness import environment

        return environment()
    except Exception:
        return {}


def dataset_block(corpus: Any, views: Sequence[Any]) -> Dict[str, Any]:
    items = corpus_items(corpus)
    return {
        "id": str(getattr(corpus, "dataset_id", "") or corpus_name(corpus)),
        "name": corpus_name(corpus),
        "split": getattr(corpus, "split", None),
        "digest": corpus_digest(corpus),
        "n_items": len(items),
        "n_tasks": len(views),
        "n_answerable": sum(1 for v in views if v.answerable),
    }


def task_views(corpus: Any) -> List[Any]:
    return [task_view(t) for t in corpus_tasks(corpus)]


def ref_sessions(corpus: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, it in enumerate(corpus_items(corpus)):
        sess = item_session(it)
        if sess is not None:
            out[item_ref(it, i)] = sess
    return out


def category_id_of(task: Any) -> Any:
    """LoCoMo numeric category id when the task carries one (mirrors
    ``track_r._category_id`` — kept local so forensics never depends on
    a sibling file's private helper)."""
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


def delivered_units(
    refs: Sequence[Any],
    granularity: str,
    ref_session: Mapping[str, str],
) -> Tuple[frozenset, ...]:
    """Delivered refs → gold-granularity coverage sets (track_r's
    ``_delivered_units`` semantics, re-implemented locally)."""
    out: List[frozenset] = []
    for r in refs:
        if granularity == "session":
            sess = ref_session.get(str(r))
            out.append(frozenset((sess,)) if sess else frozenset((str(r),)))
        else:
            out.append(frozenset((str(r),)))
    return tuple(out)


def gold_ranks(refs: Sequence[Any], gold: Mapping[str, float]) -> List[int]:
    """1-based ranks of delivered units covering ≥ 1 gold ref."""
    gset = set(gold)
    return [
        i + 1
        for i, r in enumerate(refs)
        if str(r) in gset
    ]


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj, default=str)
        return obj
    except Exception:
        return repr(obj)


def write_report(report: Mapping[str, Any], out: Optional[str]) -> Optional[str]:
    if not out:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, default=str)
    return out


# ---------------------------------------------------------------------------
# PolicyPatch — the policy/feature override seam (verified, never assumed)
# ---------------------------------------------------------------------------


class PolicyPatch:
    """Apply a policy-table / feature-weight variant to the V7 engine
    for the duration of a measurement block.

    * ``lanes_disabled`` removes names from the resolved policy's lane
      tuple via ``policy.ablation_lanes`` (the §02.3-correct mechanism —
      the lane never reaches ``ctx.policy.lanes``).
    * ``policy_doc`` is a ``load_policy`` ``policy_json`` document
      resolved by the real ``load_policy(profile, doc)`` — it may carry
      ``lanes``/``lane_weights``/``lane_gates``/``nominate_*`` and
      ``profiles`` overlays.  ``lanes_disabled`` applies *after* the doc.
    * ``feature_weights`` merges over ``rerank_features.FEATURE_WEIGHTS_V1``
      through ``score_candidates(weights=...)``.

    Mechanism: ``verbatim.memory.facade._v7_load_policy`` and
    ``verbatim.retrieval.v7.rerank_features.score_candidates`` are module
    attributes resolved at call time; the patch wraps them, and restores
    the originals on exit.  ``verify(outcome_diag)`` inspects what the
    pipeline actually printed so a silently-unapplied override lands in
    the manifest as ``applied: false`` — never as a fabricated arm.
    """

    def __init__(
        self,
        *,
        lanes_disabled: Optional[Iterable[str]] = None,
        policy_doc: Optional[Mapping[str, Any]] = None,
        feature_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.lanes_disabled = tuple(str(x) for x in (lanes_disabled or ()))
        self.policy_doc = (
            copy.deepcopy(dict(policy_doc)) if policy_doc is not None else None
        )
        self.feature_weights = (
            {str(k): float(v) for k, v in feature_weights.items()}
            if feature_weights
            else None
        )
        self._undo: List[Any] = []
        self.errors: List[str] = []

    # -- validation ----------------------------------------------------

    def dry_resolve(self, profile: str = "default") -> Optional[Any]:
        """Resolve the patched policy WITHOUT installing the patch —
        validation + manifest echo.  ``None`` when resolution failed
        (the error is in ``self.errors``)."""
        try:
            from verbatim.retrieval.v7.policy import ablation_lanes, load_policy
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"policy module unavailable: {exc!r}")
            return None
        try:
            pol = load_policy(profile, self.policy_doc) if self.policy_doc \
                else load_policy(profile)
            if self.lanes_disabled:
                pol = ablation_lanes(pol, self.lanes_disabled)
            return pol
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return None

    # -- install ---------------------------------------------------------

    def __enter__(self) -> "PolicyPatch":
        from verbatim.retrieval.v7 import policy as v7_policy

        import verbatim.memory.facade as facade_mod

        orig_loader = getattr(facade_mod, "_v7_load_policy", None)
        doc = self.policy_doc
        disabled = self.lanes_disabled
        errors = self.errors

        if orig_loader is None:
            errors.append("facade._v7_load_policy is None — v7 pipeline "
                          "unavailable to patch")
        elif doc is not None or disabled:
            def _patched(profile: str, _orig=orig_loader) -> Any:
                base = (
                    v7_policy.load_policy(profile, doc)
                    if doc is not None
                    else _orig(profile)
                )
                if disabled:
                    base = v7_policy.ablation_lanes(base, disabled)
                return base

            facade_mod._v7_load_policy = _patched
            self._undo.append(
                lambda: setattr(facade_mod, "_v7_load_policy", orig_loader)
            )

        if self.feature_weights is not None:
            try:
                from verbatim.retrieval.v7 import rerank_features as rf_mod
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"rerank_features unavailable: {exc!r}"
                )
            else:
                orig_score = rf_mod.score_candidates
                overrides = dict(self.feature_weights)

                def _patched_score(query, fused, *a, **kw):
                    if kw.get("weights") is None:
                        merged = dict(rf_mod.FEATURE_WEIGHTS_V1)
                        merged.update(overrides)
                        kw["weights"] = merged
                    return orig_score(query, fused, *a, **kw)

                rf_mod.score_candidates = _patched_score
                self._undo.append(
                    lambda: setattr(rf_mod, "score_candidates", orig_score)
                )
        return self

    def __exit__(self, *exc: Any) -> bool:
        while self._undo:
            self._undo.pop()()
        return False

    # -- manifest --------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        via: List[str] = []
        if self.policy_doc is not None or self.lanes_disabled:
            via.append("harness:verbatim.memory.facade._v7_load_policy"
                       "→load_policy/ablation_lanes")
        if self.feature_weights is not None:
            via.append("harness:rerank_features.score_candidates(weights=…)")
        return {
            "lanes_disabled": list(self.lanes_disabled) or None,
            "policy_doc": _jsonable(self.policy_doc),
            "feature_weights": self.feature_weights,
            "applied_via": via,
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# ForensicVerbatimArm — VerbatimArm + per-call census/explain capture
# ---------------------------------------------------------------------------


class ForensicVerbatimArm(VerbatimArm):
    """``VerbatimArm`` that retains what the pipeline printed.

    ``mem.search`` is instance-wrapped for the duration of each
    ``query()`` so the measured call *and* the arm's own pool probe hand
    back their (census snapshot, result) pairs — no extra searches, no
    re-implemented query path.  ``diag`` gains:

    * ``sql`` — census of the measured call (``SqlCensus`` attached);
    * ``sql_calls`` — per-``search``-call census list (measured first);
    * ``policy_lanes`` — ``coverage.explain.policy.lanes`` as applied;
    * ``lane_statuses`` — ``coverage.lanes`` (``v7.*`` statuses);
    * ``explain`` — the full explain payload when ``full_explain``.
    """

    def __init__(
        self,
        *args: Any,
        name: str = "verbatim",
        full_explain: bool = False,
        census: Any = None,
        **kw: Any,
    ) -> None:
        super().__init__(*args, **kw)
        self.name = name  # instance-shadow the class attr (arm_name reads it)
        self.full_explain = bool(full_explain)
        self.census = census
        self.explains: Dict[str, Any] = {}
        self.coverages: Dict[str, Any] = {}
        self.sql_calls: Dict[str, List[Any]] = {}

    def query(self, task: Any, k: int) -> QueryOutcome:
        mem = self._mem
        captured: List[Tuple[Any, Any]] = []
        if mem is None:
            return super().query(task, k)

        cen = self.census
        orig = mem.search

        def counted(*a: Any, **kw: Any) -> Any:
            if cen is not None:
                cen.reset()
            try:
                res = orig(*a, **kw)
            except Exception:
                if cen is not None:
                    captured.append((cen.snapshot(), None))
                raise
            captured.append(
                (cen.snapshot() if cen is not None else None, res)
            )
            return res

        # Explain seam: the production facade calls ``run_search``
        # without ``explain=True``, so ``coverage.explain`` — the policy
        # echo AND the per-item lane ranks/signals/features every
        # forensic decomposer reads — never reaches the caller.  The
        # arm forces ``explain=True`` on the facade's own
        # ``_v7_run_search`` reference (a module attribute resolved at
        # call time — the same seam PolicyPatch uses for
        # ``_v7_load_policy``) inside the *same measured search*: real
        # pipeline output, no second unmeasured query.  The explain
        # build is scorer-side payload assembly — rankings are
        # unaffected; its cost is inside the measured latency and the
        # manifest records ``explain_forced``.
        undo: List[Any] = []
        explain_forced = False
        try:
            import verbatim.memory.facade as facade_mod

            orig_run = getattr(facade_mod, "_v7_run_search", None)
            if orig_run is not None:
                def _explain_forced(ctx: Any, qv: Any, **kw: Any) -> Any:
                    kw["explain"] = True
                    return orig_run(ctx, qv, **kw)

                facade_mod._v7_run_search = _explain_forced
                explain_forced = True
                undo.append(
                    lambda: setattr(
                        facade_mod, "_v7_run_search", orig_run
                    )
                )
        except Exception:  # noqa: BLE001 — explain stays best-effort
            pass

        mem.search = counted  # instance attr shadows the bound method
        try:
            out = super().query(task, k)
        finally:
            mem.search = orig
            while undo:
                undo.pop()()
        out.diag["explain_forced"] = explain_forced

        tid = str(getattr(task, "task_id", ""))
        if captured:
            self.sql_calls[tid] = [snap for snap, _res in captured]
            out.diag["sql_calls"] = [snap for snap, _res in captured]
            out.diag["sql"] = captured[0][0]
            res = captured[-1][1]
            cov = (getattr(res, "coverage", None) or {}) if res is not None else {}
            self.coverages[tid] = {k2: v for k2, v in cov.items()
                                   if k2 != "explain"}
            explain = cov.get("explain")
            if explain:
                pol = explain.get("policy") or {}
                out.diag["policy_lanes"] = list(pol.get("lanes") or [])
                if self.full_explain:
                    self.explains[tid] = explain
                    out.diag["explain"] = explain
            if cov.get("lanes"):
                out.diag["lane_statuses"] = dict(cov["lanes"])
        elif self._mem is not None:
            # no calls captured means search never ran — leave the
            # parent's outcome untouched (error path is already typed)
            pass
        return out


# ---------------------------------------------------------------------------
# arm run — Track R question stream under one spec
# ---------------------------------------------------------------------------


@dataclass
class ArmSpec:
    """One measured variant in a paired run."""

    label: str
    patch: Optional[PolicyPatch] = None
    census: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class QuestionRow:
    """Scorer-side record for one task × one spec (pre-JSON)."""

    task_id: str
    conv_id: Optional[str]
    category: str
    category_id: Any
    answerable: bool
    gold: Dict[str, float]
    gold_session: Dict[str, float]


def _session_indexed(refs: Iterable[str], ref_session: Mapping[str, str]) -> set:
    return {ref_session[r] for r in refs if r in ref_session}


def _question_record(
    tv: Any,
    out: QueryOutcome,
    *,
    att: str,
    att_sess: Optional[str],
    ref_session: Mapping[str, str],
) -> Dict[str, Any]:
    delivered = list(out.refs)
    surfaced = list(out.surfaced) if out.surfaced is not None else None
    gr = gold_ranks(delivered, tv.gold_item)
    pr = gold_ranks(surfaced or (), tv.gold_item)
    return {
        "delivered": delivered,
        "surfaced": surfaced,
        "n_items": out.n_items,
        "status": out.status,
        "abstained": bool(out.abstained),
        "gold_ranks": gr,
        "first_gold_rank": min(gr) if gr else None,
        "pool_gold_rank": min(pr) if pr else None,
        "attribution": att,
        "attribution_session": att_sess,
        "latency_ms": round(out.latency_ms, 3),
        "warnings": list(out.warnings),
        "error": out.error,
        "verdict": out.diag.get("verdict"),
        "suppressed": out.diag.get("suppressed"),
        "omitted": out.diag.get("omitted"),
        "policy_lanes": out.diag.get("policy_lanes"),
        "lane_statuses": out.diag.get("lane_statuses"),
        "explain_forced": out.diag.get("explain_forced"),
        "sql": out.diag.get("sql"),
        "sql_calls": out.diag.get("sql_calls"),
    }


def _task_score(
    tv: Any,
    out: QueryOutcome,
    att: str,
    arm_label: str,
    granularity: str,
    ref_session: Mapping[str, str],
) -> Any:
    gold = tv.gold_item if granularity == "item" else tv.gold_session
    toks = M.estimate_tokens(out.delivered_text)
    return M.TaskScore(
        task_id=tv.task_id,
        arm=arm_label,
        category=tv.category,
        answerable=tv.answerable,
        gold=dict(gold),
        delivered=delivered_units(out.refs, granularity, ref_session),
        n_items=out.n_items,
        withheld=out.abstained,
        status=out.status,
        latency_ms=out.latency_ms,
        tokens=toks,
        delivered_bytes=len(out.delivered_text.encode("utf-8")),
        attribution=att,
        granularity=granularity,
        warnings=tuple(out.warnings),
        error=out.error,
    )


def run_spec(
    views: Sequence[Any],
    arm: ForensicVerbatimArm,
    spec: ArmSpec,
    *,
    k: int,
    ref_session: Mapping[str, str],
) -> Dict[str, Any]:
    """Run every task under ``spec`` (its PolicyPatch active for the
    query loop).  ``arm`` is already ingested — the identical store is
    queried under each variant, the tightest pairing available."""
    rows: Dict[str, Dict[str, Any]] = {}
    item_recs: List[Any] = []
    sess_recs: List[Any] = []
    indexed = set(arm.indexed_refs())
    indexed_sessions = _session_indexed(indexed, ref_session)

    patch = spec.patch or PolicyPatch()
    # Eager validation: an unresolvable patched policy (typo'd lane
    # name, malformed doc) is a typed spec failure, not N identical
    # per-question errors.  ``dry_resolve`` records its reason in
    # ``patch.errors`` — the manifest keeps it.
    if patch.lanes_disabled or patch.policy_doc:
        if patch.dry_resolve() is None:
            return {
                "rows": {},
                "item_recs": [],
                "sess_recs": [],
                "patch": patch.describe(),
                "error": "policy resolution failed: "
                         + "; ".join(patch.errors),
                "explains": {},
                "coverages": {},
                "sql_calls": {},
            }
    arm.explains.clear()
    arm.coverages.clear()
    arm.sql_calls.clear()
    with patch:
        for tv in views:
            try:
                out = arm.query(arm_task(tv), k)
            except Exception as exc:  # noqa: BLE001 — typed, not a crash
                out = QueryOutcome(
                    refs=[], surfaced=None, status="error",
                    error=f"{type(exc).__name__}: {exc}", k=k,
                )
            if tv.gold_item:
                att = attribute(tv, out, indexed=indexed, k=k)
            else:
                att = attribute(tv, out, indexed=indexed, k=k)
            att_sess = (
                attribute(tv, out, indexed=indexed_sessions, k=k,
                          granularity="session")
                if tv.gold_session else None
            )
            rows[tv.task_id] = _question_record(
                tv, out, att=att, att_sess=att_sess,
                ref_session=ref_session,
            )
            item_recs.append(_task_score(
                tv, out, att, spec.label, "item", ref_session,
            ))
            if tv.gold_session:
                sess_recs.append(_task_score(
                    tv, out, att_sess or att, spec.label, "session",
                    ref_session,
                ))

    return {
        "rows": rows,
        "item_recs": item_recs,
        "sess_recs": sess_recs,
        "patch": patch.describe(),
        "explains": dict(arm.explains),
        "coverages": dict(arm.coverages),
        "sql_calls": dict(arm.sql_calls),
    }


# ---------------------------------------------------------------------------
# paired run — one ingested store, N variant query loops
# ---------------------------------------------------------------------------


def _confusion(rows_a: Mapping[str, Any], rows_b: Mapping[str, Any],
               golds: Mapping[str, Mapping[str, float]], k: int) -> Dict[str, int]:
    """Paired any@k hit/miss counts — the McNemar substrate."""
    both = only_a = only_b = neither = 0
    for tid, gold in golds.items():
        if not gold:
            continue
        ra = rows_a.get(tid) or {}
        rb = rows_b.get(tid) or {}
        ha = M.evidence_any_at_k(gold, ra.get("delivered") or (), k)
        hb = M.evidence_any_at_k(gold, rb.get("delivered") or (), k)
        if ha is None or hb is None:
            continue
        if ha and hb:
            both += 1
        elif ha:
            only_a += 1
        elif hb:
            only_b += 1
        else:
            neither += 1
    return {"both_hit": both, "only_first": only_a,
            "only_second": only_b, "both_miss": neither}


def paired_run(
    corpus: Any,
    specs: Sequence[ArmSpec],
    *,
    k_list: Sequence[int] = (10, 20),
    arm_kwargs: Optional[Mapping[str, Any]] = None,
    policy_overrides: Optional[Mapping[str, Any]] = None,
    full_explain: bool = False,
    need_census: bool = False,
    tool: str = "paired",
    extra_manifest: Optional[Mapping[str, Any]] = None,
    sink: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Paired measurement: one ingested ``Memory`` store, one query loop
    per spec under its :class:`PolicyPatch`.

    Per-question granularity is the contract — ``questions[]`` carries
    both variants' delivered refs, gold ranks, status, attribution and
    latency so ``eval/v8/stats.py``-style paired bootstrap/McNemar can
    consume the rows directly.

    ``sink`` (optional) receives out-of-band heavy data that must not
    bloat the JSON report: ``sink["explains"][label]`` per-spec explain
    payloads, ``sink["source_ref"]`` the arm's source→corpus-ref map,
    and ``sink["coverages"][label]`` per-task coverage blocks.
    """
    from .sql_census import SqlCensus

    ks = tuple(sorted({int(k) for k in k_list}))
    kmax = ks[-1]
    views = task_views(corpus)
    ref_session = ref_sessions(corpus)

    labels = [s.label for s in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate spec labels: {labels}")

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "constants_tag": CONSTANTS_TAG,
        "tool": tool,
        "determinism_exempt": list(DETERMINISM_EXEMPT),
        "dataset": dataset_block(corpus, views),
        "k_list": list(ks),
        "k_attribution": kmax,
        "status": "executed",
        "not_run": [],
        "manifest": {
            "arm_class": "eval.v7.forensics.ForensicVerbatimArm",
            "arm_kwargs": dict(arm_kwargs or {}),
            "policy_overrides": _jsonable(policy_overrides),
            "shared_store": True,
            "conv_id_field": "task.group_id",
            "explain_forced": True,
            "environment": env_block(),
            **(dict(extra_manifest or {})),
        },
        "specs": {},
        "questions": [],
    }

    census = SqlCensus() if (need_census or any(s.census for s in specs)) else None
    arm = ForensicVerbatimArm(
        name="verbatim",
        full_explain=full_explain,
        census=census,
        policy_overrides=dict(policy_overrides or {}),
        **dict(arm_kwargs or {}),
    )
    t0 = time.time()
    try:
        try:
            ingest = arm.ingest(corpus)
            report["manifest"]["ingest"] = ingest
        except Exception as exc:  # noqa: BLE001 — recorded, nothing runs
            report["status"] = "not_run"
            report["not_run"].append(
                f"ingest failed: {type(exc).__name__}: {exc}"
            )
            return report
        if census is not None:
            try:
                census.attach_store(arm._mem._store)
                report["manifest"]["sql_scopes"] = census.scopes
            except Exception as exc:  # noqa: BLE001
                report["manifest"]["sql_scopes"] = []
                report["not_run"].append(
                    f"census attach failed: {type(exc).__name__}: {exc}"
                )

        if sink is not None:
            sink["source_ref"] = dict(getattr(arm, "_source_ref", {}) or {})
            sink["explains"] = {}
            sink["coverages"] = {}
        outcomes: Dict[str, Dict[str, Any]] = {}
        for spec in specs:
            try:
                outcomes[spec.label] = run_spec(
                    views, arm, spec, k=kmax, ref_session=ref_session,
                )
            except Exception as exc:  # noqa: BLE001
                outcomes[spec.label] = {
                    "rows": {}, "item_recs": [], "sess_recs": [],
                    "patch": (spec.patch or PolicyPatch()).describe(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if outcomes[spec.label].get("error"):
                report["not_run"].append(
                    f"spec {spec.label!r}: "
                    f"{outcomes[spec.label]['error']}"
                )
            if sink is not None:
                sink["explains"][spec.label] = outcomes[spec.label].get(
                    "explains") or {}
                sink["coverages"][spec.label] = outcomes[spec.label].get(
                    "coverages") or {}
        report["wall_ms"] = round((time.time() - t0) * 1000.0, 1)

        # ---- per-spec manifests + aggregates ------------------------------
        golds = {tv.task_id: tv.gold_item for tv in views}
        for spec in specs:
            res = outcomes[spec.label]
            agg = (
                M.aggregate(res["item_recs"], ks) if res["item_recs"] else None
            )
            cats = (
                M.per_category(res["item_recs"], ks)
                if res["item_recs"] else None
            )
            sess = (
                {
                    "overall": M.aggregate(res["sess_recs"], ks),
                    "categories": M.per_category(res["sess_recs"], ks),
                }
                if res["sess_recs"] else None
            )
            applied, verify = _verify_application(
                spec, res["rows"], res.get("explains") or {}
            )
            report["specs"][spec.label] = {
                "label": spec.label,
                "patch": res["patch"],
                "applied": applied,
                "verify": verify,
                "error": res.get("error"),
                "notes": list(spec.notes),
                "overall": agg,
                "categories": cats,
                "session": sess,
            }

        # ---- per-question paired rows -------------------------------------
        for tv in views:
            cid = category_id_of(tv.raw)
            qrow: Dict[str, Any] = {
                "task_id": tv.task_id,
                "conv_id": tv.group_id,
                "category": tv.category,
                "category_id": cid,
                "answerable": tv.answerable,
                "gold": dict(tv.gold_item),
                "gold_session": dict(tv.gold_session),
                "arms": {},
            }
            for spec in specs:
                row = (outcomes.get(spec.label) or {}).get("rows", {}).get(
                    tv.task_id
                )
                qrow["arms"][spec.label] = row if row is not None else {
                    "status": "not_run", "error": "spec produced no row"
                }
            report["questions"].append(qrow)

        # ---- paired substrate (McNemar counts + rank deltas) ---------------
        if len(specs) >= 2:
            a, b = specs[0].label, specs[1].label
            rows_a = outcomes[a]["rows"]
            rows_b = outcomes[b]["rows"]
            report["paired"] = {
                "first": a,
                "second": b,
                "any_at_k": {
                    str(k): _confusion(rows_a, rows_b, golds, k) for k in ks
                },
                "metric_delta": _metric_deltas(
                    report["specs"][a].get("overall"),
                    report["specs"][b].get("overall"),
                    ks,
                ),
            }
        return report
    finally:
        arm.close()


def _verify_application(
    spec: ArmSpec,
    rows: Mapping[str, Mapping[str, Any]],
    explains: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Did the pipeline print the requested configuration?

    Reads ``policy_lanes``/``lane_statuses`` captured from real results —
    never assumes.  Returns ``(applied, verify)`` where ``applied`` is
    True/False/``"unverifiable"``."""
    patch = spec.patch
    observed_lanes: Optional[List[str]] = None
    lane_statuses: Optional[Dict[str, Any]] = None
    n_rows = 0
    for row in rows.values():
        pl = row.get("policy_lanes")
        if pl is not None:
            observed_lanes = list(pl)
            n_rows += 1
        ls = row.get("lane_statuses")
        if ls is not None and lane_statuses is None:
            lane_statuses = dict(ls)
    verify: Dict[str, Any] = {
        "observed_policy_lanes": observed_lanes,
        "observed_lane_statuses": lane_statuses,
        "rows_with_policy_echo": n_rows,
        "disabled_lanes_seen": [],
    }
    if patch is None or not (patch.lanes_disabled or patch.policy_doc
                             or patch.feature_weights):
        verify["basis"] = "baseline arm — nothing to apply"
        return True, verify
    if patch.errors:
        verify["basis"] = "patch errors recorded"
        return False, verify

    # feature-weight overrides verify through explain items'
    # detail.weights — the applied-weight entries printed per feature
    # that fired.  A feature absent from every item is UNOBSERVED (it
    # never fired — nothing to compare), not a mismatch.
    if patch.feature_weights:
        observed_weights: Dict[str, float] = {}
        for expl in (explains or {}).values():
            for it in (expl or {}).get("items") or []:
                w = ((it.get("detail") or {}).get("weights")) or {}
                for name, val in w.items():
                    observed_weights.setdefault(str(name), float(val))
        verify["observed_feature_weights"] = {
            k: observed_weights.get(k) for k in patch.feature_weights
        } if observed_weights else None
        if observed_weights:
            mismatched = [
                k for k, v in patch.feature_weights.items()
                if k in observed_weights
                and float(observed_weights[k]) != float(v)
            ]
            unobserved = [
                k for k in patch.feature_weights
                if k not in observed_weights
            ]
            verify["unobserved_weight_features"] = unobserved
            if mismatched:
                verify["basis"] = (
                    f"feature weights differ on {mismatched}"
                )
                return False, verify
            if unobserved and not (
                patch.lanes_disabled or patch.policy_doc
            ):
                verify["basis"] = (
                    f"features {unobserved} never fired — weight arm "
                    "unverifiable"
                )
                return "unverifiable", verify
        elif not (patch.lanes_disabled or patch.policy_doc):
            verify["basis"] = (
                "no item weights captured — weight arm unverifiable"
            )
            return "unverifiable", verify

    if observed_lanes is None:
        if patch.feature_weights and not (
            patch.lanes_disabled or patch.policy_doc
        ):
            # weights verified above; policy echo not needed
            verify["basis"] = "feature weights echoed by pipeline"
            return True, verify
        verify["basis"] = "no explain policy echo captured"
        return "unverifiable", verify
    seen = [l for l in patch.lanes_disabled if l in observed_lanes]
    verify["disabled_lanes_seen"] = seen
    if patch.lanes_disabled and seen:
        verify["basis"] = "disabled lane still present in policy tuple"
        return False, verify
    if patch.policy_doc is not None and "lanes" in patch.policy_doc:
        want = [str(x) for x in patch.policy_doc["lanes"]]
        want = [x for x in want if x not in patch.lanes_disabled]
        if observed_lanes != want:
            verify["basis"] = "policy lanes differ from doc"
            verify["expected_lanes"] = want
            return False, verify
    verify["basis"] = "applied configuration echoed by pipeline"
    return True, verify


def _metric_deltas(
    a: Optional[Mapping[str, Any]],
    b: Optional[Mapping[str, Any]],
    ks: Sequence[int],
) -> Dict[str, Any]:
    """``second − first`` deltas over the shared metric keys."""
    keys = {"mrr@10", "ndcg@10", "zero_rate", "abstain_rate",
            "false_abstention", "correct_refusal"}
    for k in set(ks) | {10}:
        keys.update({f"any@{k}", f"all@{k}", f"prop@{k}"})
    out: Dict[str, Any] = {}
    for key in sorted(keys):
        va = (a or {}).get(key)
        vb = (b or {}).get(key)
        out[key] = {
            "first": va,
            "second": vb,
            "delta": (vb - va) if (va is not None and vb is not None) else None,
        }
    lat_a = (a or {}).get("latency_ms") or {}
    lat_b = (b or {}).get("latency_ms") or {}
    out["latency_ms"] = {
        "first": {"p50": lat_a.get("p50"), "p95": lat_a.get("p95")},
        "second": {"p50": lat_b.get("p50"), "p95": lat_b.get("p95")},
    }
    return out


def arm_report_kwargs(
    *,
    timeout_ms: Optional[float] = None,
    pool_limit: Optional[int] = None,
    settle_timeout_s: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble ``VerbatimArm`` kwargs for the CLI/tools — only the keys
    a caller actually pinned land in the manifest (defaults stay the
    arm's own, e.g. the 2000 ms eval-profile deadline)."""
    kw: Dict[str, Any] = {}
    if timeout_ms is not None:
        kw["timeout_ms"] = float(timeout_ms)
    if pool_limit is not None:
        kw["pool_limit"] = int(pool_limit)
    if settle_timeout_s is not None:
        kw["settle_timeout_s"] = float(settle_timeout_s)
    kw.update(dict(extra or {}))
    return kw


__all__ = [
    "ArmSpec",
    "CONSTANTS_TAG",
    "DETERMINISM_EXEMPT",
    "ForensicVerbatimArm",
    "PolicyPatch",
    "QuestionRow",
    "SCHEMA",
    "arm_report_kwargs",
    "category_id_of",
    "dataset_block",
    "delivered_units",
    "env_block",
    "gold_ranks",
    "paired_run",
    "ref_sessions",
    "run_spec",
    "task_views",
    "write_report",
]
