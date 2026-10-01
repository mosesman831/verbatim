"""S3 — weighted reciprocal-rank fusion (``rrf/v7``, ``provisional/v7-r0``).

Implements SPEC_V7 §32.3 / V7-10.01–05. This stage *reorders* the eligible
candidates the lanes produced; it never admits, generates, or drops
eligibility (a unit absent from every lane stays absent).

Contract:

- ``rrf(d) = Σ_l  w_l / (k + rank_l(d))`` with ``k = 60`` and 1-based ranks
  (V7-10.01). Raw lane scores are never summed across lanes — only ranks.
- **Rank space only (V75-03.03/J17):** a lane weight enters exclusively as
  the numerator of ``w/(k + rank)`` — equivalently as the divisor form
  ``1/(k + rank/w)``. Nothing in this module multiplies a lane's raw score
  by a weight: heterogeneous lane score scales (a BM25F ~10 vs a cosine
  ~0.7) are never blended, which is exactly the Hindsight score-space
  collapse the spec forbids (``notes/d1-hindsight-code.md``).
- A lane absent from ``weights`` contributes the declared default ``1.0``
  (V7-10.01: "all 1.0 unless tuned on the dev split").
- A candidate absent from a lane contributes ``0`` from that lane.
- Only lanes reporting ``ok`` or ``partial`` contribute candidates. A lane
  reporting ``skipped`` / ``unavailable`` / ``deadline`` is honest degradation
  (V7-04.03); trusting candidates emitted under those statuses would let a
  degraded path smuggle results into the order. Ignored lanes are counted in
  ``FusedList.stats["lanes_ignored"]``.
- Lane-rank provenance is retained per candidate (V7-10.05) in
  ``FusedCandidate.lane_ranks`` and in ``signals["_lanes"]``.
- Ordering is a pure function of the inputs: ``(-rrf, source_id, unit_id)``.
  No clock, no store, no randomness.
- V7-10.04's rerank-pool cut is available via the optional ``limit`` keyword
  (the pipeline passes the budget's R); truncation is reported, never silent.

Weak-lane gates (V75-04.03, machinery — nothing gated by default):

- ``lane_gates={lane: g}`` caps a lane's contribution to ranks ``<= g``;
  ``g = 0`` removes the lane from fusion entirely. Gated-out rows never
  enter the fused pool (no zero-score tail), and the drops are counted in
  ``stats["lanes_gated"][lane]["dropped"]``.
- J13 fallback lane: when every lane that produced candidates is gated,
  the gates are bypassed (``stats["gates_bypassed"]``) so a gated lane
  still answers standalone.

Facet coverage bonus (V75-03.04 / V7-05.13 — ``facet_bonus/v1``):

- Candidates surfaced by a decomposition facet carry
  ``signals["facet"] = <facet index>`` (the pipeline tags them). A unit
  covered by ``>= 2`` distinct facets earns a bounded rank-space term::

      bonus(d) = min(FACET_BONUS_BETA * (covered(d) - 1), FACET_BONUS_MAX)
                 / (k + min_rank(d))

  i.e. each additional covering facet acts like a fractional extra lane at
  the unit's best rank. ``FACET_BONUS_BETA * (covered - 1)`` is hard-capped
  at ``FACET_BONUS_MAX = 1.0`` so the bonus can never exceed one default
  lane's contribution at the same rank. ``stats["facet_bonus"]["applied"]``
  counts the units bonused; the pipeline declares
  ``coverage.facets["bonus"]`` only when that count is non-zero (J08).

Signal merge discipline (feeding S4 without leaking lane-local semantics):

- Lane-independent unit facts (``_UNIT_FACT_KEYS``) merge flat,
  first-writer-wins — any lane reporting ``speaker_canon`` asserts the same
  fact about the same unit.
- Matched-term/canon sets union across lanes; per-term idf dicts union with
  the max observed value; feature-shaped signals (``phrase``, ``event_pred``,
  ``ident_exact``, ``corroboration``) take the max across lanes.
- Every contributing lane's *complete* raw signal dict, raw score, and rank
  are preserved under ``signals["_lanes"][lane]`` — lossless provenance for
  ``explain=True`` (V7-05.15).
- Any other key stays lane-local as ``signals["_lanes"][lane]["signals"][k]``
  so e.g. a ``score`` signal on two lanes can never alias each other.

``detect_constant_signals`` implements V7-10.03: a signal key whose modal
value covers ≥ ``threshold`` (default 0.9) of all candidate rows is constant
and contributes zero to downstream ranking; the caller reports it in
``coverage``/``explain`` and passes it to ``score_candidates(suppress=...)``.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from collections.abc import Mapping
from dataclasses import replace as dc_replace
from typing import Any, Iterable, Optional

from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    CandidateV7,
    FusedCandidate,
    LaneOutput,
)
from .turn_position import SessionIndex, member_row, member_sort_key

#: Fusion stage identifier recorded in stats/details.
FUSION_MODEL_ID = "rrf/v7"

#: RRF rank constant (§32.3).
RRF_K = 60

#: Default lane weight when a lane is absent from the weights table
#: (V7-10.01: all 1.0 unless tuned on the dev split).
DEFAULT_LANE_WEIGHT = 1.0

#: V7-10.03 constant-signal threshold.
CONSTANT_SIGNAL_THRESHOLD = 0.9

#: Facet coverage bonus version tag (V75-03.04): emitted in stats and in
#: ``coverage.facets["bonus"]`` when a bonus is actually applied.
FACET_BONUS_ID = "facet_bonus/v1"

#: Weight per additional covering facet (beyond the first). With the
#: pool cap of <= 3 facets the total facet weight stays <= 1.0.
FACET_BONUS_BETA = 0.5

#: Hard bound on the summed facet weight — the bonus can never exceed one
#: default-weight lane's contribution at the same rank (V75-03.04).
FACET_BONUS_MAX = 1.0

# ---------------------------------------------------------------------------
# V8 context propagation/injection (SPEC_V8 §06, §21.1, §23 context.*)
# ---------------------------------------------------------------------------

#: Context stage identifier recorded in stats/details (SPEC_V8 §06).
CONTEXT_MODEL_ID = "context/v8"

#: ``context.mode`` arm values (§23). ``combined`` = propagate+inject
#: (default), ``propagate`` = score boosts only, ``inject`` = injection
#: only, ``field``/``off`` disable the stage.
CTX_MODE_COMBINED = "combined"
CTX_MODE_PROPAGATE = "propagate"
CTX_MODE_INJECT = "inject"
CTX_MODE_OFF = "off"
CTX_MODE_FIELD = "field"

#: §23 context.* defaults.
DEFAULT_CONTEXT_W = 0.7
DEFAULT_CONTEXT_WINDOW = 1
DEFAULT_M_CTX = 50
CTX_INJECT_CAP_FACTOR = 2  # injected candidates per lane ≤ 2*M_ctx (§21.1)

#: §23 dense.N_d default (V8-08.04 — guaranteed fused-pool slots).
#: SPEC_V8_5 §06 don't-ship: the pack measured 0/137 unique gold rescues
#: at N_d>0 — the shipped prior is 0 (reservation disabled).
DEFAULT_DENSE_SLOTS = 0

#: §23 facets.whole_share default (V8-10.02 — reserved facet slots).
DEFAULT_FACET_WHOLE_SHARE = 0.5

#: §23 fusion.lex_anchor default divisor (V8-11.03, rank-space form of
#: the V75-03.03 anchor: ``1 / (k + rank/divisor)`` for lex ranks ≤ 3).
DEFAULT_LEX_ANCHOR_DIVISOR = 2.0

#: §23 fusion.dense_form (V8-08.06) — the fused-score form. ``"rrf"``
#: (default) keeps every lane term in rank space; ``"combsum"`` is the
#: Q1 research arm: the lexical lane contributes lane-max-normalized
#: BM25 (``bm25_norm``) and the dense lane contributes ``α·cos`` — both
#: score-space — while every other lane keeps its ``w/(k+rank)`` rank
#: term (V75-03.03: a lane weight stays in rank space, never a
#: raw-score multiplier).
DEFAULT_DENSE_FORM = "rrf"
DEFAULT_DENSE_FORM_ALPHA = 1.2
_DENSE_FORMS = {
    "rrf": "rrf",
    "rank": "rrf",
    "rank_space": "rrf",
    "combsum": "combsum",
    "comb_sum": "combsum",
    "score_sum": "combsum",
    "sum": "combsum",
}

#: Lane-name strings referenced by the V8 arms.
LEX_LANE = "lex"
DENSE_LANE = "dense"

#: Turn-view keys consulted in candidate signals for the §21.1
#: same-session, sequence-adjacent neighbor definition.
_SESSION_KEY = "session_id"
_SEQ_KEY = "seq"
_KIND_TURN = "turn"
_KIND_SENTENCE = "sentence_window"
_SORT_SEQ_SENTINEL = 1 << 62

#: Lane statuses whose candidate lists are trustworthy enough to fuse.
_FUSABLE_STATUSES = frozenset({"ok", "partial"})

#: Lane-independent unit properties: merged flat, first writer wins.
_UNIT_FACT_KEYS = frozenset({
    "speaker_canon", "perspective", "kind", "seq", "session_id",
    "occurred_start_us", "occurred_end_us", "occurred_precision",
    "occurred_source", "recorded_at_us", "lifecycle", "derived",
    "proof_count", "identifiers", "event_predicates",
    "unit_terms", "unit_canons", "family_count",
})

#: Set-valued match evidence: union across lanes.
_SET_MERGE_KEYS = frozenset({
    "matched_terms", "matched_canons", "matched_terms_ctx",
})

#: Dict-valued idf evidence: union, keeping the max value per term.
_DICT_MERGE_KEYS = frozenset({
    "term_idf", "entity_idf", "ctx_term_idf",
})

#: Feature-shaped signals: keep the max across lanes.
_MAX_MERGE_KEYS = frozenset({
    "phrase", "event_pred", "ident_exact", "corroboration",
})

#: V8-20.04 cross-lane explain/provenance keys: merged flat, first writer
#: wins. These are the cross-worker signal contract fields that must
#: surface on fused candidates for downstream explain/coverage:
#: ``ctx_from``/``pre_ctx_score``/``ctx_injected`` (context, §06),
#: ``dense_slot`` (V8-08.04), ``joint`` (V8-10.03), ``facet``
#: (V8-10.02 — the candidate's facet index, not a query-side list),
#: ``mention_interval``/``rescue`` (temporal worker contract).
_EXPLAIN_KEYS = frozenset({
    "ctx_from", "pre_ctx_score", "ctx_injected", "dense_slot",
    "joint", "facet", "rescue", "mention_interval",
})

#: Signals bucket holding lossless per-lane provenance.
LANES_SIGNALS_KEY = "_lanes"


class FusedList(list):
    """``list[FusedCandidate`` plus run metadata for coverage/explain.

    ``stats`` echoes the applied contract (``k``, ``weights``), honest
    coverage (rows in, per-lane contribution, ignored degraded lanes,
    duplicates dropped, truncation), and the V7-10.03 constant-signal set.
    """

    def __init__(self, rows: Iterable = (), stats: Optional[dict] = None) -> None:
        super().__init__(rows)
        self.stats: dict = dict(stats or {})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _lane_key(name: Any) -> str:
    """Normalize a lane identifier (``LaneName`` or ``str``) to its value."""
    return str(getattr(name, "value", name))


def _norm_weights(weights: Optional[Mapping]) -> dict:
    out: dict = {}
    for key, value in dict(weights or {}).items():
        try:
            out[_lane_key(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _norm_gates(lane_gates: Optional[Mapping]) -> dict:
    """Normalize ``lane_gates`` to ``{lane_str: max_rank}`` (0 = exclude).

    Accepted values mirror the policy document forms: ``"exclude"``, a
    non-negative int, or ``{"top_n": n}``. Malformed entries raise
    ``ValueError`` — a declared gate that cannot be parsed must fail
    loudly, never silently un-gate a lane.
    """

    out: dict = {}
    for key, value in dict(lane_gates or {}).items():
        lane = _lane_key(key)
        if isinstance(value, str) and value.strip() == "exclude":
            out[lane] = 0
            continue
        if isinstance(value, Mapping):
            value = value.get("top_n")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"lane_gates[{lane!r}] must be 'exclude', a non-negative "
                f"int, or {{'top_n': n}} — got {value!r}"
            )
        out[lane] = int(value)
    return out


def _status_key(status: Any) -> str:
    return str(getattr(status, "value", status))


def _rank_of(cand: Any, pos: int) -> int:
    """1-based rank: trust ``cand.rank`` when it is a positive int, else the
    candidate's position in the lane list."""
    rank = getattr(cand, "rank", None)
    if isinstance(rank, bool):
        return pos + 1
    if isinstance(rank, int) and rank >= 1:
        return rank
    return pos + 1


def _policy_arm(policy: Any, attr: str, *keys: str) -> Any:
    """Resolve one §23 arm off a policy carrier object.

    Carrier order follows the lexical-lane convention: a direct
    attribute on the policy (``fusion_dense_form``), then a
    ``params``/``knobs``/``arms`` mapping keyed by the dotted or
    underscored flag name. ``None`` is absent everywhere. The canonical
    :func:`retrieval.v7.policy.policy_param` channel is the ``params``
    map; the extra carriers keep hand-built fixtures honest.
    """
    if policy is None:
        return None
    val = getattr(policy, attr, None)
    if val is not None:
        return val
    for carrier in ("params", "knobs", "arms"):
        mapping = getattr(policy, carrier, None)
        if isinstance(mapping, Mapping):
            for key in keys:
                val = mapping.get(key)
                if val is not None:
                    return val
    return None


def _resolve_dense_form(
    dense_form: Any, alpha: Any, policy: Any
) -> tuple:
    """Normalize the ``fusion.dense_form`` arm (V8-08.06).

    Returns ``(form, alpha)`` — ``form`` ∈ ``{"rrf", "combsum"}``;
    ``alpha`` is the CombSUM cosine weight (prior
    :data:`DEFAULT_DENSE_FORM_ALPHA`), validated whenever the form is
    combsum or the caller declared it. ``dense_form`` accepts ``"rrf"``
    /``"combsum"`` spellings, ``True`` → combsum, ``False``/``"off"``
    → rrf, or a mapping ``{"form": …, "alpha": …}``. Undeclared values
    fall through to the ``policy`` carrier, then the §23 priors.
    Malformed declarations raise ``ValueError`` — a mistyped arm must
    never silently re-form the fused score.
    """
    raw = dense_form
    if raw is None:
        raw = _policy_arm(
            policy, "fusion_dense_form", "fusion.dense_form", "dense_form"
        )
    form_map: Optional[Mapping] = None
    if raw is None or raw is False:
        form = "rrf"
    elif raw is True:
        form = "combsum"
    elif isinstance(raw, Mapping):
        form_map = raw
        inner = raw.get("form", "combsum")
        if not isinstance(inner, str):
            raise ValueError(
                f"fusion.dense_form 'form' must be a string, got {inner!r}"
            )
        form = _DENSE_FORMS.get(inner.strip().lower())
        if form is None:
            raise ValueError(
                f"unknown fusion.dense_form {inner!r} — "
                f"expected one of {sorted(_DENSE_FORMS)}"
            )
    elif isinstance(raw, str):
        key = raw.strip().lower()
        if key in ("off", "none", "disabled"):
            form = "rrf"
        else:
            form = _DENSE_FORMS.get(key)
            if form is None:
                raise ValueError(
                    f"unknown fusion.dense_form {raw!r} — "
                    f"expected one of {sorted(_DENSE_FORMS)} or off"
                )
    else:
        raise ValueError(
            f"fusion.dense_form must be a form name, bool, or mapping — "
            f"got {raw!r}"
        )
    raw_alpha = alpha
    if raw_alpha is None and form_map is not None:
        raw_alpha = form_map.get("alpha")
    if raw_alpha is None:
        raw_alpha = _policy_arm(
            policy,
            "fusion_dense_form_alpha",
            "fusion.dense_form_alpha",
            "dense_form_alpha",
        )
    resolved_alpha: Optional[float] = None
    if raw_alpha is not None or form == "combsum":
        try:
            resolved_alpha = float(
                DEFAULT_DENSE_FORM_ALPHA if raw_alpha is None else raw_alpha
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"fusion.dense_form_alpha must be a number, "
                f"got {raw_alpha!r}"
            )
        if not math.isfinite(resolved_alpha) or resolved_alpha <= 0.0:
            raise ValueError(
                f"fusion.dense_form_alpha must be finite and > 0, "
                f"got {raw_alpha!r}"
            )
    return form, resolved_alpha


def _canonicalize(value: Any) -> Any:
    """Deterministic canonical form for hashing/equality of signal values."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {
            str(k): _canonicalize(value[k])
            for k in sorted(value, key=lambda x: repr(x))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonicalize(v) for v in value), key=repr)
    return repr(value)


def _value_key(value: Any) -> tuple:
    """Hashable canonical key for one signal value (constant detection)."""
    # Scalar fast path — ``_canonicalize`` returns these unchanged
    # (non-finite floats as ``repr``) and they are always hashable.
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return ("v", value)
    if isinstance(value, float):
        return ("v", value if math.isfinite(value) else repr(value))
    canon = _canonicalize(value)
    try:
        hash(canon)
        return ("v", canon)
    except TypeError:
        return ("r", repr(canon))


def _merge_signals(dst: dict, lane: str, sig: Mapping) -> None:
    """Merge one lane-candidate's raw signals into the unit's merged view."""
    if not isinstance(sig, Mapping):
        return
    lanes_view = dst.setdefault(LANES_SIGNALS_KEY, {})
    lane_entry = lanes_view.setdefault(lane, {})
    lane_entry["signals"] = dict(sig)
    for key, value in sig.items():
        if key in _UNIT_FACT_KEYS:
            dst.setdefault(key, value)
        elif key in _SET_MERGE_KEYS:
            cur = dst.get(key)
            base = set(cur) if isinstance(cur, (set, frozenset, list, tuple)) else set()
            if isinstance(value, Mapping):
                # a {term: detail} dict merges its keys (the term names)
                base.update(value.keys())
            elif isinstance(value, (set, frozenset, list, tuple)):
                base.update(value)
            else:
                try:
                    base.add(value)
                except TypeError:
                    base.add(repr(value))
            dst[key] = base
        elif key in _DICT_MERGE_KEYS:
            cur = dst.get(key)
            merged = dict(cur) if isinstance(cur, Mapping) else {}
            if isinstance(value, Mapping):
                for tk, tv in value.items():
                    prev = merged.get(tk)
                    if prev is None:
                        merged[tk] = tv
                    else:
                        try:
                            merged[tk] = max(float(prev), float(tv))
                        except (TypeError, ValueError):
                            pass
            dst[key] = merged
        elif key in _MAX_MERGE_KEYS:
            prev = dst.get(key)
            if prev is None:
                dst[key] = value
            else:
                try:
                    dst[key] = max(float(prev), float(value))
                except (TypeError, ValueError):
                    pass
        elif key in _EXPLAIN_KEYS:
            # V8-20.04 explain pass-through — first writer wins; a
            # falsy-but-real value (``facet=0``, ``pre_ctx_score=0.0``)
            # still counts as written.
            if key not in dst:
                dst[key] = value
        # Anything else remains available under _lanes[lane]["signals"]
        # only — never silently aliased across lanes.


def _freeze_merged_signals(signals: dict) -> dict:
    """Deterministic final form: set merges become sorted tuples."""
    out: dict = {}
    for key, value in signals.items():
        if isinstance(value, (set, frozenset)):
            out[key] = tuple(sorted(value, key=repr))
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# V8 context propagation/injection — SPEC_V8 §06, §21.1 (normative)
# ---------------------------------------------------------------------------


def _ctx_mode(value: Any) -> str:
    """Normalize ``context.mode``; unknown values fail loudly (§23 arms
    are declared surface — silently ignoring a typo'd arm would hide
    the misconfiguration)."""
    mode = str(getattr(value, "value", value)).strip().lower()
    if mode in ("propagate+inject", "propagate_inject", "on", "true"):
        return CTX_MODE_COMBINED
    if mode in ("propagate-only", "propagate_only", "boost"):
        return CTX_MODE_PROPAGATE
    if mode in ("inject-only", "inject_only"):
        return CTX_MODE_INJECT
    if mode in (
        CTX_MODE_COMBINED, CTX_MODE_PROPAGATE, CTX_MODE_INJECT,
        CTX_MODE_OFF, CTX_MODE_FIELD,
    ):
        return mode
    raise ValueError(f"context.mode must be combined|propagate|inject|off — got {value!r}")


def _turn_view(signals: Any) -> Optional[tuple]:
    """§21.1 unit view: ``(session_id, seq)`` for a *turn* unit, else
    ``None``. Candidates without a usable turn view are neither context
    sources nor context targets — sentence-window units are excluded on
    both sides of the algorithm (§21.1)."""
    if not isinstance(signals, Mapping):
        return None
    kind = signals.get("kind")
    if kind is not None and str(kind) != _KIND_TURN:
        return None
    session = signals.get(_SESSION_KEY)
    seq = signals.get(_SEQ_KEY)
    if session is None or seq is None:
        return None
    if isinstance(seq, bool):
        return None
    try:
        seq_i = int(seq)
    except (TypeError, ValueError):
        return None
    return (str(session), seq_i)


def _ctx_sort_key(cand: Any) -> tuple:
    """§21.1 deterministic lane re-rank after propagation:
    propagated score desc, then ``(session_id, seq, unit_id)``."""
    sig = getattr(cand, "signals", None) or {}
    session = sig.get(_SESSION_KEY)
    seq = sig.get(_SEQ_KEY)
    try:
        seq_i = int(seq)
    except (TypeError, ValueError):
        seq_i = _SORT_SEQ_SENTINEL
    try:
        score = float(getattr(cand, "raw_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return (-score, "" if session is None else str(session), seq_i,
            str(getattr(cand, "unit_id", "")))


def _neighbor_row(row: Any) -> Optional[dict]:
    """Normalize one neighbor-source row →
    :func:`verbatim.retrieval.v7.turn_position.member_row`.

    Accepted shapes (the pipeline's neighbor fetch may produce either):
    ``(unit_id, session_id, seq[, kind][, source_id][, revision])``
    tuples, or mappings with the same keys plus the V85-03 ordering
    columns (``occurred_start_us``, ``recorded_at_us``, ``byte_start``).
    Rows that are not turn units (``kind != 'turn'`` when declared) or
    that lack ``unit_id``/``session_id`` return ``None`` — the inventory
    may only contain eligible turns (§21.1). A missing ``seq`` no longer
    drops the row: effective position orders it by the remaining
    components (V85-03).
    """
    return member_row(row)


def _eligible_member(eligible: Any, unit_id: str) -> bool:
    """Fail-closed eligibility check for injected units (§21.1 —
    neighbors must be eligible). ``None`` means the caller guarantees
    the neighbor source is already eligibility-filtered."""
    if eligible is None:
        return True
    if callable(eligible):
        try:
            return bool(eligible(unit_id))
        except Exception:
            return False
    try:
        return unit_id in eligible
    except TypeError:
        return True


def _resolve_ctx_arg(context: Any) -> dict:
    """Normalize the ``context`` arm accepted by ``rrf_fuse``.

    Shapes: a mode string, ``True`` (combined defaults), or a mapping
    with ``mode``/``w``/``W``/``M_ctx``/``neighbors``/``eligible``.
    """
    out = {
        "mode": CTX_MODE_COMBINED,
        "w": DEFAULT_CONTEXT_W,
        "W": DEFAULT_CONTEXT_WINDOW,
        "M_ctx": DEFAULT_M_CTX,
        "neighbors": None,
        "eligible": None,
    }
    if context is True or context is None:
        return out
    if isinstance(context, str):
        out["mode"] = context
        return out
    if isinstance(context, Mapping):
        for key in ("mode", "w", "W", "M_ctx", "neighbors", "eligible"):
            if key in context:
                out[key] = context[key]
        return out
    raise TypeError(f"context must be a mode string, True, or a mapping — got {type(context)!r}")


def propagate_context(
    outputs: Iterable[LaneOutput],
    neighbors: Any = None,
    *,
    mode: Any = CTX_MODE_COMBINED,
    w: float = DEFAULT_CONTEXT_W,
    W: int = DEFAULT_CONTEXT_WINDOW,
    M_ctx: int = DEFAULT_M_CTX,
    eligible: Any = None,
) -> tuple:
    """SPEC_V8 §06/§21.1 — propagate/inject turn context **per lane**.

    For each lane output (before fusion consumes lane ranks):

    - every nominated turn target ``u`` gets ``s'(u) = s(u) + w *
      max{s(v)}`` over nominated eligible neighbors ``v`` within ``W``
      **effective positions** of ``u`` in the same session — reading the
      **original** pre-context scores only (no cascading);
    - every non-nominated eligible turn ``x`` within ``W`` positions of a
      top-``M_ctx`` nominated unit may be injected with
      ``score(x) = w * max{s(v)}`` over nominated neighbors, capped at
      ``2 * M_ctx`` injected units per lane;
    - injected candidates carry ``signals["ctx_from"]`` (the maximizing
      nominated neighbor) and ``signals["pre_ctx_score"] = 0.0``;
      boosted candidates carry ``pre_ctx_score`` = original score and
      ``ctx_from`` = argmax neighbor;
    - lane candidates are then re-ranked by propagated score with the
      deterministic ``(session_id, seq, unit_id)`` tie-break, and
      ``rank`` is re-stamped 1..n (RRF consumes these ranks).

    **Effective position (V85-03, SPEC_V8_5 §V85-03):** a member's
    position is its index among the session's eligible turn rows ordered
    by ``(seq, occurred_start_us, recorded_at_us, byte_start, unit_id)``.
    The spec's ``1 <= |Δseq| <= W`` rule is the special case where
    ``seq`` IS the dense per-session ordinal; stores that emit one
    source per turn carry ``seq = 0`` on every turn, which silently
    zeroed the raw-delta window. Positions are computed over the flat
    inventory regardless of how ``seq`` was assigned.

    ``neighbors`` is either an adjacency mapping ``{unit_id: [rows]}``
    (caller-declared neighbor relation — session/eligibility checks
    still apply, the window is the caller's) or a flat inventory of
    eligible turn rows
    (``(unit_id, session_id, seq[, kind, source_id, revision])`` or
    mappings, plus the ordering columns) over which this stage builds
    the position index. Scope/session/generation fencing and
    eligibility are the CALLER's query contract — this stage never
    invents units. ``eligible`` (when given) filters the position index:
    neighbors are defined over the eligible member set ``E`` (§21.1), so
    held/fenced members do not occupy positions.

    Returns ``(outputs, stats)``; stats carries ``injected``/``boosted``
    per lane for ``coverage.context`` (§20). Lanes already stamped
    ``stats["context_applied"]`` are skipped (idempotent).
    """
    mode = _ctx_mode(mode)
    outputs = [o for o in (outputs or ())]
    stats: dict = {
        "model": CONTEXT_MODEL_ID,
        "mode": mode,
        "w": float(w),
        "W": int(W) if isinstance(W, int) and not isinstance(W, bool) else W,
        "M_ctx": M_ctx,
        "injected": 0,
        "boosted": 0,
        "lanes": {},
    }
    if mode in (CTX_MODE_OFF, CTX_MODE_FIELD):
        stats["applied"] = False
        return outputs, stats
    try:
        w = float(w)
    except (TypeError, ValueError):
        raise ValueError(f"context.w must be numeric, got {w!r}")
    if not (0.0 < w < 1.0):
        raise ValueError(f"context.w must lie in (0, 1) — got {w!r}")
    if isinstance(W, bool) or not isinstance(W, int) or W < 0:
        raise ValueError(f"context.W must be a non-negative int — got {W!r}")
    if isinstance(M_ctx, bool) or not isinstance(M_ctx, int) or M_ctx < 0:
        raise ValueError(f"context.M_ctx must be a non-negative int — got {M_ctx!r}")

    # Neighbor source: adjacency {uid: [rows]} or flat eligible inventory.
    # V85-03 — a flat inventory is indexed by *effective position*
    # (SessionIndex); an adjacency mapping is the caller-declared
    # neighbor relation and is trusted as the window already applied.
    adj: Optional[Mapping] = neighbors if isinstance(neighbors, Mapping) else None
    if adj is not None:
        inv_rows: list = []
        for rows in adj.values():
            inv_rows.extend(rows or ())
        index: Optional[SessionIndex] = None
    elif neighbors is not None:
        inv_rows = list(neighbors)
        index = SessionIndex.build(inv_rows, eligible=eligible)
    else:
        inv_rows = []
        index = SessionIndex.build(())
    stats["applied"] = True
    stats["neighbor_rows"] = len(inv_rows)
    if index is not None:
        stats["member_rows"] = len(index.pos_of)
        stats["sessions"] = len(index.sessions)
        if index.skipped_rows:
            stats["skipped_rows"] = index.skipped_rows
        if index.ineligible_rows:
            stats["ineligible_rows"] = index.ineligible_rows

    new_outputs: list = []
    for out in outputs:
        lane_stats = (out.stats or {})
        if lane_stats.get("context_applied"):
            new_outputs.append(out)
            continue
        cands = list(out.candidates or ())
        lane_name = _lane_key(getattr(out, "lane", ""))
        # Original pre-context scores — boosts never cascade (§21.1:
        # a propagated score is never a neighbor-source score).
        base = {id(c): float(getattr(c, "raw_score", 0.0) or 0.0) for c in cands}
        nom_ids = {str(getattr(c, "unit_id", "")) for c in cands}
        # Nominated index: unit_id -> [candidates] for participating
        # turn units. A candidate participates when its signals declare
        # a turn view (adjacency mode — the caller's lists define the
        # relation, the view supplies the session boundary) or when it
        # is a member of the position index (inventory mode — the
        # eligibility-fenced member list is authoritative).
        nom_by_uid: dict = {}
        views: dict = {}
        part: dict = {}
        for c in cands:
            uid = str(getattr(c, "unit_id", ""))
            sig = getattr(c, "signals", None) or {}
            kind = sig.get("kind")
            ok_kind = kind is None or str(kind) == _KIND_TURN
            v = _turn_view(sig)
            views[id(c)] = v
            if adj is not None:
                part[id(c)] = ok_kind and v is not None
            else:
                part[id(c)] = ok_kind and uid in index.pos_of
            if part[id(c)]:
                nom_by_uid.setdefault(uid, []).append(c)

        def _member_rows(uid: str, session: Optional[str]):
            """Neighbor member rows of ``uid`` — ±W effective positions
            over the session index (inventory mode), or the caller's
            declared adjacency list filtered to ``session`` (adjacency
            mode)."""
            if adj is not None:
                for row in adj.get(uid, ()) or ():
                    rv = _neighbor_row(row)
                    if rv is None:
                        continue
                    if session is not None and rv[_SESSION_KEY] != session:
                        continue
                    yield rv
                return
            for rv in index.neighbors(uid, W):
                yield rv

        def _best_neighbor(uid: str, session: Optional[str],
                           extra: Iterable[str] = ()):
            """(max original score, argmax unit_id) over nominated
            neighbors; argmax tie-break = the member ordering tuple
            (session order — the §21.1 ``(seq, unit_id)`` rule
            generalized to effective position). ``extra`` names
            nominated units that reach ``uid`` without appearing in its
            own row list (adjacency callers may declare asymmetric
            edges)."""
            best_s: Optional[float] = None
            best_uid: Optional[str] = None
            best_key: Optional[tuple] = None
            for rv in _member_rows(uid, session):
                xu = rv["unit_id"]
                if xu == uid:
                    continue
                for src in nom_by_uid.get(xu, ()):
                    s = base[id(src)]
                    key = rv["key"]
                    if best_s is None or s > best_s or (
                        s == best_s and key < best_key
                    ):
                        best_s, best_uid, best_key = s, xu, key
            for xu in extra:
                if xu == uid:
                    continue
                for src in nom_by_uid.get(xu, ()):
                    s = base[id(src)]
                    key = member_sort_key(getattr(src, "signals", None) or {})
                    if best_s is None or s > best_s or (
                        s == best_s and key < best_key
                    ):
                        best_s, best_uid, best_key = s, xu, key
            return best_s, best_uid

        boosted = 0
        injected: dict = {}
        new_cands: list = []
        if mode in (CTX_MODE_PROPAGATE, CTX_MODE_COMBINED):
            for c in cands:
                if not part[id(c)]:
                    new_cands.append(c)
                    continue
                v = views[id(c)]
                session = v[0] if v is not None else None
                bs, bx = _best_neighbor(str(c.unit_id), session)
                if bx is None or bs is None:
                    new_cands.append(c)
                    continue
                sig = dict(getattr(c, "signals", None) or {})
                sig["pre_ctx_score"] = base[id(c)]
                sig["ctx_from"] = bx
                new_cands.append(dc_replace(
                    c, raw_score=base[id(c)] + w * bs, signals=sig))
                boosted += 1
        else:
            new_cands = cands

        if mode in (CTX_MODE_INJECT, CTX_MODE_COMBINED) and M_ctx > 0:
            for c in cands[:M_ctx]:
                if not part[id(c)]:
                    continue
                c_uid = str(c.unit_id)
                v = views[id(c)]
                session = v[0] if v is not None else None
                for rv in _member_rows(c_uid, session):
                    x_uid = rv["unit_id"]
                    if x_uid == c_uid or x_uid in nom_ids or x_uid in injected:
                        continue
                    if not _eligible_member(eligible, x_uid):
                        continue
                    # Nominated sources reaching x: the position window
                    # is symmetric, so x's own member rows cover every
                    # nominated neighbor; ``extra`` keeps adjacency-mode
                    # edges that only name x from c's side.
                    bs, bx = _best_neighbor(
                        x_uid, rv[_SESSION_KEY], extra=(c_uid,))
                    if bx is None or bs is None:
                        continue
                    injected[x_uid] = (w * bs, bx, rv)
            cap = CTX_INJECT_CAP_FACTOR * M_ctx
            if len(injected) > cap:
                keep = sorted(
                    injected.items(),
                    key=lambda kv: (
                        -kv[1][0], kv[1][2][_SESSION_KEY], kv[1][2]["key"],
                        kv[0]),
                )[:cap]
                injected = dict(keep)
            for x_uid, (score, bx, rv) in sorted(
                injected.items(),
                key=lambda kv: (kv[1][2][_SESSION_KEY], kv[1][2]["key"],
                                kv[0]),
            ):
                new_cands.append(CandidateV7(
                    unit_id=x_uid,
                    source_id=rv["source_id"],
                    revision=rv["revision"],
                    lane=lane_name,
                    rank=0,
                    raw_score=score,
                    signals={
                        "ctx_from": bx,
                        "pre_ctx_score": 0.0,
                        "ctx_injected": True,
                        "kind": _KIND_TURN,
                        _SESSION_KEY: rv[_SESSION_KEY],
                        _SEQ_KEY: rv[_SEQ_KEY],
                    },
                ))

        # Re-rank by propagated score + deterministic identity fields and
        # re-stamp 1-based ranks — fusion consumes these ranks (§21.1).
        order = sorted(range(len(new_cands)), key=lambda i: _ctx_sort_key(new_cands[i]))
        reranked = [dc_replace(new_cands[i], rank=pos + 1)
                    for pos, i in enumerate(order)]
        lstat = {"boosted": boosted, "injected": len(injected)}
        stats["lanes"][lane_name] = lstat
        stats["boosted"] += boosted
        stats["injected"] += len(injected)
        st = dict(lane_stats)
        st["context_applied"] = True
        st["context"] = lstat
        new_outputs.append(LaneOutput(
            lane=out.lane,
            status=out.status,
            candidates=reranked,
            reason=out.reason,
            examined=out.examined,
            eligible=out.eligible,
            stats=st,
        ))
    return new_outputs, stats


# ---------------------------------------------------------------------------
# V8 reserved facet slots — SPEC_V8 §10.02, §21.7 (normative)
# ---------------------------------------------------------------------------


def _facet_key_default(item: Any) -> tuple:
    """Default ``(is_whole_query, facet_ids)`` view for a candidate.

    Lane-merge candidates carry ``signals["facet"] = <int>`` (facet arm)
    or no key (whole query); fused candidates may carry merged ``facet``
    (scalar) or ``facets`` (collection)."""
    sig = getattr(item, "signals", None) or {}
    if not isinstance(sig, Mapping):
        sig = {}
    facets = sig.get("facets")
    if isinstance(facets, (list, tuple, set, frozenset)):
        fset = {f for f in facets}
    elif facets is not None:
        fset = {facets}
    else:
        fset = set()
    single = sig.get("facet")
    if single is not None and not isinstance(single, (list, tuple, set, frozenset)):
        fset.add(single)
    is_whole = single is None and not fset or bool(sig.get("whole"))
    return is_whole, fset


def _dedupe_key_default(item: Any) -> Any:
    """Dedupe identity for slot accounting: ``unit_id`` when present —
    the same unit surfacing in the whole run and a facet run is ONE
    unit, not two (§21.7 "not already kept")."""
    uid = getattr(item, "unit_id", None)
    return ("u", str(uid)) if uid is not None else ("o", id(item))


def reserve_facet_slots(
    items: Iterable,
    capv: int,
    *,
    whole_share: float = DEFAULT_FACET_WHOLE_SHARE,
    n_facets: Optional[int] = None,
    key_of: Any = None,
    uid_of: Any = None,
) -> tuple:
    """SPEC_V8 §10.02/§21.7 — reserved facet slots in a ``capv`` pool.

    ``share_q = floor(capv * whole_share)`` goes to the whole-query run;
    each facet gets ``share_f = floor((capv - share_q) / n_facets)``
    reserved slots filled from its own run (skipping units already
    kept). Remaining capacity is filled by a deterministic round-robin
    merge of the leftover lists, whole query first — a facet that
    under-produces rolls its slots over rather than starving the pool.

    ``key_of(item) -> (is_whole, facet_ids)`` adapts the caller's item
    shape; the default reads ``signals["facet"]``/``signals["facets"]``.
    ``uid_of(item)`` supplies the dedupe identity (default: ``unit_id``
    else object identity). ``n_facets`` should be the decomposition's
    facet count — a facet that produced zero rows still divides the
    reserved share (pass it explicitly at the lane-merge call site).

    Returns ``(kept_items, stats)`` where stats reports per-facet
    ``produced``/``kept``/``rolled_over`` plus ``whole`` for
    ``coverage.facets`` (§20).
    """
    items = list(items or ())
    if isinstance(capv, bool) or not isinstance(capv, int) or capv < 0:
        raise ValueError(f"capv must be a non-negative int — got {capv!r}")
    try:
        whole_share = float(whole_share)
    except (TypeError, ValueError):
        raise ValueError(f"facets.whole_share must be numeric — got {whole_share!r}")
    if not (0.0 < whole_share <= 1.0):
        raise ValueError(
            f"facets.whole_share must lie in (0, 1] — got {whole_share!r}")
    if key_of is None:
        key_of = _facet_key_default
    if uid_of is None:
        uid_of = _dedupe_key_default

    whole_list: list = []
    facet_lists: dict = {}
    facet_ids: list = []
    for it in items:
        is_whole, fset = key_of(it)
        if is_whole or not fset:
            whole_list.append(it)
        for f in sorted(fset, key=repr):
            if f not in facet_lists:
                facet_lists[f] = []
                facet_ids.append(f)
            facet_lists[f].append(it)
    facet_ids.sort(key=repr)
    if n_facets is None:
        n_facets = len(facet_ids)
    if isinstance(n_facets, bool) or not isinstance(n_facets, int) or n_facets < 0:
        raise ValueError(f"n_facets must be a non-negative int — got {n_facets!r}")

    share_q = math.floor(capv * whole_share)
    share_f = math.floor((capv - share_q) / n_facets) if n_facets else 0

    kept: list = []
    kept_keys: set = set()
    for it in whole_list[:share_q]:
        kept.append(it)
        kept_keys.add(uid_of(it))
    whole_kept = len(kept)
    whole_rolled = 0
    per_facet: dict = {}
    for f in facet_ids:
        produced = len(facet_lists[f])
        take = 0
        for it in facet_lists[f]:
            if take >= share_f:
                break
            if uid_of(it) in kept_keys:
                continue
            kept_keys.add(uid_of(it))
            kept.append(it)
            take += 1
        per_facet[f] = {"produced": produced, "kept": take, "rolled_over": 0}

    leftover = capv - len(kept)
    if leftover > 0:
        # Deterministic round-robin over the remaining lists, whole
        # query first (§21.7 rollover).
        lists = [[it for it in whole_list[share_q:] if uid_of(it) not in kept_keys]]
        lists += [[it for it in facet_lists[f] if uid_of(it) not in kept_keys]
                  for f in facet_ids]
        ptr = [0] * len(lists)
        while leftover > 0:
            progressed = False
            for li, lst in enumerate(lists):
                while ptr[li] < len(lst):
                    it = lst[ptr[li]]
                    ptr[li] += 1
                    if uid_of(it) in kept_keys:
                        continue
                    kept_keys.add(uid_of(it))
                    kept.append(it)
                    leftover -= 1
                    if li > 0:
                        f = facet_ids[li - 1]
                        per_facet[f]["kept"] += 1
                        per_facet[f]["rolled_over"] += 1
                    else:
                        whole_rolled += 1
                        whole_kept += 1
                    progressed = True
                    break
                if leftover <= 0:
                    break
            if not progressed:
                break

    stats = {
        "capv": capv,
        "whole_share": whole_share,
        "share_q": share_q,
        "share_f": share_f,
        "whole": {
            "produced": len(whole_list),
            "kept": whole_kept,
            "rolled_over": whole_rolled,
        },
        "per_facet": {str(f): per_facet[f] for f in facet_ids},
        "kept_total": len(kept),
        "overflow": max(0, len(kept) - capv),
    }
    return kept, stats


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def rrf_fuse(
    outputs: Iterable[LaneOutput],
    weights: Optional[Mapping] = None,
    k: int = RRF_K,
    *,
    limit: Optional[int] = None,
    lane_gates: Optional[Mapping] = None,
    weights_tag: Optional[str] = None,
    context: Any = None,
    dense_slots: Optional[int] = None,
    facet_whole_share: Any = None,
    lex_anchor: Any = None,
    dense_form: Any = None,
    dense_form_alpha: Any = None,
    policy: Any = None,
) -> FusedList:
    """Fuse lane outputs into one ranked pool via weighted RRF (§32.3).

    ``outputs`` — per-lane ``LaneOutput`` objects; ``weights`` — the
    intent-resolved lane weight table (``LaneName`` or ``str`` keys, missing
    lanes default to ``DEFAULT_LANE_WEIGHT``); ``k`` — the RRF constant;
    ``limit`` — optional rerank-pool cut (V7-10.04), reported as
    ``stats["truncated"]``. ``lane_gates`` — optional V75-04.03 weak-lane
    containment map (``{lane: "exclude" | top_n int}``); ``weights_tag`` —
    optional arm tag echoed into ``stats["weights_tag"]`` so consumers can
    attribute which weight table produced the order (V75-03.03).

    V8 arms (SPEC_V8 §23; all keyword-only, all off/defaulted):

    - ``context`` — §06/§21.1 context stage. ``None``/``False`` leaves
      lane outputs untouched (the stage is expected to run upstream via
      :func:`propagate_context`); a mode string, ``True``, or a mapping
      ``{"mode","w","W","M_ctx","neighbors","eligible"}`` runs it here.
      ``stats["context"]`` records mode/w/W/injected/boosted for
      ``coverage.context`` either way.
    - ``dense_slots`` — V8-08.04 ``dense.N_d`` (default 10): dense-lane
      candidates ranked ≤ N_d (or lane-marked ``dense_slot``) are
      guaranteed fused-pool slots against the ``limit`` cut and carry
      ``signals["dense_slot"]``. ``0`` disables.
    - ``facet_whole_share`` — V8-10.02 ``facets.whole_share`` (default
      0.5; ``False``/1.0 disables): when ``limit`` truncates a pool that
      holds facet-tagged units, :func:`reserve_facet_slots` selects the
      survivors so facet runs keep their reserved slots.
    - ``lex_anchor`` — V8-11.03 ``fusion.lex_anchor`` (default off):
      ``True``/a divisor adds the rank-space anchor
      ``1 / (k + rank/divisor)`` for lex ranks ≤ 3 — never a raw-score
      scale (§11.03).
    - ``dense_form`` — V8-08.06 ``fusion.dense_form`` (default
      ``"rrf"``): ``"combsum"`` switches the lexical/dense pair to
      score-space — lex contributes lane-max-normalized BM25
      (``bm25_norm``), dense contributes ``α·cos`` with
      ``dense_form_alpha`` (prior 1.2, or the
      ``fusion.dense_form_alpha`` arm); every other lane keeps its
      ``w/(k+rank)`` rank term — lane weights never become raw-score
      multipliers (V75-03.03). ``policy`` is the optional §23 params
      carrier (a ``RetrievalPolicyV7``/``GatedPolicyV7`` or anything
      exposing ``params``/``knobs``/``arms`` maps or the underscored
      attribute); explicit kwargs outrank it. ``stats["dense_form"]``
      records the resolved form and what it applied.

    Returns a :class:`FusedList` ordered by ``(-rrf, source_id, unit_id)``.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"rrf k must be a positive int, got {k!r}")
    w = _norm_weights(weights)
    gates = _norm_gates(lane_gates)
    outputs = list(outputs or ())  # materialize — scanned twice (fuse + V7-10.03)

    # ---- V8-06.01 context stage (§21.1): per lane, before lane ranks ----
    ctx_stats: Optional[dict] = None
    if context:
        cpar = _resolve_ctx_arg(context)
        outputs, ctx_stats = propagate_context(
            outputs,
            neighbors=cpar.get("neighbors"),
            mode=cpar["mode"],
            w=cpar["w"],
            W=cpar["W"],
            M_ctx=cpar["M_ctx"],
            eligible=cpar.get("eligible"),
        )
    elif any((getattr(o, "stats", None) or {}).get("context_applied")
             for o in outputs):
        # Stage already ran upstream — surface its coverage block.
        lanes = {
            _lane_key(getattr(o, "lane", "")): (o.stats or {}).get("context")
            for o in outputs if (o.stats or {}).get("context_applied")
        }
        ctx_stats = {
            "model": CONTEXT_MODEL_ID,
            "mode": "applied_upstream",
            "injected": sum((s or {}).get("injected", 0) for s in lanes.values()),
            "boosted": sum((s or {}).get("boosted", 0) for s in lanes.values()),
            "lanes": lanes,
        }

    # ---- V8 arm normalization -----------------------------------------
    if dense_slots is None:
        n_d = DEFAULT_DENSE_SLOTS
    else:
        if isinstance(dense_slots, bool) or not isinstance(dense_slots, int) \
                or dense_slots < 0:
            raise ValueError(f"dense_slots must be a non-negative int — got {dense_slots!r}")
        n_d = dense_slots
    if facet_whole_share in (None, True):
        whole_share = DEFAULT_FACET_WHOLE_SHARE
    elif facet_whole_share is False:
        whole_share = 1.0  # reservation disabled — whole run takes the cap
    else:
        whole_share = float(facet_whole_share)
        if not (0.0 < whole_share <= 1.0):
            raise ValueError(
                f"facet_whole_share must lie in (0, 1] — got {facet_whole_share!r}")
    lex_div: Optional[float] = None
    if lex_anchor not in (None, False):
        lex_div = DEFAULT_LEX_ANCHOR_DIVISOR if lex_anchor is True else float(lex_anchor)
        if not math.isfinite(lex_div) or lex_div <= 0:
            raise ValueError(f"lex_anchor divisor must be > 0 — got {lex_anchor!r}")
    dense_form, dense_form_alpha = _resolve_dense_form(
        dense_form, dense_form_alpha, policy
    )

    # J13 fallback: a gated lane still answers standalone. When every lane
    # that produced candidates is gated, the gates are bypassed — the
    # declared map is still printed in stats for attribution.
    contributing = {
        _lane_key(getattr(out, "lane", ""))
        for out in outputs
        if _status_key(getattr(out, "status", "")) in _FUSABLE_STATUSES
        and getattr(out, "candidates", None)
    }
    gates_bypassed = bool(gates) and bool(contributing) and contributing <= set(gates)
    eff_gates = {} if gates_bypassed else gates

    stats: dict = {
        "model": FUSION_MODEL_ID,
        "formula_status": FORMULA_STATUS_PROVISIONAL,
        "k": k,
        "weights": dict(w),
        "weights_tag": weights_tag,
        "default_weight": DEFAULT_LANE_WEIGHT,
        "lane_gates": dict(gates),
        "gates_bypassed": gates_bypassed,
        "lanes_contributing": [],
        "lanes_ignored": {},
        "lanes_gated": {},
        "candidate_rows": 0,
        "duplicates_in_lane": 0,
        "units": 0,
        "truncated": False,
        "limit": limit,
        "facet_bonus": {
            "tag": FACET_BONUS_ID,
            "beta": FACET_BONUS_BETA,
            "cap": FACET_BONUS_MAX,
            "applied": 0,
        },
        "dense_slots": {"n_d": n_d, "marked": 0, "kept": 0, "displaced": 0},
        # V8-08.06 — the resolved form; ``applied`` counts candidates
        # that received a score-space term, ``lex_max`` the CombSUM
        # normalizer (``None`` under rrf).
        "dense_form": {
            "form": dense_form,
            "alpha": dense_form_alpha,
            "applied": 0,
            "lex_max": None,
        },
    }
    if ctx_stats is not None:
        stats["context"] = ctx_stats
    if lex_div is not None:
        stats["lex_anchor"] = {"divisor": lex_div, "applied": 0}

    # unit_id -> aggregate
    agg: dict = {}
    order: list = []
    for out in outputs or []:
        lane = _lane_key(getattr(out, "lane", ""))
        status = _status_key(getattr(out, "status", ""))
        cands = list(getattr(out, "candidates", None) or [])
        if status not in _FUSABLE_STATUSES:
            if cands or status:
                stats["lanes_ignored"][lane] = status
            continue
        stats["lanes_contributing"].append(lane)
        gate = eff_gates.get(lane)
        if gate is not None:
            stats["lanes_gated"][lane] = {"gate": gate, "dropped": 0}
        seen_in_lane: set = set()
        for pos, cand in enumerate(cands):
            stats["candidate_rows"] += 1
            uid = str(getattr(cand, "unit_id"))
            rank = _rank_of(cand, pos)
            if gate is not None and rank > gate:
                # V75-04.03: beyond the lane's top-N (or excluded at 0) —
                # gated rows never enter the fused pool at all.
                stats["lanes_gated"][lane]["dropped"] += 1
                continue
            sig = getattr(cand, "signals", None)
            entry = agg.get(uid)
            if entry is None:
                entry = {
                    "unit_id": uid,
                    "source_id": str(getattr(cand, "source_id", "")),
                    "revision": int(getattr(cand, "revision", 0) or 0),
                    "lane_ranks": {},
                    "signals": {},
                    "lane_raw": {},
                    "facets": set(),
                    "whole": False,
                    "dense_marked": False,
                }
                agg[uid] = entry
                order.append(uid)
            if isinstance(sig, Mapping) and sig.get("facet") is not None:
                # V75-03.04 facet coverage: which decomposition facets
                # surfaced this unit (evidence, counted on every accepted
                # row — including a within-lane duplicate whose rank loses).
                fv = sig["facet"]
                try:
                    entry["facets"].add(fv)
                except TypeError:
                    entry["facets"].add(repr(fv))
            else:
                # V8-10.02: an untagged accepted row is a whole-query
                # surfacing for reserved-facet-slot accounting.
                entry["whole"] = True
            if isinstance(sig, Mapping) and sig.get("dense_slot"):
                entry["dense_marked"] = True
            if uid in seen_in_lane or lane in entry["lane_ranks"]:
                # Duplicate within one lane: keep the better (smaller) rank.
                stats["duplicates_in_lane"] += 1
                if rank >= entry["lane_ranks"].get(lane, rank):
                    continue
            seen_in_lane.add(uid)
            entry["lane_ranks"][lane] = rank
            entry["lane_raw"][lane] = getattr(cand, "raw_score", None)
            _merge_signals(entry["signals"], lane, sig or {})

    # V8-08.06 CombSUM — lane-max normalizer for the lex BM25 term,
    # computed over the *accepted* lane rows (gated/dropped rows never
    # enter the denominator — the normalizer describes what fused).
    lex_max = 0.0
    if dense_form == "combsum":
        lex_raws = [
            float(v)
            for e in agg.values()
            for v in [e["lane_raw"].get(LEX_LANE)]
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(float(v))
        ]
        lex_max = max(lex_raws, default=0.0)
        stats["dense_form"]["lex_max"] = lex_max

    fused: list = []
    for uid in order:
        entry = agg[uid]
        lane_ranks = entry["lane_ranks"]
        lane_raw = entry["lane_raw"]
        parts: list = []
        combsum_terms: dict = {}
        for lane, rank in lane_ranks.items():
            if dense_form == "combsum" and lane == LEX_LANE:
                # Score-space term (V8-08.06): bm25_norm — the lane
                # weight stays rank-space and does NOT multiply this.
                raw = lane_raw.get(LEX_LANE)
                bm25_norm = (
                    float(raw) / lex_max
                    if lex_max > 0.0
                    and isinstance(raw, (int, float))
                    and not isinstance(raw, bool)
                    and math.isfinite(float(raw))
                    else 0.0
                )
                parts.append(bm25_norm)
                combsum_terms["bm25_norm"] = bm25_norm
                stats["dense_form"]["applied"] += 1
            elif dense_form == "combsum" and lane == DENSE_LANE:
                # Score-space term: α·cos.
                raw = lane_raw.get(DENSE_LANE)
                cos = (
                    float(raw)
                    if isinstance(raw, (int, float))
                    and not isinstance(raw, bool)
                    and math.isfinite(float(raw))
                    else 0.0
                )
                parts.append(dense_form_alpha * cos)
                combsum_terms["dense_cos"] = cos
                stats["dense_form"]["applied"] += 1
            else:
                # Rank-space term — the lane weight applies here only.
                parts.append(
                    w.get(lane, DEFAULT_LANE_WEIGHT) / (k + rank)
                )
        signals = _freeze_merged_signals(entry["signals"])
        if combsum_terms:
            signals["dense_form"] = "combsum"
            signals["dense_form_alpha"] = dense_form_alpha
            signals["dense_form_lex_max"] = lex_max
            signals.update(combsum_terms)
        lanes_view = signals.setdefault(LANES_SIGNALS_KEY, {})
        for lane, rank in lane_ranks.items():
            lane_entry = lanes_view.setdefault(lane, {})
            lane_entry["rank"] = rank
            lane_entry["raw_score"] = entry["lane_raw"].get(lane)
        # Facet coverage bonus (facet_bonus/v1): >= 2 distinct facets must
        # have surfaced the unit; the term is a bounded rank-space
        # contribution at the unit's best rank.
        covered = len(entry["facets"])
        if covered >= 2:
            bw = min(FACET_BONUS_BETA * (covered - 1), FACET_BONUS_MAX)
            bonus = bw / (k + min(lane_ranks.values()))
            parts.append(bonus)
            stats["facet_bonus"]["applied"] += 1
            signals["facet_coverage"] = tuple(
                sorted(entry["facets"], key=repr)
            )
            signals["facet_bonus"] = bonus
        # V8-11.03 lexical anchor — rank-space only (never a raw-score
        # scale): a lex rank <= 3 adds 1/(k + rank/divisor).
        if lex_div is not None:
            lr = lane_ranks.get(LEX_LANE)
            if isinstance(lr, int) and 1 <= lr <= 3:
                bonus = 1.0 / (k + lr / lex_div)
                parts.append(bonus)
                signals["lex_anchor"] = bonus
                signals["lex_anchor_divisor"] = lex_div
                stats["lex_anchor"]["applied"] += 1
        # V8-08.04 dense-slot marking — the lane's own top-N_d mark is
        # honored; an unmarked dense lane derives it from its rank.
        if n_d and (
            entry["dense_marked"]
            or (isinstance(lane_ranks.get(DENSE_LANE), int)
                and lane_ranks[DENSE_LANE] <= n_d)
        ):
            signals["dense_slot"] = True
            stats["dense_slots"]["marked"] += 1
        fused.append(FusedCandidate(
            unit_id=uid,
            source_id=entry["source_id"],
            revision=entry["revision"],
            rrf=math.fsum(parts),
            lane_ranks=dict(sorted(lane_ranks.items())),
            signals=signals,
        ))

    fused.sort(key=lambda c: (-c.rrf, c.source_id, c.unit_id))

    if limit is not None and len(fused) > max(int(limit), 0):
        capv = max(int(limit), 0)
        stats["truncated"] = True
        pool = fused  # full pre-truncation pool (dense guarantee scans it)
        facet_ids: set = set()
        for c in pool:
            facet_ids.update(agg.get(c.unit_id, {}).get("facets") or ())
        if facet_ids and whole_share < 1.0:
            # V8-10.02/§21.7 — reserved facet slots inside the fused cap.
            kept, fstats = reserve_facet_slots(
                pool, capv,
                whole_share=whole_share,
                key_of=lambda c: (
                    agg.get(c.unit_id, {}).get("whole", False),
                    set(agg.get(c.unit_id, {}).get("facets") or ()),
                ),
            )
            keep_ids = {c.unit_id for c in kept}
            fused = [c for c in pool if c.unit_id in keep_ids]
            stats["facet_slots"] = fstats
        else:
            fused = pool[:capv]
        # V8-08.04 — guaranteed dense slots: a marked unit outside the
        # kept pool displaces the lowest-ranked unmarked keeper (never a
        # marked one, and the delivered pin is untouched — the cut only
        # re-selects pool membership, it never rewrites a candidate).
        if n_d:
            keep_ids = {c.unit_id for c in fused}
            for cand in pool:
                if not cand.signals.get("dense_slot") or cand.unit_id in keep_ids:
                    continue
                for j in range(len(fused) - 1, -1, -1):
                    if not fused[j].signals.get("dense_slot"):
                        keep_ids.discard(fused[j].unit_id)
                        keep_ids.add(cand.unit_id)
                        fused[j] = cand
                        stats["dense_slots"]["displaced"] += 1
                        break
                else:
                    stats["dense_slots"]["unplaced"] = (
                        stats["dense_slots"].get("unplaced", 0) + 1)
            fused.sort(key=lambda c: (-c.rrf, c.source_id, c.unit_id))
        stats["dense_slots"]["kept"] = sum(
            1 for c in fused if c.signals.get("dense_slot"))

    stats["units"] = len(fused)
    stats["constant_signals"] = sorted(detect_constant_signals(outputs))
    return FusedList(fused, stats=stats)


def detect_constant_signals(
    outputs: Iterable[LaneOutput],
    threshold: float = CONSTANT_SIGNAL_THRESHOLD,
) -> set:
    """Signal keys constant on ≥ ``threshold`` of candidate rows (V7-10.03).

    "Present with the same value on ≥ 90% of candidates": for each signal
    key, the count of its modal value over *all* candidate rows from fusable
    (``ok``/``partial``) lanes must reach ``threshold`` of the row total.
    Values are compared on a deterministic canonical form, so ``1`` and
    ``1.0`` are the same value while ``nan`` never equals itself.
    """
    if not (0.0 < float(threshold) <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    total = 0
    counts: dict = {}
    vkey_of = _value_key
    isfinite = math.isfinite
    for out in outputs or []:
        if _status_key(getattr(out, "status", "")) not in _FUSABLE_STATUSES:
            continue
        for cand in getattr(out, "candidates", None) or []:
            total += 1
            sig = getattr(cand, "signals", None)
            if not isinstance(sig, Mapping):
                continue
            for key, value in sig.items():
                bucket = counts.setdefault(str(key), {})
                # Scalar fast path mirrors ``_value_key`` exactly —
                # ``_canonicalize`` passes these through unchanged.
                tv = type(value)
                if tv is str or value is None or tv is int or tv is bool:
                    vkey = ("v", value)
                elif tv is float:
                    vkey = ("v", value if isfinite(value) else repr(value))
                else:
                    vkey = vkey_of(value)
                try:
                    bucket[vkey] += 1
                except KeyError:
                    bucket[vkey] = 1
    if total == 0:
        return set()
    constant = set()
    for key, bucket in counts.items():
        if max(bucket.values()) / total >= float(threshold):
            constant.add(key)
    return constant


def score_detail(
    cand: FusedCandidate,
    weights: Optional[Mapping] = None,
    k: int = RRF_K,
) -> dict:
    """Explain helper (V7-10.12, V7-05.15): per-lane RRF contributions.

    Returns ``{"rrf", "k", "lane_ranks", "contributions", "weights"}`` where
    ``contributions[lane] = w_lane / (k + rank_lane)`` — every number that
    produced the fused score, byte-stably reconstructable. When a facet
    coverage bonus was applied (``signals["facet_bonus"]``), the term is
    recomputed under ``"facet_bonus"`` so ``Σ contributions + bonus ==
    rrf`` holds exactly.
    """
    w = _norm_weights(weights)
    sig = cand.signals or {}
    # V8-08.06 — under ``combsum`` the lex/dense terms are score-space;
    # reconstruct them from the recorded signals so
    # ``Σ contributions + bonuses == rrf`` stays exact.
    combsum = sig.get("dense_form") == "combsum"
    alpha = sig.get("dense_form_alpha")
    contributions = {}
    for lane, rank in cand.lane_ranks.items():
        if combsum and lane == LEX_LANE:
            contributions[lane] = float(sig.get("bm25_norm") or 0.0)
        elif combsum and lane == DENSE_LANE:
            contributions[lane] = float(alpha or 0.0) * float(
                sig.get("dense_cos") or 0.0
            )
        else:
            contributions[lane] = (
                w.get(lane, DEFAULT_LANE_WEIGHT) / (k + rank)
            )
    detail = {
        "rrf": cand.rrf,
        "k": k,
        "lane_ranks": dict(cand.lane_ranks),
        "contributions": contributions,
        "weights": {lane: w.get(lane, DEFAULT_LANE_WEIGHT) for lane in cand.lane_ranks},
        "model": FUSION_MODEL_ID,
        "formula_status": FORMULA_STATUS_PROVISIONAL,
    }
    coverage = (cand.signals or {}).get("facet_coverage")
    if coverage and len(coverage) >= 2 and cand.lane_ranks:
        bw = min(FACET_BONUS_BETA * (len(coverage) - 1), FACET_BONUS_MAX)
        detail["facet_bonus"] = {
            "tag": FACET_BONUS_ID,
            "coverage": list(coverage),
            "weight": bw,
            "value": bw / (k + min(cand.lane_ranks.values())),
        }
    sig = cand.signals or {}
    # V8-11.03 — recomputed so Σ contributions + bonuses == rrf.
    if sig.get("lex_anchor") is not None:
        lr = cand.lane_ranks.get(LEX_LANE)
        div = sig.get("lex_anchor_divisor") or DEFAULT_LEX_ANCHOR_DIVISOR
        if isinstance(lr, int):
            detail["lex_anchor"] = {
                "rank": lr,
                "divisor": div,
                "value": 1.0 / (k + lr / div),
            }
    # V8-20.04 — context provenance (§06): injected units report their
    # pre-context score (0.0) and the nominated neighbor that sourced
    # them; boosted units report the same fields for their argmax.
    if sig.get("ctx_from") is not None or sig.get("pre_ctx_score") is not None:
        detail["context"] = {
            "ctx_from": sig.get("ctx_from"),
            "pre_ctx_score": sig.get("pre_ctx_score"),
            "injected": bool(sig.get("ctx_injected")),
        }
    # V8-08.06 — the CombSUM arm's resolved surface: which lanes moved
    # to score-space, the normalizer, and α.
    if combsum:
        detail["dense_form"] = {
            "form": "combsum",
            "alpha": sig.get("dense_form_alpha"),
            "lex_max": sig.get("dense_form_lex_max"),
            "bm25_norm": sig.get("bm25_norm"),
            "dense_cos": sig.get("dense_cos"),
        }
    return detail


#: Read-only view of the merge policy, for docs/coverage reporting.
SIGNAL_MERGE_POLICY = MappingProxyType({
    "unit_fact": sorted(_UNIT_FACT_KEYS),
    "set_union": sorted(_SET_MERGE_KEYS),
    "dict_max": sorted(_DICT_MERGE_KEYS),
    "scalar_max": sorted(_MAX_MERGE_KEYS),
    "explain_passthrough": sorted(_EXPLAIN_KEYS),
    "lanes_key": LANES_SIGNALS_KEY,
})


__all__ = [
    "CONSTANT_SIGNAL_THRESHOLD",
    "CONTEXT_MODEL_ID",
    "CTX_INJECT_CAP_FACTOR",
    "CTX_MODE_COMBINED",
    "CTX_MODE_FIELD",
    "CTX_MODE_INJECT",
    "CTX_MODE_OFF",
    "CTX_MODE_PROPAGATE",
    "DEFAULT_CONTEXT_W",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_DENSE_FORM",
    "DEFAULT_DENSE_FORM_ALPHA",
    "DEFAULT_DENSE_SLOTS",
    "DEFAULT_FACET_WHOLE_SHARE",
    "DEFAULT_LANE_WEIGHT",
    "DEFAULT_LEX_ANCHOR_DIVISOR",
    "DEFAULT_M_CTX",
    "DENSE_LANE",
    "FACET_BONUS_BETA",
    "FACET_BONUS_ID",
    "FACET_BONUS_MAX",
    "FUSION_MODEL_ID",
    "FusedList",
    "LANES_SIGNALS_KEY",
    "LEX_LANE",
    "RRF_K",
    "SIGNAL_MERGE_POLICY",
    "detect_constant_signals",
    "propagate_context",
    "reserve_facet_slots",
    "rrf_fuse",
    "score_detail",
]
