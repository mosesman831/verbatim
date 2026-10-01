"""S4 — deterministic feature reranker (``rerank_features/v1``,
``provisional/v7-r0``).

Implements SPEC_V7 §32.4 / V7-10.02, V7-10.06, V7-10.07, V7-10.12.

- ``FEATURE_WEIGHTS_V1`` is the §32.4 default coefficient table verbatim.
- ``score_candidates(query, fused, feature_fn=None, ...)`` computes the
  feature vector per candidate, scores ``Σ w·feature`` with ``math.fsum``
  (order-independent rounding), and returns ``ScoredCandidate`` objects with
  ``score_family = "ranking/v7"``.
- ``feature_fn`` is the injection seam: ``(query, cand, FeatureContext) ->
  Mapping[str, float] | FeatureVector``. The default implementation
  (:func:`default_features`) computes what is computable from candidate
  signals + the query — ``rrf_norm`` from rrf, ``cov_idf`` from
  matched-term idf — and consults optional
  :class:`FeatureProviders` callables for corpus/context-dependent features.
  Lane-computed feature-shaped signals (``phrase``) pass through, clamped
  to the declared §32.4 ranges.
- V8.5 §05.04 slim set: the shipped model is ``{rrf_norm, cov_idf,
  ent_idf(×0), phrase, speaker_match, t_prox}``.  The seven measured dead
  features — ``cov_idf_ctx``, ``ident_exact``, ``life_state``, ``corrob``,
  ``perspective_fit``, ``event_pred``, ``lane_agree`` — are no longer
  computed by the default function: they are absent from the vector and
  from ``score_detail``, never a silent zero.
- ``speaker_match`` resolves the question's subject speaker once per
  ``score_candidates`` invocation (V75-03.05): a caller-supplied
  ``query.speaker_canon`` hint wins (source ``hint``); otherwise the
  ``query_speakers`` provider resolves which of the query's entity canons
  are speaker canons present in scope — exactly one → ``query``; two or
  more → ``ambiguous`` and the feature stays neutral 0.5, never a guess;
  zero/unresolvable → ``none``. When the pipeline hands a
  ``LaneContextV7``-shaped ``ctx``, the provider is synthesized against
  its pinned read snapshot (``units.speaker_canon`` at/below the fence —
  ``retrieval/v7/entity.resolve_query_speaker_canons``). The source label
  is reported on every candidate's ``detail["speaker_match_source"]`` and
  in ``ScoredList.stats`` so evaluations can tell which path fired. It is
  a bounded feature only — never an eligibility gate (V7-05.08/05.12).
- Missing features are absent from the vector, never invented; absent
  features contribute ``0`` to the score.
- **No ties (V7-10.02, fixture H13).** The D7-06 degenerate-normalization
  collapse is impossible here: members of an exactly-tied feature-score
  group receive a deterministic ``tie_epsilon`` — a BLAKE2b hash of the
  candidate's *canonical signal vector* scaled into ``[0, 1e-9)`` — so
  distinct signal vectors yield distinct final scores (up to a 2^-64 hash
  collision). Untied candidates keep ``score == Σ w·feature`` bit-for-bit
  and ``tie_epsilon == 0.0``. Identical signal vectors keep an honest tie
  which the declared tie-break resolves — ``rrf_norm`` desc, then
  ``(source_id asc, revision desc, unit seq asc)`` — so ordering is always
  total and byte-stable. ``detail["feature_score"]`` always holds the pure
  ``Σ w·feature`` value and ``detail["tie_epsilon"]`` the disambiguation.
- Determinism: no clock, no randomness — identical inputs produce
  identical scores and ordering. The one optional store touch is the
  ``query_speakers`` provider resolving the question's speaker canons,
  once per invocation over the caller's pinned read snapshot (never a new
  transaction); per snapshot it is deterministic.
- Cost: pure stdlib, O(R · features); ≤ 3 ms p95 at R = 100 (V7-10.07).

Signal key conventions consumed by :func:`default_features` (produced by
lanes and merged by ``fusion.rrf_fuse``): ``matched_terms``,
``matched_canons``, ``term_idf``, ``entity_idf``, ``phrase``,
``speaker_canon``, ``occurred_start_us`` / ``occurred_end_us`` /
``recorded_at_us``, ``seq``, and per-lane provenance under ``_lanes``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace as dc_replace
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any, Callable, Iterable, Optional

from ...core.types import ErrorCode, VerbatimError
from ...core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    FeatureVector,
    FusedCandidate,
    QueryViewV7,
    ScoredCandidate,
)

#: Model / artifact identifiers.
RERANK_MODEL_ID = "rerank_features/v1"
SCORE_FAMILY = "ranking/v7"

#: The shipped weight table — the V8.5 §05.04 slim set, measured on the
#: corpus (LOFO + b4 real pass, three independent measurements):
#: ``ent_idf`` keeps its feature but its coefficient is 0 (measured dead
#: weight — a forensics ``feature_weights`` arm can still re-arm it), and
#: the seven dead features ``cov_idf_ctx`` / ``ident_exact`` /
#: ``life_state`` / ``corrob`` / ``perspective_fit`` / ``event_pred`` /
#: ``lane_agree`` are no longer computed at all — absent from the vector
#: and from ``score_detail``, never a silent zero.
FEATURE_WEIGHTS_V1: Mapping[str, float] = MappingProxyType({
    "rrf_norm": 1.0,
    "cov_idf": 0.9,
    "ent_idf": 0.0,
    "phrase": 0.3,
    "speaker_match": 0.4,
    "t_prox": 0.4,
})

#: Declared §32.4 ranges — computed and injected values are clamped to
#: them.  Ranges for features the shipped default no longer computes are
#: retained so a ``feature_fn`` injection arm re-adding one is still
#: clamped to its declared range.
FEATURE_RANGES: Mapping[str, tuple] = MappingProxyType({
    "cov_idf": (0.0, 1.0),
    "cov_idf_ctx": (0.0, 1.0),
    "phrase": (0.0, 1.0),
    "ent_idf": (0.0, 1.0),
    "speaker_match": (0.0, 1.0),
    "ident_exact": (0.0, 1.0),
    "t_prox": (0.0, 1.0),
    "event_pred": (0.0, 1.0),
    "life_state": (0.0, 1.0),
    "corrob": (0.0, 1.0),
    "lane_agree": (0.0, 1.0),
    "rrf_norm": (0.0, 1.0),
    "perspective_fit": (0.0, 1.0),
})

#: Tie-disambiguation epsilon bound (V7-10.02). Well below any meaningful
#: feature-score gap; disclosed as ``detail["tie_epsilon"]``.
TIE_EPSILON = 1e-9

#: §23 ``rerank.speaker_match_weight`` (V8-11.02) — the declared arm for
#: the ``speaker_match`` coefficient. The prior is the §32.4 table value;
#: the V8-11.02 variants ``{current, half, zero}`` map onto the weight
#: scale. (``rank-space bonus only`` is a different mechanism — it does
#: not reduce to a feature weight and is refused loudly.)
SPEAKER_MATCH_WEIGHT_DEFAULT = float(FEATURE_WEIGHTS_V1["speaker_match"])
_SPEAKER_MATCH_FORMS = {
    "current": SPEAKER_MATCH_WEIGHT_DEFAULT,
    "half": SPEAKER_MATCH_WEIGHT_DEFAULT / 2.0,
    "zero": 0.0,
    "off": 0.0,
    "none": 0.0,
}


def _speaker_match_weight_arm(ctx: Any) -> Optional[float]:
    """Resolve ``rerank.speaker_match_weight`` off ``ctx.policy``.

    Carrier order mirrors the lane convention: a ``speaker_match_weight``
    /``rerank_speaker_match_weight`` attribute on the policy object, then
    a ``params``/``knobs``/``arms`` map keyed by the dotted or
    underscored flag name, then ``ctx.manifest``. Accepted values: a
    finite number ≥ 0 (``0`` disables the feature's contribution —
    V8-11.02's ``zero`` arm), or the named variants ``current`` /
    ``half`` / ``zero`` / ``off``. Anything else is a VALIDATION
    failure — a declared-but-mistyped arm never silently reverts to the
    table default.
    """
    if ctx is None:
        return None
    pol = getattr(ctx, "policy", None)
    manifest = getattr(ctx, "manifest", None) or {}
    raw = None
    for attr in ("rerank_speaker_match_weight", "speaker_match_weight"):
        val = getattr(pol, attr, None)
        if val is not None:
            raw = val
            break
    if raw is None:
        for carrier in ("params", "knobs", "arms"):
            mapping = getattr(pol, carrier, None)
            if isinstance(mapping, Mapping):
                for key in (
                    "rerank.speaker_match_weight",
                    "rerank_speaker_match_weight",
                    "speaker_match_weight",
                ):
                    val = mapping.get(key)
                    if val is not None:
                        raw = val
                        break
            if raw is not None:
                break
    if raw is None and isinstance(manifest, Mapping):
        for key in (
            "rerank.speaker_match_weight",
            "rerank_speaker_match_weight",
            "speaker_match_weight",
        ):
            val = manifest.get(key)
            if val is not None:
                raw = val
                break
    if raw is None:
        return None
    if isinstance(raw, str):
        val = _SPEAKER_MATCH_FORMS.get(raw.strip().lower())
        if val is None:
            raise VerbatimError(
                ErrorCode.VALIDATION,
                "rerank.speaker_match_weight must be a number >= 0 or one "
                f"of {sorted(_SPEAKER_MATCH_FORMS)}, got {raw!r}",
            )
        return float(val)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"rerank.speaker_match_weight must be a number >= 0, "
            f"got {raw!r}",
        )
    out = float(raw)
    if not math.isfinite(out) or out < 0.0:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            f"rerank.speaker_match_weight must be finite and >= 0, "
            f"got {raw!r}",
        )
    return out

_US_PER_DAY = 86_400_000_000.0

#: Signal keys checked by the default feature function, in precedence order.
_KEY_MATCHED_TERMS = "matched_terms"
_KEY_MATCHED_CANONS = "matched_canons"
_KEY_TERM_IDF = "term_idf"
_KEY_ENTITY_IDF = "entity_idf"


# ---------------------------------------------------------------------------
# providers / context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureProviders:
    """Optional injected lookups for features needing corpus/ctx access.

    Every field is a callable or ``None``. The default feature function
    prefers candidate-carried signals and only consults a provider when the
    signal is absent; whatever cannot be resolved is simply not emitted.
    """

    term_idf: Optional[Callable[[str], float]] = None
    canon_idf: Optional[Callable[[str], float]] = None
    unit_terms: Optional[Callable[[str], Iterable]] = None
    ctx_terms: Optional[Callable[[str], Iterable]] = None  # unit ∪ ±1 neighbors
    unit_canons: Optional[Callable[[str], Iterable]] = None
    unit_identifiers: Optional[Callable[[str], Iterable]] = None
    unit_speaker: Optional[Callable[[str], Optional[str]]] = None
    unit_perspective: Optional[Callable[[str], Optional[str]]] = None
    unit_lifecycle: Optional[Callable[[str], Optional[str]]] = None
    unit_interval: Optional[Callable[[str], Any]] = None  # IntervalUs|(s,e)|us
    unit_event_predicates: Optional[Callable[[str], Iterable]] = None
    query_predicates: Optional[Callable[[QueryViewV7], Iterable]] = None
    corroboration_families: Optional[Callable[[str], int]] = None
    expected_perspective: Optional[Callable[[QueryViewV7], Optional[str]]] = None
    phrase_score: Optional[Callable[[QueryViewV7, FusedCandidate], float]] = None
    ident_equivalent: Optional[Callable[[str, str], bool]] = None
    #: V75-03.05 — the query's entity canons that are *speaker* canons
    #: present in the queried scope. Consulted once per scoring call in
    #: ``make_context`` (query-invariant), and only when the caller gave
    #: no ``speaker_hint`` — the hint always wins. ``None``/``()``
    #: means unresolvable; ≥ 2 means ambiguous (neutral, never a guess).
    query_speakers: Optional[Callable[[QueryViewV7], Iterable]] = None


@dataclass(frozen=True)
class FeatureContext:
    """Pool-level + query-invariant facts handed to every ``feature_fn``
    call. The ``q_*``/``window_*`` fields are derived once per
    ``score_candidates`` invocation so feature functions never re-parse the
    query per candidate (V7-10.07 latency budget)."""

    max_rrf: float
    n_lanes: int
    providers: FeatureProviders
    suppress: frozenset  # V7-10.03 constant signals / features to zero
    query_time_us: Optional[int]
    q_terms: tuple = ()       # deduped non-identifier query terms
    q_idents: tuple = ()      # identifier-channel query terms
    q_canons: tuple = ()
    q_speaker: Optional[str] = None
    q_intent: str = ""        # primary intent value
    window: Any = None        # intent.window (IntervalUs | None)
    window_known: bool = False
    # V75-03.05 — query-derived speaker resolution, computed once.
    q_speakers: tuple = ()           # resolved speaker canons (query path)
    q_speaker_source: str = "none"   # hint | query | ambiguous | none


def make_context(
    query: QueryViewV7,
    fused: Iterable,
    providers: Optional[FeatureProviders] = None,
    n_lanes: Optional[int] = None,
    suppress: Iterable = (),
) -> FeatureContext:
    """Build the per-invocation :class:`FeatureContext` (also usable by the
    pipeline for ``feature_vector`` explain calls)."""
    items = list(fused or [])
    max_rrf = max((float(c.rrf) for c in items), default=0.0)
    if n_lanes is None:
        lanes_seen = {lane for c in items for lane in (c.lane_ranks or {})}
        n_lanes = max(len(lanes_seen), 1)
    window = _query_window(query)
    prov = providers or FeatureProviders()
    q_speaker, q_speakers, q_speaker_source = _resolve_q_speaker(query, prov)
    return FeatureContext(
        max_rrf=max_rrf,
        n_lanes=int(n_lanes),
        providers=prov,
        suppress=frozenset(str(s) for s in (suppress or ())),
        query_time_us=getattr(query, "query_time_us", None),
        q_terms=tuple(_query_content_terms(query)),
        q_idents=tuple(_query_identifiers(query)),
        q_canons=tuple(str(c) for c in (getattr(query, "entity_canons", ()) or ())),
        q_speaker=q_speaker,
        q_intent=_ival(getattr(getattr(query, "intent", None), "primary", "")),
        window=window,
        window_known=_window_known(window),
        q_speakers=q_speakers,
        q_speaker_source=q_speaker_source,
    )


class ScoredList(list):
    """``list[ScoredCandidate]`` plus run metadata for coverage/explain."""

    def __init__(self, rows: Iterable = (), stats: Optional[dict] = None) -> None:
        super().__init__(rows)
        self.stats: dict = dict(stats or {})


# ---------------------------------------------------------------------------
# small deterministic helpers
# ---------------------------------------------------------------------------


def _canonicalize(value: Any) -> Any:
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


def _tie_epsilon(cand: FusedCandidate) -> float:
    """Deterministic disambiguation in ``[0, TIE_EPSILON)`` from the
    candidate's canonical signal vector (V7-10.02)."""
    canon = _canonicalize(cand.signals or {})
    blob = repr(canon).encode("utf-8")
    digest = hashlib.blake2b(blob, digest_size=8).digest()
    return (int.from_bytes(digest, "big") / float(1 << 64)) * TIE_EPSILON


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _sig(signals: Mapping, *keys: str) -> Any:
    for key in keys:
        if key in signals:
            return signals[key]
    return None


def _prov(call: Optional[Callable], *args: Any) -> Any:
    if call is None:
        return None
    try:
        return call(*args)
    except Exception:
        return None


def _ival(intent: Any) -> str:
    return str(getattr(intent, "value", intent))


def _query_content_terms(query: QueryViewV7) -> list:
    """Deduped non-identifier query terms (analysis order preserved)."""
    terms = getattr(getattr(query, "norm", None), "terms", ()) or ()
    seen: dict = {}
    for t in terms:
        term = getattr(t, "term", None)
        channel = getattr(t, "channel", "text")
        if term is None or channel == "identifier":
            continue
        seen.setdefault(str(term), None)
    return list(seen)


def _query_identifiers(query: QueryViewV7) -> list:
    idents = getattr(getattr(query, "norm", None), "identifiers", ()) or ()
    seen: dict = {}
    for t in idents:
        term = getattr(t, "term", None)
        if term is not None:
            seen.setdefault(str(term), None)
    return list(seen)


def _query_window(query: QueryViewV7) -> Any:
    return getattr(getattr(query, "intent", None), "window", None)


def _resolve_q_speaker(query: QueryViewV7, prov: FeatureProviders) -> tuple:
    """``(q_speaker, q_speakers, source)`` — the question's subject
    speaker, resolved once per scoring call (V75-03.05).

    - ``query.speaker_canon`` set (caller ``speaker_hint``) → wins
      outright; source ``hint``, provider never consulted.
    - Otherwise the ``query_speakers`` provider resolves which of the
      query's entity canons are speaker canons in scope: exactly one →
      that canon, source ``query``; two or more → ``ambiguous`` (the
      feature stays neutral 0.5 for every unit — never a guess);
      zero/``None``/provider error → ``none``.
    """
    hint = getattr(query, "speaker_canon", None)
    if hint is not None:
        return str(hint), (), "hint"
    resolved = _prov(prov.query_speakers, query)
    if resolved:
        speakers = tuple(dict.fromkeys(str(c) for c in resolved if str(c)))
        if len(speakers) == 1:
            return speakers[0], speakers, "query"
        if speakers:
            return None, speakers, "ambiguous"
    return None, (), "none"


def _lane_ctx_query_speakers(ctx: Any) -> Optional[Callable]:
    """Synthesize the ``query_speakers`` provider from a LaneContextV7-
    shaped ``ctx`` (the pipeline calls ``score_candidates(..., ctx=ctx)``).

    When the object carries ``scope_id``/``generation`` plus a resolvable
    pinned read connection (``entity.lane_conn`` — the lane's own
    resolution order), the query's entity canons probe
    ``units.speaker_canon`` at/below the fence via
    ``entity.resolve_query_speaker_canons``. Anything unresolvable —
    wrong shape, no snapshot, module absent — yields ``None`` and the
    feature simply stays neutral; a provider is never fabricated.
    """
    if (
        ctx is None
        or isinstance(ctx, (FeatureProviders, Mapping))
        or callable(ctx)
    ):
        return None
    if getattr(ctx, "scope_id", None) is None:
        return None
    generation = getattr(ctx, "generation", None)
    try:
        generation = int(generation)
    except (TypeError, ValueError):
        return None
    try:
        from .entity import lane_conn, resolve_query_speaker_canons
    except Exception:  # noqa: BLE001 — lane module absent
        return None
    try:
        conn = lane_conn(ctx)
    except Exception:  # noqa: BLE001
        return None
    if conn is None:
        return None
    scope_id = str(ctx.scope_id)

    def query_speakers(q: QueryViewV7) -> tuple:
        return tuple(
            resolve_query_speaker_canons(
                conn,
                scope_id,
                generation,
                getattr(q, "entity_canons", ()) or (),
            )
        )

    return query_speakers


def _window_known(window: Any) -> bool:
    return (
        window is not None
        and getattr(window, "start_us", None) is not None
        and getattr(window, "end_us", None) is not None
    )


def _unit_time_us(signals: Mapping, prov: FeatureProviders, unit_id: str) -> Optional[float]:
    """Midpoint of the occurred interval when known, else recorded time."""
    interval = _prov(prov.unit_interval, unit_id)
    start = end = None
    if interval is not None:
        if isinstance(interval, (tuple, list)) and len(interval) >= 2:
            start, end = _num(interval[0]), _num(interval[1])
        else:
            start = _num(getattr(interval, "start_us", None))
            end = _num(getattr(interval, "end_us", None))
            single = _num(interval)
            if single is not None:
                start = end = single
    if start is None:
        start = _num(_sig(signals, "occurred_start_us"))
    if end is None:
        end = _num(_sig(signals, "occurred_end_us"))
    if start is not None and end is not None:
        return (start + end) / 2.0
    if start is not None:
        return start
    if end is not None:
        return end
    return _num(_sig(signals, "recorded_at_us"))


# ---------------------------------------------------------------------------
# default feature function
# ---------------------------------------------------------------------------


def default_features(
    query: QueryViewV7,
    cand: FusedCandidate,
    ctx: FeatureContext,
) -> Mapping:
    """Compute the §32.4 feature vector from candidate signals + query.

    Lane-emitted feature-shaped signals take precedence (clamped to the
    declared range); corpus/context features resolve through
    ``ctx.providers``; whatever cannot be computed is absent — never
    invented.
    """
    signals = cand.signals or {}
    prov = ctx.providers or FeatureProviders()
    vals: dict = {}

    # -- rrf_norm: rrf(d) / max rrf in pool — always computable ----------
    vals["rrf_norm"] = (cand.rrf / ctx.max_rrf) if ctx.max_rrf > 0 else 0.0

    # -- cov_idf: Σ idf(matched query terms) / Σ idf(all query terms) ----
    qterms = ctx.q_terms
    if qterms:
        idf_of = prov.term_idf or (lambda t: 1.0)
        term_idf_sig = _sig(signals, _KEY_TERM_IDF)
        matched = _sig(signals, _KEY_MATCHED_TERMS)
        matched_set: Optional[set] = None
        if matched is not None or isinstance(term_idf_sig, Mapping):
            matched_set = set(matched or ()) | set(term_idf_sig or ())
        else:
            unit_terms = _prov(prov.unit_terms, cand.unit_id)
            if unit_terms is not None:
                matched_set = set(unit_terms) & set(qterms)
        if matched_set is not None:
            denom = math.fsum(float(idf_of(t)) for t in qterms)
            num = math.fsum(float(idf_of(t)) for t in matched_set if t in qterms)
            vals["cov_idf"] = (num / denom) if denom > 0 else 0.0

    # -- phrase: lane NEAR/phrase signal or provider ----------------------
    phrase = _num(_sig(signals, "phrase"))
    if phrase is None:
        phrase = _num(_prov(prov.phrase_score, query, cand))
    if phrase is not None:
        vals["phrase"] = phrase

    # -- ent_idf: Σ idf(matched query canons) / Σ idf(query canons) -------
    canons = ctx.q_canons
    if canons:
        canon_idf_sig = _sig(signals, _KEY_ENTITY_IDF)
        matched_c = _sig(signals, _KEY_MATCHED_CANONS)
        canon_set: Optional[set] = None
        if matched_c is not None or isinstance(canon_idf_sig, Mapping):
            canon_set = set(matched_c or ()) | set(canon_idf_sig or ())
        else:
            unit_canons = _prov(prov.unit_canons, cand.unit_id)
            if unit_canons is not None:
                canon_set = set(str(c) for c in unit_canons) & set(canons)
        if canon_set is not None:
            cidf_sig = canon_idf_sig if isinstance(canon_idf_sig, Mapping) else {}

            def cidf(c: str) -> float:
                v = _num(cidf_sig.get(c))
                if v is not None:
                    return v
                pv = _num(_prov(prov.canon_idf, c))
                return pv if pv is not None else 1.0

            denom = math.fsum(cidf(c) for c in canons)
            num = math.fsum(cidf(c) for c in canon_set if c in canons)
            vals["ent_idf"] = (num / denom) if denom > 0 else 0.0

    # -- speaker_match ----------------------------------------------------
    # ctx.q_speaker is the effective subject speaker resolved once in
    # make_context: caller hint (source "hint"), else the query-derived
    # scope speaker canon (source "query"); ambiguous/none → None → the
    # neutral 0.5 (V75-03.05). Bounded feature only — never an
    # eligibility gate.
    q_speaker = ctx.q_speaker
    if q_speaker is None:
        vals["speaker_match"] = 0.5
    else:
        u_speaker = _sig(signals, "speaker_canon")
        if u_speaker is None:
            u_speaker = _prov(prov.unit_speaker, cand.unit_id)
        if u_speaker is None:
            vals["speaker_match"] = 0.5
        else:
            vals["speaker_match"] = 1.0 if str(u_speaker) == str(q_speaker) else 0.0

    # -- t_prox: distance from window centre relative to half-width -------
    window = ctx.window
    if window is None or not ctx.window_known:
        vals["t_prox"] = 0.5
    else:
        t_unit = _unit_time_us(signals, prov, cand.unit_id)
        if t_unit is None:
            vals["t_prox"] = 0.5
        else:
            centre = (float(window.start_us) + float(window.end_us)) / 2.0
            half = (float(window.end_us) - float(window.start_us)) / 2.0
            if half <= 0:
                vals["t_prox"] = 1.0 if t_unit == centre else 0.0
            else:
                vals["t_prox"] = 1.0 - min(abs(t_unit - centre) / half, 1.0)

    # V85-05.04 — the measured dead features (``cov_idf_ctx``,
    # ``ident_exact``, ``life_state``, ``corrob``, ``perspective_fit``,
    # ``event_pred``, ``lane_agree``) are no longer computed: they are
    # absent from the vector and from ``score_detail``, never a silent
    # zero.  The ``feature_fn``/provider seam can still re-add one for a
    # declared arm — ``FEATURE_RANGES`` keeps its clamp.
    return vals


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _clean_features(raw: Any, suppress: frozenset) -> dict:
    """Normalize a feature_fn result to a clean ``{name: float}`` mapping.

    Suppressed names (V7-10.03 constant signals) are dropped; declared
    features are clamped to their §32.4 range; non-finite values are absent.
    """
    if isinstance(raw, FeatureVector):
        raw = raw.values
    vals: dict = {}
    for name, value in dict(raw or {}).items():
        name = str(name)
        if name in suppress:
            continue
        fv = _num(value)
        if fv is None:
            continue
        rng = FEATURE_RANGES.get(name)
        if rng is not None:
            fv = min(max(fv, rng[0]), rng[1])
        vals[name] = fv
    return vals


def feature_vector(
    query: QueryViewV7,
    cand: FusedCandidate,
    ctx: FeatureContext,
    feature_fn: Optional[Callable] = None,
) -> FeatureVector:
    """Public explain helper: the candidate's clamped feature vector."""
    fn = feature_fn or default_features
    return FeatureVector(unit_id=cand.unit_id,
                         values=_clean_features(fn(query, cand, ctx), ctx.suppress))


def _unit_seq_key(cand: Any) -> tuple:
    """``unit seq`` tie-break element: integer ``seq`` signal when present,
    else ``unit_id`` — deterministic either way."""
    signals = getattr(cand, "signals", None)
    if signals is None:
        signals = (getattr(cand, "detail", None) or {}).get("signals") or {}
    seq = _num((signals or {}).get("seq"))
    if seq is not None and float(seq).is_integer():
        return (0, int(seq), "")
    return (1, 0, str(getattr(cand, "unit_id", "")))


def _coerce_ctx_arg(ctx: Any, providers: Optional[FeatureProviders],
                    feature_fn: Optional[Callable]):
    """Back-compat shim for the frozen ``(query, fused, ctx)`` call shape:
    a ``ctx`` that is a callable becomes ``feature_fn``; a
    ``FeatureProviders`` becomes ``providers``; a mapping is splatted into
    ``FeatureProviders``. Any other object (e.g. the pipeline's
    ``LaneContextV7``) is left for ``score_candidates`` — it can still
    contribute the ``query_speakers`` provider (V75-03.05)."""
    if ctx is None:
        return providers, feature_fn
    if isinstance(ctx, FeatureProviders):
        return ctx, feature_fn
    if callable(ctx):
        return providers, ctx
    if isinstance(ctx, Mapping):
        fields = {f for f in FeatureProviders.__dataclass_fields__}
        return FeatureProviders(**{k: v for k, v in ctx.items() if k in fields}), feature_fn
    return providers, feature_fn


def score_candidates(
    query: QueryViewV7,
    fused: Iterable[FusedCandidate],
    feature_fn: Optional[Callable] = None,
    *,
    ctx: Any = None,
    providers: Optional[FeatureProviders] = None,
    weights: Optional[Mapping] = None,
    n_lanes: Optional[int] = None,
    suppress: Iterable = (),
    cheap: bool = False,
    budget_exceeded: Optional[Callable[[], bool]] = None,
) -> ScoredList:
    """Score the fused pool with the ``rerank_features/v1`` linear model.

    ``feature_fn(query, cand, FeatureContext) -> Mapping[str,float]`` is the
    injection seam; ``None`` selects :func:`default_features`. ``providers``
    (or a compatible ``ctx``) supplies corpus lookups. ``weights`` overrides
    :data:`FEATURE_WEIGHTS_V1` outright. ``n_lanes`` sizes
    ``FeatureContext.n_lanes`` for custom feature functions (default:
    distinct lanes observed in the pool; the shipped feature set no longer
    consumes it). ``suppress`` zeroes named features/signals (V7-10.03
    constants — pass ``FusedList.stats["constant_signals"]``).

    ``cheap=True`` selects the V8-14.04 cheap feature subset: every
    provider lookup is suppressed, so only signal-carried and
    query-invariant features emit (``rrf_norm`` and any lane-computed
    signals) — no SQL, no corpus probes. ``budget_exceeded``
    is the deadline hook the pipeline passes so the check fires *inside*
    this post stage: polled before each candidate, the first ``True``
    switches the remaining pool to the cheap subset and records
    ``stats["deadline_degraded"]`` (the per-candidate count follows the
    same honest-partial rule as a lane cut by its slice).

    Ordering: ``(-score, -rrf_norm, source_id, revision desc, unit seq)`` —
    total and byte-stable.
    """
    providers, feature_fn = _coerce_ctx_arg(ctx, providers, feature_fn)
    prov = providers or FeatureProviders()
    if cheap:
        prov = FeatureProviders()  # V8-14.04: signal-carried features only
    if prov.query_speakers is None and not cheap:
        # V75-03.05 — when ``ctx`` is the pipeline's LaneContextV7 the
        # query's speaker canons resolve against its pinned read
        # snapshot; explicit providers always win.
        qs = _lane_ctx_query_speakers(ctx)
        if qs is not None:
            prov = dc_replace(prov, query_speakers=qs)
    fn = feature_fn or default_features
    w = dict(weights) if weights is not None else dict(FEATURE_WEIGHTS_V1)
    # V8-11.02 ``rerank.speaker_match_weight`` — the §23 arm governs the
    # coefficient whenever it is declared on ``ctx.policy`` (or the
    # manifest) and the caller's ``weights`` map did not pin
    # ``speaker_match`` explicitly: per-coefficient, the most local
    # declaration wins (caller table > policy arm > §32.4 default).
    speaker_arm = _speaker_match_weight_arm(ctx)
    speaker_w_src = "default"
    if speaker_arm is not None and not (
        weights is not None and "speaker_match" in weights
    ):
        w["speaker_match"] = speaker_arm
        speaker_w_src = "policy_arm"
    elif weights is not None and "speaker_match" in weights:
        speaker_w_src = "weights"
    suppress_set = frozenset(str(s) for s in (suppress or ()))

    items = list(fused or [])
    stats: dict = {
        "model": RERANK_MODEL_ID,
        "score_family": SCORE_FAMILY,
        "formula_status": FORMULA_STATUS_PROVISIONAL,
        "candidates": len(items),
        "weights": dict(w),
        "suppressed_requested": sorted(suppress_set),
        "mode": "cheap" if cheap else "full",
        # V8-11.02 — the resolved §23 arm value and which carrier won;
        # recorded even on an empty pool (the arm still resolved).
        "speaker_match_weight": {
            "value": w.get("speaker_match", SPEAKER_MATCH_WEIGHT_DEFAULT),
            "declared": speaker_arm,
            "source": speaker_w_src,
        },
    }
    if not items:
        stats["features_seen"] = []
        return ScoredList([], stats=stats)

    fctx = make_context(query, items, providers=prov, n_lanes=n_lanes,
                        suppress=suppress_set)

    # Pass 1: pure Σ w·feature per candidate.
    rows: list = []  # (cand, vals, raw_map, feature_score)
    features_seen: set = set()
    suppressed_seen: set = set()
    degraded_at: Optional[int] = None  # index where the deadline hook fired
    full_fctx = fctx                   # stats provenance keeps the full ctx
    cheap_fctx: Optional[FeatureContext] = None
    for idx, cand in enumerate(items):
        if not cheap and budget_exceeded is not None:
            try:
                blown = bool(budget_exceeded())
            except Exception:  # noqa: BLE001 — a bad clock never widens work
                blown = False
            if blown:
                cheap = True
                degraded_at = idx
                if cheap_fctx is None:
                    cheap_fctx = make_context(
                        query,
                        items,
                        providers=FeatureProviders(),
                        n_lanes=fctx.n_lanes,
                        suppress=suppress_set,
                    )
                fctx = cheap_fctx
        raw = fn(query, cand, fctx)
        if isinstance(raw, FeatureVector):
            raw = raw.values
        raw_map = dict(raw or {})
        vals = _clean_features(raw_map, suppress_set)
        suppressed_seen.update(suppress_set & set(raw_map))
        features_seen.update(vals)
        parts = [float(w[name]) * vals[name] for name in vals if name in w]
        rows.append((cand, vals, raw_map, math.fsum(parts)))

    # Pass 2 (V7-10.02): only members of an exactly-tied feature-score group
    # receive a deterministic tie_epsilon — untied candidates keep
    # score == Σ w·feature bit-for-bit.
    groups: dict = {}
    for idx, (_c, _v, _r, fs) in enumerate(rows):
        groups.setdefault(fs, []).append(idx)
    eps_of: dict = {}
    for fs, members in groups.items():
        if len(members) > 1:
            for idx in members:
                eps_of[idx] = _tie_epsilon(rows[idx][0])
    stats["tied_groups"] = sum(1 for m in groups.values() if len(m) > 1)

    scored: list = []
    for idx, (cand, vals, raw_map, feature_score) in enumerate(rows):
        eps = eps_of.get(idx, 0.0)
        score = feature_score + eps
        detail = {
            "model": RERANK_MODEL_ID,
            "formula_status": FORMULA_STATUS_PROVISIONAL,
            "features": dict(sorted(vals.items())),
            "feature_score": feature_score,
            "tie_epsilon": eps,
            "weights": {name: float(w[name]) for name in vals if name in w},
            "rrf": float(cand.rrf),
            "rrf_norm": vals.get("rrf_norm", 0.0),
            "lane_ranks": dict(cand.lane_ranks or {}),
            "suppressed": sorted(suppress_set & set(raw_map)),
            "signals": cand.signals or {},
            # V75-03.05 provenance: which path resolved the subject
            # speaker — hint | query | ambiguous | none.
            "speaker_match_source": fctx.q_speaker_source,
        }
        scored.append(ScoredCandidate(
            unit_id=cand.unit_id,
            source_id=cand.source_id,
            revision=cand.revision,
            score=score,
            score_family=SCORE_FAMILY,
            detail=detail,
        ))

    scored.sort(key=lambda c: (
        -c.score,
        -float(c.detail.get("rrf_norm", 0.0)),
        c.source_id,
        -c.revision,
        _unit_seq_key(c),
    ))
    stats["features_seen"] = sorted(features_seen)
    stats["suppressed_applied"] = sorted(suppressed_seen)
    if degraded_at is not None:
        # V8-14.04 — the deadline hook fired mid-pool; the tail scored on
        # the cheap subset (provider-free), the head kept full features.
        stats["deadline_degraded"] = len(items) - degraded_at
        stats["mode"] = "deadline_degraded"
    # V75-03.05 speaker_match provenance for coverage/evals.
    stats["speaker_match_source"] = full_fctx.q_speaker_source
    stats["speaker_match_canon"] = full_fctx.q_speaker
    stats["speaker_match_query_canons"] = list(full_fctx.q_speakers)
    return ScoredList(scored, stats=stats)


__all__ = [
    "FEATURE_RANGES",
    "FEATURE_WEIGHTS_V1",
    "SPEAKER_MATCH_WEIGHT_DEFAULT",
    "FeatureContext",
    "FeatureProviders",
    "RERANK_MODEL_ID",
    "SCORE_FAMILY",
    "ScoredList",
    "TIE_EPSILON",
    "default_features",
    "feature_vector",
    "make_context",
    "score_candidates",
]
