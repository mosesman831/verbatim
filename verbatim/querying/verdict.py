"""Per-class support verdicts for the consumer search route
(``support_verdict/v1`` — SPEC_V5 §31.3, V5-31.07/31.08/31.09; worker
contract docs/v5_contracts.md §8).

``search_verdict`` is the deterministic support gate the consumer route
applies to the *merged* hit list — after eligibility, authorization,
and ``ranking/v1`` ordering have run — to decide which candidates
actually carry support for the asked question. It is pure and versioned
like ``analyze``: same inputs, same verdict, receipt-loggable detail.

Rules (V5-31.07 — per-class coverage floors; a similarity score is never
a global threshold copied between encoders, V5-31.08):

- **identifier-bearing queries** require the exact identifier
  (V5-30.17): a hit supports only when it carries the
  ``identifier_hit`` signal or its delivered text contains the
  identifier's exact surface form — byte-exact, case preserved.
- **all other queries** require content coverage: a hit needs a real
  match signal (``lexical``/``identifier_hit``/``entity_overlap``) *and*
  — when its delivered text is inspectable — at least
  ``ceil(term_floor_ratio · |terms|)`` of the query's content terms
  (default 0.5, the same SUPPORTED floor the governed lanes apply under
  V3-29.05). A ``similarity``-only candidate may instead carry support
  through the *calibrated* path: its raw similarity — the encoder-native
  value, never the fused ``score_detail`` contribution — must meet the
  provisioned encoder's fitted support floor
  (``querying/calibration.py``, bound to the pinned ``encoder_id``).
  An encoder with no fitted calibration fails closed: similarity alone
  never supports under it, so a high score on unrelated text remains
  the least-bad hit V5-31.09 forbids.
- **likely-no-answer queries** (``no_answer_likely``) suppress every
  candidate: the class asserts the requested content is absent, so any
  delivered hit would be the least-bad one (V5-31.09).

The verdict distinguishes "not found in a completed declared search"
(``no_candidates``) from "support insufficient" (``insufficient`` —
candidates were examined and none carried support), per V5-13.10.
Suppressed hits are never delivered; the coverage detail reports their
count and the rejection reason codes — never the objects themselves
(V5-31.06). Kept hits are stamped ``support_status="supported"``: the
assessment the verdict performed is what the hit then reports.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Iterable, Optional, Tuple

from ..memory.types import QueryClass, SupportStatus

try:  # support_calibration/v1 — same package; a broken import must fail
    from .calibration import CALIBRATION_VERSION as _CALIBRATION_VERSION
    from .calibration import similarity_floor as _similarity_floor
except Exception:  # pragma: no cover - closed, never loud
    _similarity_floor = None
    _CALIBRATION_VERSION = "support_calibration/v1"

#: Version pin for this verdict contract — bumped on any rule change.
VERDICT_VERSION = "support_verdict/v1"

#: Default content-term coverage floor (V3-29.05 parity): a candidate
#: supports a factual read when its delivered text carries at least half
#: of the query's content terms. Identifier-class queries are exempt —
#: the exact identifier IS the support.
DEFAULT_TERM_FLOOR_RATIO = 0.5

#: Signals that constitute a real (non-similarity) match: BM25 over the
#: lexical projection, exact identifier postings, entity postings.
#: ``similarity``/``temporal_match``/``type_affinity``/``corroboration``/
#: ``lifecycle_current`` are modifiers or uncalibrated support — they
#: rank and annotate, never ground a hit alone.
_EVIDENCE_SIGNALS = frozenset(
    {"lexical", "identifier_hit", "entity_overlap"}
)

#: ``<memory_evidence ...>{json}</memory_evidence>`` item wrapper —
#: coverage is measured on the delivered content, not the envelope.
_UNTRUSTED_OPEN = "<memory_evidence"
_UNTRUSTED_CLOSE = "</memory_evidence>"

#: Rejection reason codes surfaced in ``SupportVerdict.detail``.
_REASON_NO_ANSWER_CLASS = "no_answer_query"
_REASON_NO_IDENTIFIER = "missing_exact_identifier"
_REASON_BELOW_FLOOR = "below_term_floor"
_REASON_NO_SIGNAL = "no_match_signal"
_REASON_NO_TEXT = "unverifiable_text"


def _normalize(text: str) -> str:
    """``norm/v1`` fold — same normalization the lexical projection and
    the analyzer apply, resolved through the enrichment seam."""
    from .analyze import normalize_text

    return normalize_text(text)


def _content_term_set(terms: Iterable[str]) -> frozenset:
    """Normalized, split content terms — a hyphenated/accented surface
    term folds to the same token set the projections index."""
    out = set()
    for t in terms or ():
        for tok in _normalize(str(t)).split():
            if tok:
                out.add(tok)
    return frozenset(out)


def _hit_signals(hit: Any) -> frozenset:
    """Declared signal names on a hit — fusion ``score_detail`` keys or
    a raw ``signals`` mapping, on objects or plain mappings."""
    names = set()
    for attr in ("score_detail", "signals"):
        if isinstance(hit, Mapping):
            raw = hit.get(attr)
        else:
            raw = getattr(hit, attr, None)
        if isinstance(raw, Mapping):
            names.update(str(k) for k in raw)
    return frozenset(names)


def _raw_similarity(hit: Any) -> Optional[float]:
    """The encoder-native ``similarity`` value on a hit, if present.

    Only the lane's raw ``signals`` mapping qualifies: fused
    ``score_detail`` carries *contributions* (normalized × weight), which
    are bounded by the ranking weight table and can never be compared to
    an encoder-native support floor — treating them as cosines would be
    a silent unit mismatch.
    """
    if isinstance(hit, Mapping):
        raw = hit.get("signals")
    else:
        raw = getattr(hit, "signals", None)
    if not isinstance(raw, Mapping) or "similarity" not in raw:
        return None
    try:
        value = float(raw["similarity"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _similarity_supported(
    hit: Any, *, encoder: Optional[str], query_class: Optional[str]
) -> bool:
    """True when the hit's raw similarity meets the pinned encoder's
    calibrated support floor (V5-31.07/31.08).

    Every gate fails closed: no fitted floor for the encoder identity or
    query class, no raw similarity value, or a score below the floor all
    resolve to "similarity does not carry support".
    """
    if _similarity_floor is None:
        return False
    floor = _similarity_floor(encoder, query_class)
    if floor is None:
        return False
    value = _raw_similarity(hit)
    return value is not None and value >= floor


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _delivered_text(hit: Any) -> Optional[str]:
    """The hit's delivered text for coverage measurement — ``quote`` or
    ``text``; ``<memory_evidence>`` wrappers are unwrapped to their
    content-bearing string fields so envelope metadata never fabricates
    coverage."""
    if isinstance(hit, Mapping):
        text = hit.get("quote", hit.get("text"))
    else:
        text = getattr(hit, "quote", None) or getattr(hit, "text", None)
    if not isinstance(text, str) or not text:
        return None
    if text.startswith(_UNTRUSTED_OPEN) and text.endswith(_UNTRUSTED_CLOSE):
        body = text[text.index(">") + 1: -len(_UNTRUSTED_CLOSE)]
        try:
            payload = json.loads(body)
        except Exception:
            return text  # opaque wrapper — coverage on the raw form
        parts = [s for s in _strings(payload) if s.strip()]
        return " ".join(parts) if parts else None
    return text


@dataclass(frozen=True)
class SupportVerdict:
    """Result of the consumer-route support verdict.

    ``verdict`` is ``supported`` (≥1 hit carried support),
    ``insufficient`` (candidates existed; none did), or
    ``no_candidates`` (nothing arrived to judge — "not found" is already
    honest under the caller's own status). ``kept``/``suppressed`` hold
    the hit objects; ``reasons`` deduplicates the suppression codes.
    """

    version: str = VERDICT_VERSION
    verdict: str = "no_candidates"
    kept: Tuple[Any, ...] = ()
    suppressed: Tuple[Any, ...] = ()
    reasons: Tuple[str, ...] = ()
    term_floor: int = 0
    detail: dict = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()


def _hit_supported(
    hit: Any,
    *,
    term_set: frozenset,
    floor: int,
    identifiers: Tuple[str, ...],
    entities: Tuple[str, ...],
    encoder: Optional[str] = None,
    query_class: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """``(reason, via)`` — ``reason`` is ``None`` when the hit supports
    the query (``via`` then names the support path: ``identifier``,
    ``signal``, ``coverage``, ``entity``, or ``similarity``); otherwise
    it is the suppression reason code."""
    signals = _hit_signals(hit)
    text = _delivered_text(hit)

    # Identifier-bearing queries: the exact identifier is the support
    # contract (V5-31.07 + V5-30.17) — a fuzzy or coverage hit that
    # lacks it is never an answer to an identifier read, and calibrated
    # similarity deliberately cannot rescue one.
    if identifiers:
        if "identifier_hit" in signals:
            return None, "identifier"
        if text is not None and any(i in text for i in identifiers):
            return None, "identifier"
        return _REASON_NO_IDENTIFIER, None

    covered = 0
    if text is not None and term_set:
        tokens = frozenset(_normalize(text).split())
        covered = len(term_set & tokens)

    if signals & _EVIDENCE_SIGNALS:
        # A real match (BM25 / exact posting) — coverage floor applies
        # only when the delivered text is measurable; a hit whose text
        # the caller may not see still carries its verified signal.
        if text is not None and floor and covered < floor:
            return _REASON_BELOW_FLOOR, None
        return None, "signal"

    # Similarity-only or unranked governed-lane candidates: the
    # delivered text itself must cover the term floor, OR the
    # candidate's raw similarity must clear the provisioned encoder's
    # calibrated support floor (V5-31.07/31.08 — encoder-bound, never a
    # copied global threshold; uncalibrated encoders fail closed).
    if term_set:
        if text is not None and covered >= max(1, floor):
            return None, "coverage"
        if _similarity_supported(
            hit, encoder=encoder, query_class=query_class
        ):
            return None, "similarity"
        return (
            (_REASON_NO_TEXT if text is None else _REASON_BELOW_FLOOR),
            None,
        )

    # No content terms at all — only entity coverage can support.
    if entities:
        if "entity_overlap" in signals:
            return None, "entity"
        if text is not None and any(e in text for e in entities):
            return None, "entity"
    return _REASON_NO_SIGNAL, None


def search_verdict(
    hits: Iterable[Any],
    *,
    terms: Iterable[str] = (),
    identifiers: Iterable[str] = (),
    entities: Iterable[str] = (),
    primary: Optional[str] = None,
    term_floor_ratio: float = DEFAULT_TERM_FLOOR_RATIO,
    encoder: Optional[str] = None,
) -> SupportVerdict:
    """Apply the per-class support verdict to a merged hit list.

    ``hits`` are the post-authorization, post-ranking candidates (facade
    ``Hit`` objects or duck-typed equivalents exposing ``score_detail``/
    ``signals``/``quote``). ``terms``/``identifiers``/``entities``/
    ``primary`` come from ``query_analysis/v1``. ``encoder`` is the
    pinned ``encoder_id`` the similarity channel scored under — it
    selects that encoder's fitted support floor
    (``support_calibration/v1``); ``None`` or an uncalibrated identity
    fails closed and similarity alone never carries support. Order is
    preserved; kept hits are stamped ``support_status='supported'`` —
    the verdict IS the assessment the delivered hit reports (V5-06.06).
    """
    term_set = _content_term_set(terms)
    idents = tuple(str(i) for i in identifiers or () if i)
    ents = tuple(str(e) for e in entities or () if e)
    floor = (
        min(len(term_set), max(1, math.ceil(term_floor_ratio * len(term_set))))
        if term_set
        else 0
    )
    no_answer = primary == QueryClass.NO_ANSWER_LIKELY.value
    sim_floor = (
        _similarity_floor(encoder, primary)
        if (_similarity_floor is not None and encoder)
        else None
    )

    hit_list = list(hits or ())
    kept: list = []
    suppressed: list = []
    reasons: list = []
    similarity_supported = 0
    for hit in hit_list:
        if no_answer:
            reason, via = _REASON_NO_ANSWER_CLASS, None
        else:
            reason, via = _hit_supported(
                hit,
                term_set=term_set,
                floor=floor,
                identifiers=idents,
                entities=ents,
                encoder=encoder,
                query_class=primary,
            )
        if reason is None:
            kept.append(hit)
            if via == "similarity":
                similarity_supported += 1
            try:
                hit.support_status = SupportStatus.SUPPORTED.value
            except Exception:
                pass  # immutable/mapping hits — verdict still stands
        else:
            suppressed.append(hit)
            if reason not in reasons:
                reasons.append(reason)

    if not hit_list:
        verdict = "no_candidates"
    elif kept:
        verdict = SupportStatus.SUPPORTED.value
    else:
        verdict = SupportStatus.INSUFFICIENT.value

    detail = {
        "version": VERDICT_VERSION,
        "verdict": verdict,
        "kept": len(kept),
        "suppressed": len(suppressed),
        "term_floor": floor,
        "terms": len(term_set),
        "identifier_class": bool(idents),
        "reasons": list(reasons),
        "calibration": {
            "version": _CALIBRATION_VERSION,
            "encoder": encoder,
            "similarity_floor": sim_floor,
            "similarity_supported": similarity_supported,
        },
    }
    return SupportVerdict(
        verdict=verdict,
        kept=tuple(kept),
        suppressed=tuple(suppressed),
        reasons=tuple(reasons),
        term_floor=floor,
        detail=detail,
        warnings=("support_insufficient",) if suppressed else (),
    )


__all__ = [
    "DEFAULT_TERM_FLOOR_RATIO",
    "VERDICT_VERSION",
    "SupportVerdict",
    "search_verdict",
]
