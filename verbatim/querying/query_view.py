"""``query_view/v7`` — the S1 query-analysis builder (SPEC_V7 §04.2 S1).

``build_query_view`` turns raw query text into the immutable
``QueryViewV7`` every S2 lane consumes. It is a pure composition of the
landed wave-A analyzers — no models, no I/O of its own, no wall clock:

1. ``norm/v2`` ``analyze(query)`` → ``NormAnalysis`` (folded terms +
   identifier channel with byte offsets; V7-05.10/11).
2. ``entities/v2`` ``extract_query_entities(norm, known_canons)`` →
   canonical entity keys — greedy n-gram matching against the scope's
   known-canon vocabulary plus capitalized runs (V7-08.02). When
   ``aliases`` are supplied, ``expand_query`` widens the canons through
   ``active`` alias rows (≤ 8 per mention, V7-08.04); ``candidate`` /
   ``rejected`` rows never expand.
3. ``intent/v2`` ``classify(norm, entity_canons, identifiers)`` →
   ``IntentResult`` (V7-05.12). The classifier sees the *extracted*
   canons, not the alias-expanded set: alias expansion is a
   retrieval-side widening (V7-08.04), not evidence the query names two
   entities — a single mention plus its alias must not flip the intent
   to ``multi_hop``.
4. ``temporal/v2`` ``resolve_query_window(norm, now_us)`` →
   ``IntervalUs | None``, attached onto ``IntentResult.window`` (the
   field lives on ``IntentResult``, not on ``QueryViewV7``; V7-09.05).
   The window is attached to *every* intent class — §32.14 windows
   modify any question type — and is never fabricated: it is ``None``
   unless a temporal expression actually resolved, so abstain-likely
   queries without temporal markers carry no window.
5. ``intent/v2`` ``decompose(norm, intent)`` → facet ``NormAnalysis``
   objects (V7-05.13). Each becomes a full sub-``QueryViewV7`` (its own
   entity extraction, classification, and window against the same
   ``now_us`` anchor and the parent's speaker canon) with
   ``facets == ()`` — decomposition is one level deep. The ``(norm,)``
   singleton return means "no split" and yields ``facets == ()``.
6. ``intent/v2`` ``decompose_structural(norm, resolved)`` → canon-bound
   facets for coordinated subjects (V8-10.01). Fires whenever >= 2
   resolved entity *or speaker* canons are joined by a coordinator from
   ``coord_lex/v1`` (and, or, nor, vs, as well as — joins; both, each,
   either, neither, between — markers), independent of the intent class.
   Each facet keeps its canon's terms plus the query's remaining
   predicate terms and drops the other joined canons and the interior
   coordinator sites. Comparison primaries keep the intent-decompose
   shape first (their tail-hoist is comparison-specific — reported
   ``source="intent"``); every other class prefers the structural
   binding, falling back to the intent split when no coordination
   binds. Speaker-resolved seeds pin ``speaker_canon`` on the facet and
   are marked via :data:`FACET_SOURCE_STRUCTURAL` in the facet's
   ``intent.rule_trace`` so downstream coverage can attribute
   ``coverage.facets.source`` (:func:`facet_source`).

``known_canons`` (and ``aliases``) accept either an iterable or a
zero-arg callable returning one — the caller may need a DB lookup for
the scope's entity vocabulary; callables are invoked on every call and
their result is never cached across calls.

Deviation note (measured): ``extract_query_entities`` reads
``norm.text``, which under ``norm/v2`` is the *folded* matching
projection — so its capitalized-run signal is inert and
out-of-vocabulary names degrade honestly to lexical terms (the
degradation entities_v2 documents). Feeding it the raw surface via a
shadowed ``norm.text`` was evaluated and rejected: sentence-initial
scaffold words produce noise canons ("Where does Alice Chen work" →
``where``, "Compare Rome and Lisbon" → ``compare rome``) that flip rule
R14 to ``multi_hop`` on single-entity questions. Known-canon n-gram
matching — the signal that matters — is case-insensitive and unaffected.

Determinism: identical inputs produce byte-identical views; the only
time input is the caller-supplied ``now_us`` (recorded as
``query_time_us``). Empty or whitespace-only queries raise
``VerbatimError(VALIDATION)`` — a lane must never run on an empty view.

``builder_id`` is pinned as ``BUILDER_ID = "query_view/v7"``;
``QueryViewV7`` carries no builder field, so the id is a module-level
artifact tag for coverage/explain reporting downstream. All §32
constants remain ``provisional/v7-r0``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Optional, Tuple, Union

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v7 import (
    FORMULA_STATUS_PROVISIONAL,
    AliasRow,
    AliasState,
    IntentClass,
    NormAnalysis,
    QueryViewV7,
)
from verbatim.enrichment.entities_v2 import canon as _entities_canon
from verbatim.enrichment.entities_v2 import (
    expand_query,
    extract_query_entities,
)
from verbatim.enrichment.temporal_v2 import resolve_query_window
from verbatim.querying.intent_v2 import (
    MAX_FACETS,
    classify,
    decompose,
    decompose_structural,
)
from verbatim.text.norm_v2 import analyze

BUILDER_ID = "query_view/v7"

#: §32.0 — every §32 rule table is provisional until the formula search.
FORMULA_STATUS = FORMULA_STATUS_PROVISIONAL

#: V8-10.01 — provenance marker appended to a structural facet's
#: ``intent.rule_trace``. ``QueryViewV7`` has no source field, so the
#: marker rides the facet's trace and :func:`facet_source` reads it;
#: the pipeline emits ``coverage.facets.source`` from that (V8-20.03).
FACET_SOURCE_STRUCTURAL = "V8-10.01.structural"

__all__ = [
    "BUILDER_ID",
    "FORMULA_STATUS",
    "MAX_FACETS",
    "FACET_SOURCE_STRUCTURAL",
    "build_query_view",
    "facet_source",
]

#: ``known_canons`` / ``aliases`` may be a ready iterable or a zero-arg
#: callable producing one (DB-backed callers resolve lazily per call).
_Canons = Union[Iterable[str], Callable[[], Iterable[str]]]
_Aliases = Union[Iterable[AliasRow], Callable[[], Iterable[AliasRow]], None]


def _materialize(source) -> tuple:
    """Resolve an iterable-or-callable argument to a tuple.

    Callables are invoked here, on every call — never cached. A ``None``
    result is treated as empty; ``None`` itself is empty.
    """
    if source is None:
        return ()
    items = source() if callable(source) else source
    if items is None:
        return ()
    return tuple(items)


def _speaker_canon(
    speaker_hint: Optional[str],
    canon_fn: Optional[Callable[[str], str]],
) -> Optional[str]:
    """Canonical speaker key from the caller's hint.

    ``canon_fn`` defaults to ``entities/v2`` ``canon`` (fold + possessive
    strip). A blank surface or a canonizer returning an empty string is
    ``None``, never a fabricated key.
    """
    if speaker_hint is None:
        return None
    canonizer = canon_fn if canon_fn is not None else _entities_canon
    resolved = canonizer(str(speaker_hint))
    if resolved is None:
        return None
    resolved = str(resolved)
    return resolved or None


def _extraction_vocab(
    known: Tuple[str, ...],
    alias_rows: Tuple[AliasRow, ...],
) -> Tuple[str, ...]:
    """Entity vocabulary for ``extract_query_entities``.

    ``known_canons`` plus the surfaces of *active* alias rows — an active
    alias is a reviewed surface for a scope canon, so the n-gram matcher
    must recognize it for ``expand_query``'s alias→canon direction to be
    reachable at all (``norm.text`` is folded, so no capitalized-run
    fallback exists). ``candidate``/``rejected`` rows never widen the
    vocabulary — same rule as ``expand_query``.
    """
    if not alias_rows:
        return known
    extra = [
        s
        for a in alias_rows
        if a.state == AliasState.ACTIVE
        for s in (a.canon, a.alias_canon)
    ]
    return tuple(dict.fromkeys(known + tuple(extra)))


def _coordination_canons(
    norm: NormAnalysis,
    extracted: Tuple[str, ...],
    speaker_vocab: Tuple[str, ...],
    speaker_canon: Optional[str],
) -> Tuple[Tuple[str, ...], frozenset]:
    """Resolved entity+speaker canons for structural decomposition.

    The merged set drives V8-10.01 coordination detection *only* —
    speaker-resolved canons never join ``entity_canons`` on the parent
    view, so a coordinated-speaker question keeps its honest intent
    class (a ``lookup`` stays a ``lookup``; R14 must not see a speaker
    channel it was never calibrated on) while the structural path still
    binds both participants. Returns ``(resolved, speaker_set)`` —
    ``speaker_set`` marks which canons came via the speaker channel so
    the facet builder can pin ``speaker_canon`` on speaker-seeded
    facets.
    """
    speaker_resolved = (
        tuple(extract_query_entities(norm, speaker_vocab))
        if speaker_vocab
        else ()
    )
    resolved = list(dict.fromkeys((*extracted, *speaker_resolved)))
    speaker_set = set(speaker_resolved)
    if speaker_canon:
        speaker_set.add(speaker_canon)
        if speaker_canon not in resolved:
            resolved.append(speaker_canon)
    return tuple(resolved), frozenset(speaker_set)


def _view(
    query_text: str,
    norm: NormAnalysis,
    *,
    anchor_us: int,
    vocab: Tuple[str, ...],
    alias_rows: Tuple[AliasRow, ...],
    speaker_canon: Optional[str],
    max_facets: int,
    allow_facets: bool,
    speaker_vocab: Tuple[str, ...] = (),
) -> QueryViewV7:
    """One QueryViewV7 over an existing ``NormAnalysis``.

    Facets reuse this builder so a facet gets the same honest analysis
    as a top-level query — its own extraction/classification/window —
    with ``allow_facets=False`` (decomposition is one level deep).
    """
    extracted = tuple(extract_query_entities(norm, vocab))
    canons = (
        tuple(expand_query(extracted, alias_rows))
        if alias_rows
        else extracted
    )

    # classify sees the *extracted* canons — alias expansion must not
    # manufacture a second entity for the multi_hop rule (R14).
    intent = classify(norm, extracted, norm.identifiers)
    window = resolve_query_window(norm, anchor_us)
    if window is not None:
        intent = replace(intent, window=window)

    def _subview(part: NormAnalysis) -> QueryViewV7:
        return _view(
            part.text,
            part,
            anchor_us=anchor_us,
            vocab=vocab,
            alias_rows=alias_rows,
            speaker_canon=speaker_canon,
            max_facets=max_facets,
            allow_facets=False,
            speaker_vocab=speaker_vocab,
        )

    def _structural_subview(part: NormAnalysis, seed: str) -> QueryViewV7:
        # V8-10.01 — the seed canon is the facet's entity seed: pin it on
        # the sub-view when extraction did not reproduce it (a canon
        # resolved only through the speaker channel is not in the entity
        # vocabulary). Speaker-resolved seeds also bind the facet's
        # speaker_canon so speaker-side features apply to the right
        # participant. The rule_trace marker is the facet-source
        # contract read by ``facet_source``/coverage (V8-20.03).
        fv = _view(
            part.text,
            part,
            anchor_us=anchor_us,
            vocab=vocab,
            alias_rows=alias_rows,
            speaker_canon=(
                seed if seed in speaker_set else speaker_canon
            ),
            max_facets=max_facets,
            allow_facets=False,
            speaker_vocab=speaker_vocab,
        )
        if seed not in fv.entity_canons:
            fv = replace(fv, entity_canons=(seed, *fv.entity_canons))
        return replace(
            fv,
            intent=replace(
                fv.intent,
                rule_trace=(
                    *fv.intent.rule_trace,
                    FACET_SOURCE_STRUCTURAL,
                ),
            ),
        )

    facets: Tuple[QueryViewV7, ...] = ()
    speaker_set: frozenset = frozenset()
    if allow_facets:
        parts = decompose(norm, intent, max_facets=max_facets)
        # Ordering (V8-10.01): comparison primaries keep the
        # intent-decompose shape (its tail-hoist is comparison-specific
        # — reported source="intent"). Every other class prefers the
        # structural canon-binding, which both covers the intent-blind
        # classes (lookup/abstain_likely/...) and replaces the intent
        # split's edge-stripped segments — a coordinated multi_hop
        # question gets predicate-bound facets, not a bare first name.
        if len(parts) > 1 and intent.primary == IntentClass.COMPARISON:
            facets = tuple(_subview(part) for part in parts)
        else:
            resolved, speaker_set = _coordination_canons(
                norm, extracted, speaker_vocab, speaker_canon
            )
            # V85-05.05 — the raw query surface lets the real-canon gate
            # prove capitalization off the NormTerm byte offsets (facets
            # never reach here: sub-views are built allow_facets=False).
            seed_pairs = decompose_structural(
                norm, resolved, max_facets=max_facets,
                surface=query_text,
            )
            if len(seed_pairs) > 1:
                facets = tuple(
                    _structural_subview(part, seed)
                    for part, seed in seed_pairs
                )
            elif len(parts) > 1:
                facets = tuple(_subview(part) for part in parts)

    return QueryViewV7(
        query=query_text,
        norm=norm,
        intent=intent,
        entity_canons=canons,
        speaker_canon=speaker_canon,
        facets=facets,
        query_time_us=anchor_us,
    )


def facet_source(view: QueryViewV7) -> Optional[str]:
    """``coverage.facets.source`` attribution (V8-20.03).

    ``"structural"`` when the view's facets carry the V8-10.01
    coordination marker, ``"intent"`` when facets exist from the
    V7-05.13 decompose path, ``None`` when the view has no facets.
    """
    facets = getattr(view, "facets", None) or ()
    if not facets:
        return None
    for f in facets:
        trace = getattr(getattr(f, "intent", None), "rule_trace", None) or ()
        if FACET_SOURCE_STRUCTURAL in trace:
            return "structural"
    return "intent"


def build_query_view(
    query: str,
    *,
    now_us: int,
    known_canons: _Canons = (),
    speaker_hint: Optional[str] = None,
    canon_fn: Optional[Callable[[str], str]] = None,
    aliases: _Aliases = None,
    max_facets: int = MAX_FACETS,
    known_speakers: _Canons = (),
) -> QueryViewV7:
    """Build the S1 ``QueryViewV7`` for ``query`` (SPEC_V7 §04.2 S1).

    Pure and deterministic: the only time input is ``now_us`` (the
    caller's query/session anchor, recorded as ``query_time_us``).
    ``bytes`` input is decoded strictly — invalid UTF-8 raises
    ``UnicodeDecodeError`` per the ``norm/v2`` step-1 contract;
    unpaired surrogates in ``str`` raise ``UnicodeEncodeError``.

    ``known_speakers`` (V8-10.01) is the scope's speaker-canon
    vocabulary (``units.speaker_canon`` values — same
    iterable-or-callable contract as ``known_canons``). Speaker canons
    resolved through it feed structural coordination detection and
    facet seeding only; they never join ``entity_canons`` or the intent
    classifier, so a coordinated-speaker question keeps its honest
    class.

    Raises ``VerbatimError(VALIDATION)`` for a missing/empty/whitespace
    query or a non-integral ``now_us``.
    """
    if isinstance(query, (bytes, bytearray, memoryview)):
        text = bytes(query).decode("utf-8", "strict")
    else:
        text = str(query if query is not None else "")
    if not text.strip():
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "query must be a non-empty string",
        )
    try:
        anchor_us = int(now_us)
    except (TypeError, ValueError) as exc:
        raise VerbatimError(
            ErrorCode.VALIDATION,
            "now_us must be an integer microsecond timestamp",
        ) from exc

    known = _materialize(known_canons)
    alias_rows = _materialize(aliases)
    speaker = _speaker_canon(speaker_hint, canon_fn)
    speakers = _materialize(known_speakers)

    return _view(
        text,
        analyze(text),
        anchor_us=anchor_us,
        vocab=_extraction_vocab(known, alias_rows),
        alias_rows=alias_rows,
        speaker_canon=speaker,
        max_facets=max(1, int(max_facets)),
        allow_facets=True,
        speaker_vocab=speakers,
    )
