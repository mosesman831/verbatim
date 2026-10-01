"""Tests for ``query_view/v7`` — the S1 query-analysis builder.

``build_query_view`` is a pure composition over landed wave-A analyzers
(``norm/v2``, ``entities/v2``, ``intent/v2``, ``temporal/v2``), so these
tests pin the *wiring* contract: field placement (window lives on
``IntentResult``), recursion shape for facets, callable ``known_canons``
semantics, speaker canonization, alias expansion, validation errors, and
byte-identical determinism.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict

import pytest

from verbatim.core.types import ErrorCode, VerbatimError
from verbatim.core.types_v7 import (
    AliasMethod,
    AliasRow,
    AliasState,
    IntentClass,
    IntervalUs,
    NormAnalysis,
    QueryViewV7,
)
from verbatim.querying.query_view import (
    BUILDER_ID,
    MAX_FACETS,
    build_query_view,
)

# Fixed anchor: 2025-10-09T00:00:00Z in microseconds — "last week"
# resolves to the prior Mon–Sun week. No wall clock anywhere.
NOW_US = 1_760_000_000_000_000

KNOWN = ("alice chen", "bob marley", "rome", "lisbon", "project falcon")


def _alias(canon_key: str, alias_key: str,
           state: AliasState = AliasState.ACTIVE) -> AliasRow:
    return AliasRow(
        scope_id="s",
        canon=canon_key,
        alias_canon=alias_key,
        rule_id="A5",
        evidence_count=1,
        method=AliasMethod.CALLER,
        state=state,
        generation=1,
    )


# ---------------------------------------------------------------------------
# Shape / wiring
# ---------------------------------------------------------------------------


def test_returns_query_view_type():
    qv = build_query_view("where does Alice Chen work", now_us=NOW_US,
                          known_canons=KNOWN)
    assert isinstance(qv, QueryViewV7)
    assert qv.query == "where does Alice Chen work"
    assert isinstance(qv.norm, NormAnalysis)
    assert qv.norm.analyzer_id == "norm/v2"


def test_builder_id_pinned():
    assert BUILDER_ID == "query_view/v7"


def test_query_time_us_records_now():
    qv = build_query_view("hello there", now_us=NOW_US)
    assert qv.query_time_us == NOW_US


def test_now_us_keyword_only():
    with pytest.raises(TypeError):
        build_query_view("hello", NOW_US)  # noqa: E501 - positional now_us rejected


def test_lookup_query_has_no_window():
    qv = build_query_view("where does Alice Chen work", now_us=NOW_US,
                          known_canons=KNOWN)
    assert qv.intent.primary == IntentClass.LOOKUP
    assert qv.intent.window is None


# ---------------------------------------------------------------------------
# Identifier channel
# ---------------------------------------------------------------------------


def test_identifier_query_intent_is_identifier():
    qv = build_query_view("ABC-1234", now_us=NOW_US)
    assert qv.intent.primary == IntentClass.IDENTIFIER


def test_identifier_terms_kept_on_identifier_channel():
    qv = build_query_view("what is the status of TICKET-99", now_us=NOW_US)
    assert qv.intent.primary == IntentClass.IDENTIFIER
    id_terms = [t.term for t in qv.norm.identifiers]
    assert "TICKET-99" in id_terms
    # identifiers never fold into the lexical term channel
    assert all(
        t.term != "TICKET-99"
        for t in qv.norm.terms
        if t.channel in ("text", "stem")
    )


def test_identifier_byte_offsets_round_trip():
    q = "see PROJ-42 for details"
    qv = build_query_view(q, now_us=NOW_US)
    ident = [t for t in qv.norm.identifiers if t.term == "PROJ-42"]
    assert len(ident) == 1
    raw = q.encode("utf-8")
    assert raw[ident[0].byte_start:ident[0].byte_end] == b"PROJ-42"


def test_url_identifier_query():
    qv = build_query_view(
        "did we share https://example.com/spec.pdf", now_us=NOW_US)
    assert qv.intent.primary == IntentClass.IDENTIFIER
    assert any(
        "example.com" in t.term for t in qv.norm.identifiers
    )


# ---------------------------------------------------------------------------
# Temporal window
# ---------------------------------------------------------------------------


def test_last_week_query_gets_bounded_window():
    qv = build_query_view("what did we discuss last week", now_us=NOW_US)
    assert qv.intent.primary == IntentClass.TEMPORAL_RANGE
    win = qv.intent.window
    assert isinstance(win, IntervalUs)
    assert win.known  # both bounds resolved
    assert win.start_us < win.end_us
    # the window must be anchored to now_us, not a wall clock
    assert win.end_us <= NOW_US + 7 * 86_400_000_000
    assert win.anchor_us == NOW_US


def test_window_anchored_to_passed_now():
    earlier = NOW_US - 30 * 86_400_000_000  # 30 days earlier
    qv_now = build_query_view("what did we discuss last week",
                              now_us=NOW_US)
    qv_then = build_query_view("what did we discuss last week",
                               now_us=earlier)
    assert qv_then.intent.window.end_us < qv_now.intent.window.end_us


def test_window_attached_to_any_intent_class():
    # a count question with a temporal modifier keeps its class AND gets
    # the window (§32.14: temporal windows attach to any class)
    qv = build_query_view("how many meetings did we have last week",
                          now_us=NOW_US)
    assert qv.intent.primary == IntentClass.COUNT_AGGREGATE
    assert qv.intent.window is not None
    assert qv.intent.window.known


def test_window_lives_on_intent_not_view():
    qv = build_query_view("what happened yesterday", now_us=NOW_US)
    assert qv.intent.window is not None
    assert not hasattr(qv, "window") or getattr(qv, "window", None) is None


def test_abstain_likely_carries_no_fabricated_window():
    qv = build_query_view("did i ever mention the deadline",
                          now_us=NOW_US)
    assert qv.intent.primary == IntentClass.ABSTAIN_LIKELY
    assert qv.intent.window is None


def test_function_only_query_abstains_no_window():
    qv = build_query_view("what is it", now_us=NOW_US)
    assert qv.intent.primary == IntentClass.ABSTAIN_LIKELY
    assert qv.intent.window is None


# ---------------------------------------------------------------------------
# Entity canons
# ---------------------------------------------------------------------------


def test_known_canon_matched_without_capitalization():
    qv = build_query_view("where does alice chen work",
                          now_us=NOW_US, known_canons=KNOWN)
    assert "alice chen" in qv.entity_canons


def test_multiword_canon_beats_single_token():
    qv = build_query_view("what is project falcon status",
                          now_us=NOW_US, known_canons=KNOWN)
    assert "project falcon" in qv.entity_canons


def test_capitalized_unknown_name_degrades_honestly():
    # norm.text is the folded projection, so extract_query_entities'
    # capitalized-run signal is inert (documented degradation in
    # entities_v2): an out-of-vocabulary capitalized name yields no
    # canon, but its folded lexical terms remain for the text lanes.
    qv = build_query_view("where does Zephyr Qilan live", now_us=NOW_US,
                          known_canons=KNOWN)
    assert "zephyr qilan" not in qv.entity_canons
    assert any(
        t.term == "zephyr" and t.channel == "text" for t in qv.norm.terms
    )
    assert any(
        t.term == "qilan" and t.channel == "text" for t in qv.norm.terms
    )
    # and crucially no fabricated second canon flips the intent
    assert qv.intent.primary == IntentClass.LOOKUP


def test_unknown_lowercase_entity_degrades_to_terms():
    # "xyzzy" is neither known nor capitalized -> no entity canon, but
    # the lexical terms remain so lanes still have signal
    qv = build_query_view("tell me about xyzzy", now_us=NOW_US,
                          known_canons=KNOWN)
    assert "xyzzy" not in qv.entity_canons
    assert any(
        t.term == "xyzzy" and t.channel == "text" for t in qv.norm.terms
    )
    assert qv.intent.primary == IntentClass.LOOKUP


def test_known_canons_accepts_generator():
    qv = build_query_view("where does alice chen work", now_us=NOW_US,
                          known_canons=(c for c in KNOWN))
    assert "alice chen" in qv.entity_canons


def test_callable_known_canons_invoked():
    calls = []

    def lookup():
        calls.append(1)
        return KNOWN

    qv = build_query_view("where does alice chen work", now_us=NOW_US,
                          known_canons=lookup)
    assert calls == [1]
    assert "alice chen" in qv.entity_canons


def test_callable_known_canons_never_cached():
    state = {"canons": ["alice chen"]}

    def lookup():
        return list(state["canons"])

    q1 = build_query_view("alice chen or bob marley", now_us=NOW_US,
                          known_canons=lookup)
    state["canons"] = ["alice chen", "bob marley"]
    q2 = build_query_view("alice chen or bob marley", now_us=NOW_US,
                          known_canons=lookup)
    # second call reflects the updated vocabulary — no caching
    assert "bob marley" in q2.entity_canons
    assert "bob marley" not in q1.entity_canons


def test_empty_known_canons_default():
    qv = build_query_view("where does alice chen work", now_us=NOW_US)
    # no vocabulary -> no canons (folded projection => cap-runs inert),
    # but the terms remain so the query still retrieves lexically
    assert qv.entity_canons == ()
    assert any(t.term == "alice" for t in qv.norm.terms)


# ---------------------------------------------------------------------------
# Alias expansion (V7-08.04)
# ---------------------------------------------------------------------------


def test_alias_expansion_adds_canonical():
    aliases = [_alias("melanie park", "mel")]
    qv = build_query_view("where is Mel", now_us=NOW_US,
                          known_canons=KNOWN, aliases=aliases)
    assert "mel" in qv.entity_canons
    assert "melanie park" in qv.entity_canons


def test_alias_expansion_does_not_fabricate_multi_hop():
    # "Mel" + its expansion to "melanie park" is ONE entity — classify
    # must see the extracted canons, not the expanded set.
    aliases = [_alias("melanie park", "mel")]
    qv = build_query_view("what did Mel say", now_us=NOW_US,
                          known_canons=KNOWN, aliases=aliases)
    assert qv.intent.primary != IntentClass.MULTI_HOP
    assert len(qv.entity_canons) == 2  # expansion still on the view


def test_candidate_alias_never_expands():
    # candidate rows neither expand nor join the extraction vocabulary —
    # an unreviewed merge cannot widen a query (V7-08.04)
    aliases = [_alias("melanie park", "mel", state=AliasState.CANDIDATE)]
    qv = build_query_view("where is Mel", now_us=NOW_US,
                          known_canons=KNOWN, aliases=aliases)
    assert "melanie park" not in qv.entity_canons
    assert "mel" not in qv.entity_canons
    # lexical signal survives
    assert any(t.term == "mel" for t in qv.norm.terms)


def test_active_alias_surface_is_extraction_vocabulary():
    # the alias surface is recognized even when it is not itself a known
    # canon — required for the alias->canon direction to be reachable
    aliases = [_alias("melanie park", "mel")]
    qv = build_query_view("where is mel", now_us=NOW_US,
                          known_canons=KNOWN, aliases=aliases)
    assert "mel" in qv.entity_canons
    assert "melanie park" in qv.entity_canons


def test_alias_reverse_direction_canonical_to_alias():
    # querying the canonical reaches the alias's postings too
    aliases = [_alias("melanie park", "mel")]
    qv = build_query_view("what did melanie park say", now_us=NOW_US,
                          known_canons=KNOWN + ("melanie park",),
                          aliases=aliases)
    assert "melanie park" in qv.entity_canons
    assert "mel" in qv.entity_canons


def test_no_aliases_no_expansion():
    qv = build_query_view("where is Mel", now_us=NOW_US,
                          known_canons=KNOWN)
    assert "melanie park" not in qv.entity_canons


# ---------------------------------------------------------------------------
# Facet decomposition (V7-05.13)
# ---------------------------------------------------------------------------


def test_comparison_decomposes_into_facets():
    qv = build_query_view("compare Rome and Lisbon for a honeymoon",
                          now_us=NOW_US, known_canons=KNOWN)
    assert qv.intent.primary == IntentClass.COMPARISON
    assert len(qv.facets) >= 2
    assert len(qv.facets) <= MAX_FACETS
    facet_texts = [f.norm.text for f in qv.facets]
    assert any("rome" in t for t in facet_texts)
    assert any("lisbon" in t for t in facet_texts)


def test_comparison_facets_carry_shared_tail():
    qv = build_query_view("compare Rome and Lisbon for a honeymoon",
                          now_us=NOW_US, known_canons=KNOWN)
    # "for a honeymoon" is hoisted into every facet as shared context
    for f in qv.facets:
        assert "honeymoon" in f.norm.text


def test_facets_are_full_query_views():
    qv = build_query_view("compare Rome and Lisbon for a honeymoon",
                          now_us=NOW_US, known_canons=KNOWN)
    for f in qv.facets:
        assert isinstance(f, QueryViewV7)
        assert f.facets == ()           # one level deep only
        assert f.query_time_us == NOW_US
        assert isinstance(f.norm, NormAnalysis)
        assert f.intent.primary is not None


def test_facet_entity_extraction_against_known():
    qv = build_query_view("compare Rome and Lisbon", now_us=NOW_US,
                          known_canons=KNOWN)
    by_text = {f.norm.text: f for f in qv.facets}
    assert any("rome" in f.entity_canons for f in qv.facets)
    assert any("lisbon" in f.entity_canons for f in qv.facets)


def test_multi_hop_two_entities_decomposes():
    qv = build_query_view(
        "what connects Alice Chen and Bob Marley",
        now_us=NOW_US, known_canons=KNOWN)
    assert qv.intent.primary == IntentClass.MULTI_HOP
    assert len(qv.facets) >= 2
    texts = [f.norm.text for f in qv.facets]
    assert any("alice chen" in t for t in texts)
    assert any("bob marley" in t for t in texts)


def test_multi_hop_without_split_token_yields_no_facets():
    # multi_hop intent from two entity canons, but no conjunction or
    # relative marker to split on -> decompose returns the (norm,)
    # singleton -> honest empty facets, never a fabricated split
    qv = build_query_view(
        "what did Alice Chen tell Bob Marley",
        now_us=NOW_US, known_canons=KNOWN)
    assert qv.intent.primary == IntentClass.MULTI_HOP
    assert qv.facets == ()


def test_non_decomposable_intent_has_no_facets():
    qv = build_query_view("where does Alice Chen work", now_us=NOW_US,
                          known_canons=KNOWN)
    assert qv.facets == ()


def test_max_facets_respected():
    qv = build_query_view(
        "compare Rome and Lisbon and Madrid and Porto",
        now_us=NOW_US,
        known_canons=KNOWN + ("madrid", "porto"),
        max_facets=2)
    assert len(qv.facets) <= 2


def test_facets_inherit_speaker_canon():
    qv = build_query_view("compare Rome and Lisbon", now_us=NOW_US,
                          known_canons=KNOWN, speaker_hint="Caroline")
    assert qv.speaker_canon == "caroline"
    for f in qv.facets:
        assert f.speaker_canon == "caroline"


# ---------------------------------------------------------------------------
# Speaker hint
# ---------------------------------------------------------------------------


def test_speaker_hint_canonicalized():
    qv = build_query_view("what did we decide", now_us=NOW_US,
                          speaker_hint="Caroline's")
    assert qv.speaker_canon == "caroline"


def test_speaker_hint_absent_is_none():
    qv = build_query_view("what did we decide", now_us=NOW_US)
    assert qv.speaker_canon is None


def test_speaker_hint_blank_is_none():
    qv = build_query_view("what did we decide", now_us=NOW_US,
                          speaker_hint="   ")
    assert qv.speaker_canon is None


def test_speaker_hint_custom_canon_fn():
    seen = []

    def loud_canon(surface: str) -> str:
        seen.append(surface)
        return f"speaker:{surface.strip().lower()}"

    qv = build_query_view("what did we decide", now_us=NOW_US,
                          speaker_hint="Caroline", canon_fn=loud_canon)
    assert seen == ["Caroline"]
    assert qv.speaker_canon == "speaker:caroline"


def test_canon_fn_returning_empty_is_none():
    qv = build_query_view("what did we decide", now_us=NOW_US,
                          speaker_hint="Caroline", canon_fn=lambda s: "")
    assert qv.speaker_canon is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_empty_query_raises_validation():
    with pytest.raises(VerbatimError) as ei:
        build_query_view("", now_us=NOW_US)
    assert ei.value.code == ErrorCode.VALIDATION


def test_whitespace_query_raises_validation():
    for q in ("   ", "\n\t  ", " \u00a0 "):
        with pytest.raises(VerbatimError) as ei:
            build_query_view(q, now_us=NOW_US)
        assert ei.value.code == ErrorCode.VALIDATION


def test_none_query_raises_validation():
    with pytest.raises(VerbatimError) as ei:
        build_query_view(None, now_us=NOW_US)
    assert ei.value.code == ErrorCode.VALIDATION


def test_bad_now_us_raises_validation():
    with pytest.raises(VerbatimError) as ei:
        build_query_view("hello", now_us=None)
    assert ei.value.code == ErrorCode.VALIDATION


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_byte_identical():
    kwargs = dict(
        now_us=NOW_US,
        known_canons=KNOWN,
        speaker_hint="Caroline",
        aliases=[_alias("melanie park", "mel")],
    )
    a = build_query_view("compare Rome and Lisbon for a honeymoon",
                         **kwargs)
    b = build_query_view("compare Rome and Lisbon for a honeymoon",
                         **kwargs)
    assert a == b
    assert asdict(a) == asdict(b)
    assert pickle.dumps(a) == pickle.dumps(b)


def test_determinism_with_callable_canons():
    def lookup():
        return KNOWN

    a = build_query_view("what did we discuss last week",
                         now_us=NOW_US, known_canons=lookup)
    b = build_query_view("what did we discuss last week",
                         now_us=NOW_US, known_canons=lookup)
    assert a == b


def test_bytes_query_accepted():
    qv = build_query_view("where does Alice Chen work".encode("utf-8"),
                          now_us=NOW_US, known_canons=KNOWN)
    assert qv.query == "where does Alice Chen work"
    assert "alice chen" in qv.entity_canons


def test_frozen_view_immutable():
    qv = build_query_view("hello", now_us=NOW_US)
    with pytest.raises(Exception):
        qv.query = "mutated"  # frozen dataclass rejects assignment
