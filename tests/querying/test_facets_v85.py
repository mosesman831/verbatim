"""V8.5 real-canon structural-decomposition gate — SPEC_V8_5 V85-05.05.

Structural decomposition (V8-10.01) now fires only on >= 2 *real*
canons. A real canon resolves through the entity or speaker channel
(upstream — ``canons`` is the merged resolved set), is capitalized in
the query's raw surface (``NormTerm`` byte offsets index into the
query's UTF-8 encoding; the caller passes the original text as
``surface``), and is not on the owned function-word stoplist
(:data:`FUNC_WORD_STOPLIST`).

The measured defect this fixes (research pack c4): after the entity
lane was removed, junk canon-like resolutions — ``of``, ``both`` —
fired structural decomposition and regressed multi-hop all@10
0.087 -> 0.056. Without a ``surface`` the capitalization half cannot
be proven, so the gate fails closed (no facets).

All machinery is real: ``norm/v2`` analysis +
``intent_v2.decompose_structural`` + ``query_view.build_query_view`` —
no mocks.
"""

from __future__ import annotations

from verbatim.querying.intent_v2 import (
    FUNC_WORD_STOPLIST,
    decompose_structural,
)
from verbatim.querying.query_view import (
    FACET_SOURCE_STRUCTURAL,
    build_query_view,
    facet_source,
)
from verbatim.text.norm_v2 import analyze

T0 = 1_700_000_000_000_000


def _pairs(text: str, canons, **kw):
    return decompose_structural(analyze(text), tuple(canons), surface=text, **kw)


# ---------------------------------------------------------------------------
# The stoplist — owned, fixed, and contains every spec-named junk token
# ---------------------------------------------------------------------------


def test_stoplist_covers_the_spec_named_function_words():
    """V85-05.05's list owns the measured c4 offenders and the closed
    classes around them — coordinators, cluster markers, auxiliaries,
    articles, prepositions, wh-words."""
    for w in (
        # spec-named offenders
        "of", "both", "each",
        # coordinator joins + markers
        "and", "or", "nor", "vs", "versus", "either", "neither",
        "between",
        # auxiliaries / copula (a capitalized sentence-initial "Will"
        # must not seed a facet either)
        "is", "are", "was", "were", "will", "would", "do", "does",
        "did", "have", "has", "had", "can", "could", "should", "may",
        "might", "must", "be", "am",
        # articles / determiners / wh-words / prepositions
        "the", "a", "an", "this", "that", "these", "those", "all",
        "any", "some", "what", "which", "who", "when", "where", "why",
        "how", "to", "in", "on", "for", "at", "by", "with", "from",
        "as", "well",
    ):
        assert w in FUNC_WORD_STOPLIST, w


def test_junk_canon_like_resolutions_never_seed():
    """The c4 bug shape: ``of``/``both`` genuinely resolve (they sit in
    the canon table) but can never seed decomposition."""
    # direct structural call — stoplist applies before coordination
    assert _pairs("of and both went", ("of", "both")) == ()
    # through the real query view — junk canons still resolve honestly
    # onto entity_canons; they just never decompose.
    view = build_query_view(
        "of and both went", now_us=T0, known_canons=("of", "both")
    )
    assert view.facets == ()
    assert set(view.entity_canons) == {"of", "both"}


def test_each_other_function_word_canons_blocked():
    for canon in ("each", "either", "neither", "the", "is", "well"):
        assert _pairs(
            f"Melanie and {canon} went", ("melanie", canon)
        ) == (), canon


def test_one_real_canon_plus_junk_canon_does_not_decompose():
    """The gate counts REAL canons: one real canon + one function-word
    canon is below the >= 2 bar even when a coordinator joins them."""
    assert _pairs(
        "Melanie and each went home", ("melanie", "each")
    ) == ()


# ---------------------------------------------------------------------------
# Capitalization in the raw query surface
# ---------------------------------------------------------------------------


def test_lowercase_occurrences_are_not_real():
    """Both canons resolve but appear lowercase — the common-word
    reading, not the entity reading. No structural facets."""
    assert _pairs(
        "melanie and caroline went", ("melanie", "caroline")
    ) == ()


def test_mixed_case_partial_capitalization_does_not_decompose():
    """Only one of the two canons appears capitalized -> 1 real canon
    -> no decomposition."""
    assert _pairs(
        "Melanie and caroline went", ("melanie", "caroline")
    ) == ()
    assert _pairs(
        "melanie and Caroline went", ("melanie", "caroline")
    ) == ()


def test_no_surface_fails_closed():
    """``surface=None``: capitalization cannot be proven, so no canon
    is provably real — the gate stays shut even on a well-formed
    coordinated query."""
    text = "Melanie and Caroline went"
    assert decompose_structural(analyze(text), ("melanie", "caroline")) == ()


def test_later_capitalized_occurrence_counts():
    """A canon with one lowercase and one capitalized occurrence is
    real through the capitalized one."""
    pairs = _pairs(
        "melanie said Melanie and Caroline went",
        ("melanie", "caroline"),
    )
    assert len(pairs) == 2
    assert {seed for _n, seed in pairs} == {"melanie", "caroline"}


def test_uncased_surface_is_not_capitalization():
    """Digits/punctuation-first surfaces carry no capitalization
    evidence — the occurrence is not real."""
    assert _pairs(
        "401k and Roth accounts grew", ("401k", "roth")
    ) == ()


# ---------------------------------------------------------------------------
# Real canons still decompose — entities, multi-word, speakers
# ---------------------------------------------------------------------------


def test_two_capitalized_resolved_canons_decompose():
    pairs = _pairs(
        "Melanie and Caroline went home", ("melanie", "caroline")
    )
    assert len(pairs) == 2
    assert [seed for _n, seed in pairs] == ["melanie", "caroline"]


def test_multi_word_canons_decompose():
    """Multi-word canons bind as one span; capitalization is judged on
    the occurrence's leading surface character."""
    pairs = _pairs(
        "Alice Chen and Bob Marley met", ("alice chen", "bob marley")
    )
    assert len(pairs) == 2
    assert {seed for _n, seed in pairs} == {"alice chen", "bob marley"}
    # lowercase multi-word occurrences are not real
    assert _pairs(
        "alice chen and bob marley met", ("alice chen", "bob marley")
    ) == ()


def test_coordinator_inside_a_canon_is_still_not_a_site():
    """"Rock and Roll" — the coordinator inside the canon span never
    splits it; a second canon then coordinates across the gap."""
    pairs = _pairs(
        "Rock and Roll and Pop went", ("rock and roll", "pop")
    )
    assert len(pairs) == 2
    assert {seed for _n, seed in pairs} == {"rock and roll", "pop"}


def test_speaker_channel_canons_respect_the_gate():
    """Speaker-resolved canons are real canons — capitalized speaker
    names decompose; lowercase ones do not."""
    view = build_query_view(
        "What did Melanie and Caroline each say?",
        now_us=T0,
        known_canons=(),
        known_speakers=("melanie", "caroline"),
    )
    assert len(view.facets) == 2
    assert facet_source(view) == "structural"
    for f in view.facets:
        assert FACET_SOURCE_STRUCTURAL in f.intent.rule_trace

    lower = build_query_view(
        "what did melanie and caroline each say",
        now_us=T0,
        known_canons=(),
        known_speakers=("melanie", "caroline"),
    )
    # nothing structurally decomposes — any facets would have to come
    # from the V7-05.13 intent path, which this query does not reach
    # (speaker canons never enter the classifier).
    assert lower.facets == ()


def test_gate_end_to_end_marks_structural_source():
    view = build_query_view(
        "How did Melanie and Caroline each spend the summer?",
        now_us=T0,
        known_canons=("melanie", "caroline"),
    )
    assert len(view.facets) == 2
    assert facet_source(view) == "structural"
    seeds = [f.entity_canons[0] for f in view.facets]
    assert seeds == ["melanie", "caroline"]


def test_malformed_surface_types_fail_closed():
    """A non-str/bytes surface carries no provable capitalization."""
    text = "Melanie and Caroline went"
    for bad in (None, 42, 3.5, ["Melanie"]):
        assert (
            decompose_structural(
                analyze(text), ("melanie", "caroline"), surface=bad
            )
            == ()
        ), bad
