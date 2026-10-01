"""query_analysis/v1 classification tests (SPEC_V5 §31.1, V5-31.01/02,
E83): deterministic classes for identifier / entity / temporal /
preference / procedural / factual / likely-no-answer queries."""

from __future__ import annotations

import pytest

from verbatim.memory.types import QUERY_ANALYSIS_VERSION, QueryClass
from verbatim.querying import analyze
from verbatim.querying.analyze import QueryAnalysis


# ---------------------------------------------------------------------
# identifier-bearing
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_ident",
    [
        ("what is deploy-v2", "deploy-v2"),
        ("status of https://example.com/x", "https://example.com/x"),
        ("who owns alice@example.com", "alice@example.com"),
        ("where is src/main.py defined", "src/main.py"),
        ("any update on PROJ-123", "PROJ-123"),
    ],
)
def test_identifier_bearing(query, expected_ident):
    a = analyze(query)
    assert QueryClass.IDENTIFIER.value in a.classes
    assert a.primary == QueryClass.IDENTIFIER.value
    values = [v for _k, v in a.identifiers]
    assert expected_ident in values


# ---------------------------------------------------------------------
# entity-centric
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "what do we know about Postgres",
        "tell me about Alice",
        "who is Ruth",
        "Postgres",
    ],
)
def test_entity_centric(query):
    a = analyze(query)
    assert QueryClass.ENTITY.value in a.classes
    assert a.entities, query


def test_entity_frame_without_extraction():
    # an entity-shaped question even when extraction finds no capitalized
    # span ("what do we know about the cache layer")
    a = analyze("what do we know about the cache layer")
    assert QueryClass.ENTITY.value in a.classes


# ---------------------------------------------------------------------
# temporal intent → bitemporal hint
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,hint",
    [
        ("when did we switch to postgres", "timeline"),
        ("history of the deploy script", "timeline"),
        ("what was the editor before the switch", "timeline"),
        ("what is the current editor", "current"),
        ("latest deploy status", "current"),
        ("as of March, what was the deploy command", "known_at"),
        ("what did we know at the time", "known_at"),
    ],
)
def test_temporal_hints(query, hint):
    a = analyze(query)
    assert QueryClass.TEMPORAL.value in a.classes
    assert a.primary == QueryClass.TEMPORAL.value
    assert a.temporal_hint == hint
    assert a.temporal_markers


def test_temporal_plus_identifier_is_multilabel():
    a = analyze("when was deploy-v2 released")
    assert QueryClass.TEMPORAL.value in a.classes
    assert QueryClass.IDENTIFIER.value in a.classes
    # temporal wins primary: bitemporal intent drives delivery semantics
    assert a.primary == QueryClass.TEMPORAL.value
    assert a.temporal_hint == "timeline"


def test_no_temporal_hint_for_atemporal():
    a = analyze("database schema")
    assert a.temporal_hint is None
    assert QueryClass.TEMPORAL.value not in a.classes


# ---------------------------------------------------------------------
# preference / decision
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "what is my favorite editor",
        "which database do I prefer",
        "what did we decide about the queue",
        "what was the decision on the deploy tool",
    ],
)
def test_preference_decision(query):
    a = analyze(query)
    assert QueryClass.PREFERENCE.value in a.classes
    assert a.primary == QueryClass.PREFERENCE.value


# ---------------------------------------------------------------------
# procedural
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "how did we fix the deploy pipeline",
        "how to rotate the api keys",
        "what are the steps to onboard",
    ],
)
def test_procedural(query):
    a = analyze(query)
    assert QueryClass.PROCEDURAL.value in a.classes


# ---------------------------------------------------------------------
# factual default + likely no-answer
# ---------------------------------------------------------------------


def test_factual_default():
    a = analyze("database schema")
    assert a.classes == (QueryClass.FACTUAL.value,)
    assert a.primary == QueryClass.FACTUAL.value


@pytest.mark.parametrize("query", ["", "   ", "the a an of to", "?!.."])
def test_no_answer_empty(query):
    a = analyze(query)
    assert a.primary == QueryClass.NO_ANSWER_LIKELY.value
    assert "no_signal" in a.warnings


@pytest.mark.parametrize(
    "query",
    [
        "what did I never tell you",
        "what haven't we mentioned about the roadmap",
        "show me something we never recorded",
    ],
)
def test_no_answer_impossible_markers(query):
    a = analyze(query)
    assert QueryClass.NO_ANSWER_LIKELY.value in a.classes
    # impossible markers dominate: honest insufficient beats least-bad hit
    assert a.primary == QueryClass.NO_ANSWER_LIKELY.value


# ---------------------------------------------------------------------
# determinism + versioning
# ---------------------------------------------------------------------


def test_deterministic():
    a1 = analyze("when did we decide to use Postgres at work")
    a2 = analyze("when did we decide to use Postgres at work")
    assert a1 == a2
    assert a1.to_dict() == a2.to_dict()


def test_version_pinned_and_serializable():
    a = analyze("what is deploy-v2")
    assert a.version == QUERY_ANALYSIS_VERSION == "query_analysis/v1"
    d = a.to_dict()
    assert d["version"] == "query_analysis/v1"
    assert d["classes"] == list(a.classes)


def test_meta_only_query_is_no_answer():
    # pure meta-verb chatter carries no retrievable signal either
    a = analyze("tell me something")
    assert a.primary == QueryClass.NO_ANSWER_LIKELY.value


# ---------------------------------------------------------------------
# enrichment seam (contracts §7): the resolved entry points behave
# identically whether the landed ``verbatim.enrichment`` or the local
# fallback produced them.
# ---------------------------------------------------------------------


def test_enrichment_seam_shapes():
    import importlib

    mod = importlib.import_module("verbatim.querying.analyze")

    idents = mod.extract_identifiers("the deploy command is deploy-v2")
    assert any(m.value == "deploy-v2" for m in idents)
    assert all(m.kind and m.value for m in idents)

    entities = mod.extract_entities("we moved Postgres to the rack")
    assert any(m.value.lower() == "postgres" for m in entities)

    assert mod.text_polarity("We no longer use Postgres.") == "negated"
    assert mod.text_polarity("Maybe we use Postgres.") in {
        "hedged",
        "affirmative",
    }
    assert mod.classify_type("My favorite color is blue.") == "preference"

    tv = mod.parse_temporal("released on 2024-03-01", None)
    assert tv.event_at is None or "2024" in str(tv.event_at)


def test_local_fallbacks_match_contract_shape():
    # the interim implementations used when verbatim.enrichment is absent
    import importlib

    mod = importlib.import_module("verbatim.querying.analyze")

    idents = mod._fallback_extract_identifiers("see PROJ-12 and v1.2.3")
    assert {m.kind for m in idents} >= {"ticket", "version"}
    tv = mod._fallback_parse_temporal("met on 2024-03-01", None)
    assert tv.precision == "day" and tv.event_at == "2024-03-01"
    assert mod._fallback_polarity('docs say "use deploy-v2"') == "quoted"
    assert mod._fallback_classify_type("I like tea") == "preference"
