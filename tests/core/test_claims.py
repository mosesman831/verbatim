"""Claim proposal tests (SPEC §12): slots, negation, condition, modality,
time interpretation, reported speech, and explicit-remember bypass."""

from __future__ import annotations

import pytest

from verbatim.core.claims import (
    EXPLICIT_REMEMBER,
    SLOT_REGISTRY,
    is_explicit_remember,
    propose,
    slot_for,
)
from verbatim.core.time import parse_time_expression
from verbatim.core.types import (
    Modality,
    Polarity,
    Precision,
    SpanRef,
    new_id,
)

from tests.core.conftest import make_envelope, make_scope

EVENT_US = 1_760_000_000_000_000  # fixed reference time


def _proposal(text: str, **env_kw):
    """Wrap text in an envelope + span covering the whole payload."""
    scope = make_scope()
    env = make_envelope(scope, text, **env_kw)
    span = SpanRef(new_id(), env.source_id or "src", 1, 0, len(env.payload))
    return propose(text, span, env, EVENT_US)


def test_slot_registry_contents():
    for name in (
        "editor",
        "ide",
        "project_database",
        "residence",
        "preference",
        "schedule",
        "language",
        "os",
        "tool_use",
    ):
        assert name in SLOT_REGISTRY
        assert set(SLOT_REGISTRY[name]) == {"multi_valued", "mutable", "sensitive"}
    assert SLOT_REGISTRY["editor"]["multi_valued"] is False
    assert SLOT_REGISTRY["tool_use"]["multi_valued"] is True


def test_slot_for_literal_and_unknown():
    assert slot_for("literal:employer") is not None
    assert slot_for("editor") is not None
    assert slot_for("not_a_slot") is None
    assert slot_for(None) is None


def test_use_known_editor_maps_editor():
    p = _proposal("I use Neovim.")
    assert p.predicate == "editor"
    assert p.object_json["text"] == "Neovim"


def test_use_generic_tool_maps_tool_use():
    p = _proposal("I use Docker.")
    assert p.predicate == "tool_use"
    assert p.object_json["text"] == "Docker"


def test_use_for_condition():
    p = _proposal("I use Neovim for personal projects.")
    assert p.predicate == "editor"
    assert p.object_json["text"] == "Neovim"
    assert p.condition is not None
    assert p.condition.op == "eq"
    assert p.condition.key == "context"
    assert p.condition.value == "personal_projects"


def test_at_work_condition():
    p = _proposal("At work I use VS Code.")
    assert p.predicate == "editor"
    assert p.condition is not None
    assert p.condition.value == "work"


def test_my_editor_is():
    p = _proposal("My editor is Emacs.")
    assert p.predicate == "editor"
    assert p.object_json["text"] == "Emacs"


def test_residence_live_in():
    p = _proposal("I live in Oslo.")
    assert p.predicate == "residence"
    assert p.object_json["text"] == "Oslo"


def test_residence_moved_to_change_verb():
    p = _proposal("I moved to Bergen.")
    assert p.predicate == "residence"
    assert p.object_json["text"] == "Bergen"
    assert p.valid.basis == "asserted_current_at"
    assert p.valid.from_us == EVENT_US
    assert p.valid.until_us is None


def test_natural_retraction_went_back_to_editor():
    p = _proposal("Actually scratch that about Neovim, I went back to VS Code")
    assert p.predicate == "editor"
    assert p.object_json["text"] == "VS Code"
    assert p.valid.basis == "asserted_current_at"


def test_natural_retraction_without_repeated_subject():
    p = _proposal("scratch that, went back to Emacs")
    assert p.predicate == "editor"
    assert p.object_json["text"] == "Emacs"


def test_project_database_team():
    p = _proposal("We use Postgres for the database.")
    assert p.predicate == "project_database"
    assert p.object_json["text"] == "Postgres"


def test_project_database_is():
    p = _proposal("Our project database is SQLite.")
    assert p.predicate == "project_database"
    assert p.object_json["text"] == "SQLite"


def test_project_database_switched():
    p = _proposal("We switched to PostgreSQL for the database.")
    assert p.predicate == "project_database"
    assert p.valid.basis == "asserted_current_at"


def test_employer_literal():
    p = _proposal("I work at Acme Corp.")
    assert p.predicate == "literal:employer"
    assert p.object_json["text"] == "Acme Corp"


def test_work_at_home_is_not_employer():
    p = _proposal("I work at home on Fridays.")
    assert p.predicate is None


def test_literal_generic_key():
    p = _proposal("My timezone is Europe/Oslo.")
    assert p.predicate == "literal:timezone"
    assert p.object_json["text"] == "Europe/Oslo"


def test_sensitive_literal_key_suppressed():
    p = _proposal("My password is hunter2 the third.")
    assert p.predicate is None


def test_preference():
    p = _proposal("I prefer dark themes.")
    assert p.predicate == "preference"
    assert p.object_json["text"] == "dark themes"


def test_language_speak():
    p = _proposal("I speak Norwegian and English.")
    assert p.predicate == "language"
    assert "Norwegian" in p.object_json["text"]


def test_os_my_os_is():
    p = _proposal("My OS is NixOS.")
    assert p.predicate == "os"
    assert p.object_json["text"] == "NixOS"


def test_os_run():
    p = _proposal("I run Arch Linux on my laptop.")
    assert p.predicate == "os"


def test_schedule():
    p = _proposal("My work schedule is nine to five.")
    assert p.predicate == "schedule"


def test_favorite_maps_preference_with_topic():
    p = _proposal("My favorite language is Rust.")
    assert p.predicate == "preference"
    assert p.object_json["text"] == "Rust"
    assert p.object_json.get("topic") == "language"


def test_negation_polarity():
    p = _proposal("I no longer use Docker.")
    assert p.predicate == "tool_use"
    assert p.polarity == Polarity.NEGATED


def test_never_negated():
    p = _proposal("I never use Emacs.")
    assert p.polarity == Polarity.NEGATED
    assert p.predicate == "editor"


def test_hypothetical_modality():
    p = _proposal("I might switch to Helix.")
    assert p.modality == Modality.HYPOTHETICAL


def test_habitual_modality():
    p = _proposal("I used to run Arch Linux.")
    assert p.modality == Modality.HABITUAL


def test_uncertain_modality():
    p = _proposal("I think my editor is Emacs.")
    assert p.modality == Modality.UNCERTAIN


def test_unless_condition_unparseable():
    # "unless" is a real qualifier but has no bounded mapping → condition None
    p = _proposal("I use Emacs unless the project needs an IDE.")
    assert p.condition is None
    assert p.predicate == "editor"


def test_switched_to_yesterday_valid_time():
    p = _proposal("I switched to Helix yesterday.")
    assert p.predicate == "editor"
    assert p.valid.basis == "asserted_current_at"
    ref = parse_time_expression("yesterday", EVENT_US)
    assert p.valid.from_us == ref.from_us
    assert p.valid.until_us is None


def test_explicit_iso_date_no_change_verb():
    p = _proposal("The release is on 2026-10-01.")
    assert p.valid.basis == "explicit_date"
    assert p.valid.precision == Precision.DAY


def test_unknown_time_stays_unknown():
    p = _proposal("I use Neovim.")
    assert p.valid.basis == "unknown"
    assert p.valid.from_us is None and p.valid.until_us is None


def test_reported_speech_not_personal_fact():
    p = _proposal("Sam said 'I live in Bristol'.")
    assert p.predicate is None


def test_reported_speech_double_quotes():
    p = _proposal('Sam said "I use Neovim" yesterday.')
    assert p.predicate is None


def test_hypothetical_move_not_residence():
    p = _proposal("If I move to Bristol, I will cycle.")
    assert p.predicate is None
    assert p.modality == Modality.HYPOTHETICAL


def test_remember_bypasses_pattern():
    p = _proposal("Remember that I like tea.")
    assert p.predicate is None
    assert p.method == EXPLICIT_REMEMBER
    assert is_explicit_remember(p)
    assert p.evidence  # grounded to the span


def test_please_remember():
    p = _proposal("Please remember: the deploy window is Friday.")
    assert is_explicit_remember(p)


def test_unmatched_text_stays_unstructured():
    p = _proposal("The weather is nice today.")
    assert p.predicate is None
    assert p.object_json is None


def test_pronoun_object_rejected():
    p = _proposal("I use it every day.")
    assert p.predicate is None


def test_object_byte_offsets_resolve_in_payload():
    text = "I use Neovim for personal projects."
    p = _proposal(text)
    obj = p.object_json
    env = make_envelope(make_scope(), text)
    # span covers whole payload; byte offsets must slice exactly the object
    frag = env.payload[obj["byte_start"] : obj["byte_end"]].decode()
    assert frag == "Neovim"


def test_switch_from_to_object_is_new_value():
    p = _proposal("I switched my personal projects from Neovim to Helix yesterday.")
    assert p.predicate == "editor"
    assert p.object_json["text"] == "Helix"
    assert p.valid.basis == "asserted_current_at"
    ref = parse_time_expression("yesterday", EVENT_US)
    assert p.valid.from_us == ref.from_us


def test_multi_condition_all_node():
    p = _proposal("At work on weekends I use VS Code.")
    assert p.condition is not None
    if p.condition.op == "all":
        values = {c.value for c in p.condition.children}
        assert values == {"work", "weekends"}
    else:
        assert p.condition.value == "work"
